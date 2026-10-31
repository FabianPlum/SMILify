"""Keeping DistributedDataParallel replicas in sync when the training step is a custom method.

DDP all-reduces gradients only in a backward pass that follows a call to the DDP wrapper itself:
``DistributedDataParallel.forward`` prepares the reducer, and without that call the gradient hooks do nothing.
Calling a method on ``ddp_model.module`` (``predict_from_batch``, ``compute_batch_loss``, ...) bypasses it, and every
rank then silently trains its own replica. ``ddp_forward`` runs any training-step function as the DDP forward pass
instead, the "forward redirection" used by Lightning Fabric.

Once gradients are all-reduced, every rank must run the same sequence of collectives: a rank that skips a backward pass
leaves the others waiting (or pairs their all-reduces with the wrong step). ``zero_loss`` gives a batch without a
prediction a backward pass that touches every parameter, and ``reaches_parameters`` finds a loss whose backward pass
would touch none; ``any_rank`` makes a rank-local decision (step or skip) the same on every rank.
``replica_divergence`` checks that the replicas stayed identical.
"""

from collections import deque
from typing import Any, Callable, List, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class _OpaqueInputs:
    """A training step's arguments, passed through DDP.forward untouched. DDP rebuilds nested lists, tuples and dicts
    and copies every tensor in them to its device before calling the module; an object of an unknown type is passed
    as it is."""

    __slots__ = ("args", "kwargs")

    def __init__(self, args: tuple, kwargs: dict):
        self.args = args
        self.kwargs = kwargs


def ddp_forward(model: torch.nn.Module, step_fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Return ``step_fn(module, *args, **kwargs)``. If ``model`` is a DDP wrapper, the call runs as its forward pass
    (``module`` = ``model.module``), so the following backward pass all-reduces gradients (and is skipped inside
    ``model.no_sync()``). Otherwise ``step_fn(model, ...)`` is called directly.

    Only tensors that the backward pass starts from may require grad in the return value: with
    ``find_unused_parameters=True`` DDP searches the autograd graph from every returned tensor, and a parameter reachable
    only from a tensor that is not backpropagated would be waited for. Return detached logging values.
    """
    if not isinstance(model, DistributedDataParallel):
        return step_fn(model, *args, **kwargs)
    module = model.module
    shadowed = module.__dict__.get("forward")  # normally None: forward is the class method

    def restore():
        if shadowed is None:
            module.__dict__.pop("forward", None)
        else:
            module.forward = shadowed

    def redirected(inputs: _OpaqueInputs):
        restore()  # a nested module(...) call inside step_fn reaches the real forward
        return step_fn(module, *inputs.args, **inputs.kwargs)

    module.forward = redirected
    try:
        return model(_OpaqueInputs(args, kwargs))
    finally:
        restore()


def zero_loss(module: torch.nn.Module) -> torch.Tensor:
    """A loss of exactly 0 that depends on every trainable parameter: its backward pass gives zero gradients and takes
    part in DDP's all-reduce, for a batch that produced no prediction on this rank. Zero-size parameters (e.g. a head
    with no outputs) have no element to depend on and are left out."""
    terms = [p.reshape(-1)[0] for p in module.parameters() if p.requires_grad and p.numel() > 0]
    if not terms:
        raise ValueError("zero_loss: the module has no trainable parameters")
    return torch.stack(terms).sum() * 0.0


def reaches_parameters(loss: torch.Tensor, module: torch.nn.Module) -> bool:
    """True if the backward pass of ``loss`` reaches a trainable parameter of ``module``. A loss made only of the
    placeholder leaves the losses use for missing targets (``torch.tensor(eps, requires_grad=True)``) has no
    ``grad_fn``, or a graph without a parameter: its backward pass fires none of DDP's gradient hooks, so this rank
    would skip the all-reduce the other ranks wait in. Treat it like a missing prediction (``zero_loss``). Call it
    inside the training step: on the output of the DDP forward the ``grad_fn`` is DDP's own."""
    if loss.grad_fn is None:
        return False
    trainable = {id(p) for p in module.parameters() if p.requires_grad}
    seen = {loss.grad_fn}
    queue = deque(seen)
    while queue:  # breadth-first, stopping at the first trainable parameter
        node = queue.popleft()
        variable = getattr(node, "variable", None)  # AccumulateGrad nodes: the leaf tensor they accumulate into
        if variable is not None and id(variable) in trainable:
            return True
        for next_node, _ in node.next_functions:
            if next_node is not None and next_node not in seen:
                seen.add(next_node)
                queue.append(next_node)
    return False


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def any_rank(flag: bool, device: torch.device) -> bool:
    """True if ``flag`` is True on any rank (a collective: every rank must call it). Without a process group, ``flag``."""
    if not is_distributed():
        return bool(flag)
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


@torch.no_grad()
def replica_divergence(model: torch.nn.Module, top: int = 3) -> Tuple[float, List[Tuple[str, float]]]:
    """How far the replicas differ: the largest difference between ranks in any parameter tensor's sum or norm
    (float64), and the ``top`` most different parameters as (name, difference). 0.0 and [] when the replicas are
    identical, as they must be after synchronised updates. A collective: every rank must call it."""
    if not is_distributed():
        return 0.0, []
    named = list(model.named_parameters())
    stats = torch.stack(
        [
            torch.stack(
                (p.detach().sum(dtype=torch.float64), torch.linalg.vector_norm(p.detach(), dtype=torch.float64))
            )
            for _, p in named
        ]
    )
    gathered = [torch.empty_like(stats) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, stats)
    per_param = torch.stack([(g - gathered[0]).abs().amax(dim=1) for g in gathered[1:]]).amax(dim=0)
    order = torch.argsort(per_param, descending=True)[:top]
    worst = [(named[i][0], float(per_param[i])) for i in order.tolist() if per_param[i] > 0]
    return float(per_param.max()), worst


def check_replicas(model: torch.nn.Module, epoch: int, rank: int) -> float:
    """``replica_divergence`` after an epoch, printed on rank 0; returns the largest difference. A collective."""
    divergence, worst = replica_divergence(model)
    if rank == 0:
        if divergence == 0.0:
            print(f"DDP replicas after epoch {epoch}: identical")
        else:
            most = ", ".join(f"{name} {d:.3g}" for name, d in worst)
            print(
                f"DDP replicas after epoch {epoch}: DIFFERENT (max parameter sum/norm difference {divergence:.3g}: {most})"
            )
    return divergence
