"""DDP gradient synchronisation through ddp_forward (smal_fitter/neuralSMIL/ddp_utils.py).

Two CPU processes (gloo) train a small model whose training step is a custom method, as the SMIL regressors' are.
Calling that method on ``ddp_model.module`` skips DDP's forward pass, so gradients are not all-reduced and the replicas
drift apart; routing it through ``ddp_forward`` must give every rank the averaged gradients.
"""

import os
from datetime import timedelta
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from smal_fitter.neuralSMIL.ddp_utils import any_rank, ddp_forward, reaches_parameters, replica_divergence, zero_loss

WORLD = 2


class _Net(nn.Module):
    head_type = "mlp"  # read by the single-view train_epoch

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(4, 3)
        self.b = nn.Linear(3, 1)
        self.unused = nn.Linear(2, 2)  # unused, as the regressors' leftover parameters; sporadic in the "extra" kind

    def forward(self, x):
        return self.b(torch.tanh(self.a(x)))

    def step(self, batch, use_extra=False):
        """A training step with nested inputs that calls the module itself, like predict + loss. use_extra adds a
        parameter that only some batches use, like the camera head of a slot only some samples' views fill."""
        loss = self(batch[0]["x"]).pow(2).sum()
        return loss + self.unused(batch[0]["x"][:, :2]).sum() if use_extra else loss

    def loss(self, batch, kind):
        """step's loss ("real"), the same with the sporadic parameter ("extra"), or one that reaches no parameter, as
        the regressors' losses do when every target is missing: a placeholder leaf ("leaf"), or placeholders summed into
        a graph ("graph")."""
        if kind in ("real", "extra"):
            return self.step(batch, use_extra=kind == "extra")
        eps = torch.tensor(1e-8, requires_grad=True)
        return eps if kind == "leaf" else torch.zeros((), requires_grad=True) + 0.5 * eps

    # The trainers' model interface; the targets name the kind of loss, and "none" gives no prediction
    def predict_from_batch(self, x_data_batch, y_data_batch):
        if y_data_batch == "none":
            return None, None, None
        return {"x": x_data_batch}, y_data_batch, None

    def compute_batch_loss(self, predicted_params, targets, auxiliary_data, return_components=False, loss_weights=None):
        loss = self.loss(predicted_params["x"], targets)
        return loss, {"total": loss}

    def predict_from_multiview_batch(self, x_data_batch, y_data_batch):
        return (None if y_data_batch == "none" else {"x": x_data_batch}), None, None

    def compute_multiview_batch_loss(self, predicted_params, y_data_batch, loss_weights=None, return_components=False):
        return self.compute_batch_loss(predicted_params, y_data_batch, None)


def _grads(module):
    return torch.cat([p.grad.reshape(-1) for p in (module.a.weight, module.a.bias, module.b.weight, module.b.bias)])


def _gather(t):
    out = [torch.empty_like(t) for _ in range(WORLD)]
    dist.all_gather(out, t)
    return out


class _Loader(list):
    """One epoch's (x, y) batches for a trainer's train_epoch; ``batch_idx`` is the batch being trained on."""

    def __iter__(self):
        for self.batch_idx, batch in enumerate(list.__iter__(self)):
            yield batch


def _worker(rank, store_uri, queue, exit_barrier):
    dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=WORLD, timeout=timedelta(seconds=60))
    res = {}
    torch.manual_seed(100 + rank)  # different data per rank
    batch = [{"x": torch.randn(5, 4)}]
    model = DDP(_Net(), find_unused_parameters=True)

    # 1. The bug: a custom method on .module bypasses DDP, gradients stay local
    model.module.step(batch).backward()
    local = _grads(model.module).clone()
    g = _gather(local)
    res["bypass_differs"] = not torch.allclose(g[0], g[1])
    expected_mean = (g[0] + g[1]) / 2
    model.zero_grad(set_to_none=True)

    # 2. ddp_forward: averaged gradients on every rank; inputs passed through untouched
    seen = {}

    def step_fn(module, b):
        seen["same_object"] = b is batch and b[0]["x"] is batch[0]["x"]
        return module.step(b)

    ddp_forward(model, step_fn, batch).backward()
    g = _gather(_grads(model.module))
    res["synced"] = torch.allclose(g[0], g[1]) and torch.allclose(g[0], expected_mean, atol=1e-6)
    res["inputs_untouched"] = seen["same_object"]
    res["forward_restored"] = "forward" not in model.module.__dict__
    model.zero_grad(set_to_none=True)

    # 3. A rank without a prediction backpropagates zero_loss: no hang, the other rank's gradient is averaged with 0
    loss = ddp_forward(model, lambda m, b: m.step(b) if rank == 0 else zero_loss(m), batch)
    loss.backward()
    g = _gather(_grads(model.module))
    rank0_alone = _gather(local)[0] / 2
    res["zero_loss_rank"] = torch.allclose(g[0], g[1]) and torch.allclose(g[0], rank0_alone, atol=1e-6)
    model.zero_grad(set_to_none=True)

    # 4. any_rank agrees on every rank
    res["any_rank"] = any_rank(rank == 1, torch.device("cpu")) and not any_rank(False, torch.device("cpu"))

    # 5. Synchronised steps keep the replicas identical; a local change is detected
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        ddp_forward(model, lambda m, b: m.step(b), batch).backward()
        opt.step()
    res["in_sync_after_steps"] = replica_divergence(model) == (0.0, [])
    if rank == 1:
        with torch.no_grad():
            model.module.b.bias.add_(1e-3)
    divergence, worst = replica_divergence(model)
    res["divergence_detected"] = divergence > 0.0 and worst[0][0] == "module.b.bias"
    dist.barrier()

    # 6. DDP registers only parameters that require grad when it is built: unfreezing needs a new wrapper
    net = _Net()
    net.a.requires_grad_(False)
    frozen = DDP(net, find_unused_parameters=True)
    net.a.requires_grad_(True)
    ddp_forward(frozen, lambda m, b: m.step(b), batch).backward()
    ga = _gather(net.a.weight.grad.reshape(-1).clone())
    res["unfrozen_not_synced_without_rewrap"] = not torch.allclose(ga[0], ga[1])
    net.zero_grad(set_to_none=True)
    rewrapped = DDP(net, find_unused_parameters=True)
    ddp_forward(rewrapped, lambda m, b: m.step(b), batch).backward()
    ga = _gather(net.a.weight.grad.reshape(-1).clone())
    res["unfrozen_synced_after_rewrap"] = torch.allclose(ga[0], ga[1])

    # 7. A loss that reaches no parameter fires no gradient hook, so that rank would skip the all-reduce. Both trainers'
    # _training_step backpropagate zero_loss instead, as for a missing prediction. (Imported in the worker processes:
    # importing the trainers changes CUDA_VISIBLE_DEVICES and socket.getaddrinfo.)
    from smal_fitter.neuralSMIL import train_multiview_regressor, train_smil_regressor

    net = _Net()
    guarded = DDP(net, find_unused_parameters=True)
    res["reaches_parameters"] = reaches_parameters(net.loss(batch, "real"), net) and not any(
        reaches_parameters(net.loss(batch, kind), net) for kind in ("leaf", "graph")
    )
    rank1_alone = _gather(local)[1] / 2
    training_steps = {
        "singleview": lambda m, kind: train_smil_regressor._training_step(m, batch, kind, None),
        "multiview": lambda m, kind: train_multiview_regressor._training_step(m, batch, kind, None, 1)[:2],
    }
    for trainer, training_step in training_steps.items():
        for kind in ("leaf", "graph"):  # on rank 0; rank 1's loss is real
            loss, components = ddp_forward(guarded, training_step, kind if rank == 0 else "real")
            loss.backward()
            g = _gather(_grads(net))
            res[f"{trainer}_{kind}_loss_synced"] = (
                torch.allclose(g[0], g[1])
                and torch.allclose(g[0], rank1_alone, atol=1e-6)
                and (components is None) == (rank == 0)
            )
            guarded.zero_grad(set_to_none=True)

    # 8. Both trainers' train_epoch over 7 batches whose kinds differ per rank. A rank without a prediction, or whose
    # loss reaches no parameter, still joins every all-reduce, and any_rank gives both ranks the same optimizer steps:
    # single-view after each batch with a real loss on some rank, multi-view (2 accumulation steps, clipping) after
    # each window with one. Window [4, 5] uses a parameter only in rank 0's first mini-batch and has a real loss in its
    # last mini-batch on both ranks: with model.no_sync() on the first mini-batch (master before #119) that parameter
    # keeps rank 0's local gradient on torch < 2.4 (pytorch/pytorch#124193) and the replicas drift apart. (In the
    # other windows zero_loss touches every parameter in the synchronised mini-batch, which would hide it.)
    kinds = [
        ["none", "leaf", "real", "graph", "extra", "real", "real"],
        ["graph", "none", "leaf", "leaf", "real", "real", "real"],
    ][rank]
    common = {"device": "cpu", "epoch": 0, "is_distributed": True, "rank": rank}
    mv_config = {"gradient_accumulation_steps": 2, "gradient_clip_norm": 0.5}
    train_epochs = {  # trainer: (train_epoch(model, loader, optimizer), the batches after which both ranks step)
        "singleview": (
            partial(train_smil_regressor.train_epoch, criterion=None, loss_weights={}, **common),
            [2, 4, 5, 6],
        ),
        "multiview": (partial(train_multiview_regressor.train_epoch, training_config=mv_config, **common), [3, 5, 6]),
    }
    for trainer, (train_epoch, expected_steps) in train_epochs.items():
        model = DDP(_Net(), find_unused_parameters=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        loader = _Loader((batch, kind) for kind in kinds)
        steps = []  # the batch after which each optimizer step was taken
        opt.register_step_post_hook(lambda *_: steps.append(loader.batch_idx))
        train_epoch(model, loader, opt)
        all_steps = [None] * WORLD
        dist.all_gather_object(all_steps, steps)
        res[f"{trainer}_train_epoch_steps"] = all_steps == [expected_steps] * WORLD
        res[f"{trainer}_train_epoch_replicas_identical"] = replica_divergence(model) == (0.0, [])

    results = [None] * WORLD
    dist.all_gather_object(results, res)
    if rank == 0:
        queue.put(results)
        queue.close()
        queue.join_thread()
    # Leave without closing the process group: closing a gloo group (destroy_process_group(), or at exit) now and then
    # hangs or aborts one rank. Once both ranks are past the barrier, every collective has completed.
    exit_barrier.wait(timeout=60)
    os._exit(0)


def test_ddp_forward_without_process_group_calls_step_directly():
    net = _Net()
    batch = [{"x": torch.randn(2, 4)}]
    assert torch.equal(ddp_forward(net, lambda m, b: m.step(b), batch), net.step(batch))
    assert zero_loss(net).item() == 0.0
    net.empty = nn.Parameter(torch.empty(0, 3))  # zero-size, like the weight of a head for 0 shape betas
    assert zero_loss(net).item() == 0.0


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_ddp_forward_synchronises_gradients_across_ranks(tmp_path):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    exit_barrier = ctx.Barrier(WORLD)
    store_uri = (tmp_path / "store").as_uri()  # a FileStore: no TCP port to pick (and race for)
    mp.start_processes(_worker, args=(store_uri, queue, exit_barrier), nprocs=WORLD, join=True, start_method="spawn")
    results = queue.get(timeout=10)
    for rank, res in enumerate(results):
        failed = [k for k, ok in res.items() if not ok]
        assert not failed, f"rank {rank}: {failed}"
