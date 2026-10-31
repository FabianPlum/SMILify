"""DDP gradient synchronisation through ddp_forward (smal_fitter/neuralSMIL/ddp_utils.py).

Two CPU processes (gloo) train a small model whose training step is a custom method, as the SMIL regressors' are.
Calling that method on ``ddp_model.module`` skips DDP's forward pass, so gradients are not all-reduced and the replicas
drift apart; routing it through ``ddp_forward`` must give every rank the averaged gradients.
"""

import os
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from smal_fitter.neuralSMIL.ddp_utils import any_rank, ddp_forward, replica_divergence, zero_loss

WORLD = 2


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(4, 3)
        self.b = nn.Linear(3, 1)
        self.unused = nn.Linear(2, 2)  # unused, as the regressors' leftover parameters; sporadic in step(use_extra)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x)))

    def step(self, batch, use_extra=False):
        """A training step with nested inputs that calls the module itself, like predict + loss. use_extra adds a
        parameter that only some batches use, like the camera head of a slot only some samples' views fill."""
        loss = self(batch[0]["x"]).pow(2).sum()
        return loss + self.unused(batch[0]["x"][:, :2]).sum() if use_extra else loss


def _grads(module):
    return torch.cat([p.grad.reshape(-1) for p in (module.a.weight, module.a.bias, module.b.weight, module.b.bias)])


def _gather(t):
    out = [torch.empty_like(t) for _ in range(WORLD)]
    dist.all_gather(out, t)
    return out


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

    # 7. Gradient accumulation over two mini-batches; one parameter is used only by rank 0's first mini-batch. With
    # model.no_sync() on the first one it keeps rank 0's local gradient (the trainers therefore sync every mini-batch).
    def extra_grad(use_no_sync):
        m = DDP(_Net(), find_unused_parameters=True)
        with m.no_sync() if use_no_sync else nullcontext():
            ddp_forward(m, lambda mod, b: mod.step(b, use_extra=rank == 0), batch).backward()
        ddp_forward(m, lambda mod, b: mod.step(b), batch).backward()
        grad = m.module.unused.weight.grad
        return _gather(torch.zeros(2, 2) if grad is None else grad.clone())

    g = extra_grad(use_no_sync=True)
    res["no_sync_leaves_sporadic_parameter_unsynced"] = not torch.allclose(g[0], g[1])
    g = extra_grad(use_no_sync=False)
    res["sync_every_batch_syncs_sporadic_parameter"] = torch.allclose(g[0], g[1]) and bool(g[0].abs().sum() > 0)

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
