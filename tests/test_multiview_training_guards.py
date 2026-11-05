"""Non-finite guards and the cross-rank component reduction of the multi-view trainer (train_multiview_regressor.py).

``_step_if_finite`` must drop a step whose gradients are not finite instead of writing NaN into every weight. Under DDP
the gradients are all-reduced before the step, so one rank's NaN reaches every rank and all of them skip the step.
``_all_reduce_components`` must give every rank the union of the ranks' loss-component keys, also when a rank has none
(e.g. all of its validation batches were non-finite), instead of pairing mismatched collectives and hanging.
"""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

WORLD = 2
TIMEOUT_S = 30  # a mismatched collective fails the test after this instead of hanging


@pytest.fixture
def trainer():
    """The multi-view trainer module. Importing it sets CUDA_DEVICE_ORDER (and CUDA_VISIBLE_DEVICES if unset) and
    patches socket.getaddrinfo for the whole process; put them back after the test (the DDP test imports it only in
    its worker processes)."""
    env = {name: os.environ.get(name) for name in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES")}
    getaddrinfo = socket.getaddrinfo
    from smal_fitter.neuralSMIL import train_multiview_regressor

    yield train_multiview_regressor
    socket.getaddrinfo = getaddrinfo
    for name, value in env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.mark.parametrize("clip_norm", [0.0, 1.0])
def test_step_if_finite_drops_a_non_finite_gradient(trainer, clip_norm):
    torch.manual_seed(0)
    net = nn.Linear(3, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=0.1)
    before = [p.detach().clone() for p in net.parameters()]
    x = torch.randn(4, 3)
    x[0, 0] = float("nan")  # one bad sample: the weight gradient is NaN, the bias gradient stays finite
    net(x).sum().backward()

    assert not trainer._step_if_finite(net, opt, clip_norm)
    assert not opt.state  # the optimizer never stepped
    assert all(torch.equal(p, b) for p, b in zip(net.parameters(), before))
    assert all(p.grad is None for p in net.parameters())  # all of the step's gradients are dropped

    net(torch.randn(4, 3)).sum().backward()  # a finite gradient is clipped and stepped as before
    assert trainer._step_if_finite(net, opt, clip_norm)
    assert not any(torch.equal(p, b) for p, b in zip(net.parameters(), before))


def _worker(rank, store_uri, queue, exit_barrier):
    # Imported here, not in the pytest process (see the trainer fixture), and before the process group's timeout runs
    from smal_fitter.neuralSMIL.train_multiview_regressor import _all_reduce_components, _step_if_finite

    dist.init_process_group(
        "gloo", init_method=store_uri, rank=rank, world_size=WORLD, timeout=timedelta(seconds=TIMEOUT_S)
    )
    cpu = torch.device("cpu")
    expected = ({"keypoint_2d": 3.0, "keypoint_3d": 5.0}, {"keypoint_2d": 2, "keypoint_3d": 1})
    res = {}

    # 1. Rank 1 has no components (e.g. every validation batch non-finite): both ranks get rank 0's
    sums, counts = (dict(expected[0]), dict(expected[1])) if rank == 0 else ({}, {})
    res["empty_rank"] = _all_reduce_components(sums, counts, cpu) == expected

    # 2. A data-dependent term only on rank 0: both ranks get the union of the keys, summed per key
    if rank == 0:
        sums, counts = {"keypoint_2d": 1.0, "keypoint_3d": 5.0}, {"keypoint_2d": 1, "keypoint_3d": 1}
    else:
        sums, counts = {"keypoint_2d": 2.0}, {"keypoint_2d": 1}
    res["different_keys"] = _all_reduce_components(sums, counts, cpu) == expected

    # 3. No rank has components: nothing to reduce
    res["all_empty"] = _all_reduce_components({}, {}, cpu) == ({}, {})

    # 4. A NaN gradient on rank 1 reaches rank 0 through the all-reduce: every rank skips the step and drops it
    torch.manual_seed(100 + rank)  # different data per rank
    model = DDP(nn.Linear(3, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.randn(4, 3)
    if rank == 1:
        x[0, 0] = float("nan")
    model(x).sum().backward()
    res["nan_step_skipped"] = (
        not _step_if_finite(model, opt, clip_norm=1.0)
        and not opt.state
        and all(p.grad is None for p in model.parameters())
        and all(torch.equal(p, b) for p, b in zip(model.parameters(), before))
    )

    results = [None] * WORLD
    dist.all_gather_object(results, res)
    if rank == 0:
        queue.put(results)
        queue.close()
        queue.join_thread()
    # Leave without closing the process group, as in test_ddp_utils.py: closing a gloo group now and then hangs or
    # aborts one rank. Once both ranks are past the barrier, every collective has completed.
    exit_barrier.wait(timeout=TIMEOUT_S)
    os._exit(0)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_ranks_agree_on_component_keys_and_on_skipping_a_nan_step(tmp_path):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    exit_barrier = ctx.Barrier(WORLD)
    store_uri = (tmp_path / "store").as_uri()  # a FileStore: no TCP port to pick (and race for)
    mp.start_processes(_worker, args=(store_uri, queue, exit_barrier), nprocs=WORLD, join=True, start_method="spawn")
    results = queue.get(timeout=10)
    for rank, res in enumerate(results):
        failed = [k for k, ok in res.items() if not ok]
        assert not failed, f"rank {rank}: {failed}"


class _FakeMultiView(nn.Module):
    """Enough of MultiViewSMILImageRegressor for train_epoch: one weight and a loss of factor * w on mini-batch i
    (factor factors[i], default 1), so the step's gradient norm is |factor|. The loss is NaN on the mini-batches in
    `bad`, so their gradients are NaN and _step_if_finite skips the step."""

    def __init__(self, bad=(), factors=None):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.bad, self.factors = set(bad), factors or {}

    def predict_from_multiview_batch(self, x_data_batch, y_data_batch):
        return {"batch": x_data_batch}, None, None

    def compute_multiview_batch_loss(self, predicted, y_data_batch, loss_weights=None, return_components=True):
        i = predicted["batch"]
        loss = self.w.sum() * (float("nan") if i in self.bad else self.factors.get(i, 1.0))
        return loss, {"keypoint_2d": loss.detach()}


def _epoch(trainer, bad=(), factors=None, n=10, skip_streak=0, rank=0):
    model = _FakeMultiView(bad, factors)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    loader = [(i, None) for i in range(n)]  # mini-batch i is "x_data_batch" i
    config = {"gradient_clip_norm": 1.0, "gradient_accumulation_steps": 1, "loss_weights": {}}
    metrics = trainer.train_epoch(model, loader, optimizer, "cpu", 0, config, rank=rank, skip_streak=skip_streak)
    return metrics, optimizer


def test_a_diverged_run_stops_after_the_consecutive_skip_limit(trainer, monkeypatch):
    assert trainer.MAX_CONSECUTIVE_SKIPPED_STEPS >= 20  # far above the isolated skips that healthy runs survive
    monkeypatch.setattr(trainer, "MAX_CONSECUTIVE_SKIPPED_STEPS", 3)
    # Every step skipped: every rank stops at the third mini-batch, through the per-batch error handling
    for rank in (0, 1):
        with pytest.raises(trainer.TrainingDiverged, match="3 optimizer steps in a row.*mini-batch 2"):
            _epoch(trainer, bad=range(10), rank=rank)
    # The streak is carried across epochs: two skips at the end of one epoch and one at the start of the next
    metrics, _ = _epoch(trainer, bad={8, 9})
    assert metrics["skip_streak"] == 2 and metrics["skipped_steps"] == 2
    with pytest.raises(trainer.TrainingDiverged, match="mini-batch 0"):
        _epoch(trainer, bad={0}, skip_streak=metrics["skip_streak"])
    # Isolated skips never stop a run, and a taken step resets the streak
    metrics, optimizer = _epoch(trainer, bad={0, 1, 3, 4, 6, 7})
    assert metrics["skipped_steps"] == 6 and metrics["skip_streak"] == 0
    assert optimizer.state  # the finite steps were taken


def test_the_epoch_reports_its_pre_clip_gradient_norms(trainer):
    # Gradient norms 0.5, 2, 3 and 1 (the default factor), then a skipped step, which has none; clipped at 1.0
    metrics, _ = _epoch(trainer, bad={4}, factors={0: 0.5, 1: 2.0, 2: -3.0}, n=5)
    assert metrics["grad_norm_steps"] == 4 and metrics["skipped_steps"] == 1
    assert metrics["grad_norm_mean"] == pytest.approx((0.5 + 2.0 + 3.0 + 1.0) / 4)
    assert metrics["grad_norm_max"] == pytest.approx(3.0)
    assert metrics["clipped_steps"] == 2  # 2 and 3; a norm of exactly 1.0 is not clipped
