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
