"""
Train/val/test splits shared by the neural trainers and the benchmark.

Splits are made at the *sample* (frame) level, so every camera view of a frame
lands in the same split. Two methods:

- ``"random"``: seeded random permutation of samples (the original behaviour).
  The split seed is the training seed, so existing checkpoints reproduce their
  split unchanged. ``holdout_fold`` is blocked-only and must be null here.
- ``"blocked"``: blocked k-fold cross-validation (Roberts et al. 2017,
  Ecography 40:913-929). Each recording session is ordered by ``frame_idx`` and
  cut into ``K = 1 / (val_ratio + test_ratio)`` contiguous blocks. Fold ``k``
  holds out block ``k`` of every session: the first ``val_ratio`` share of the
  block is validation, the rest is test, and the remainder of the session
  trains. With ratios 0.80 / 0.05 / 0.15 there are 5 folds, and fold 4 (the
  default) trains on the first 80% of each session, validates on the next 5%
  and tests on the last 15%. Neighbouring video frames are near-duplicates, so
  unlike random sampling this keeps them in the same split, except at the block
  edges: blocks are adjacent (no buffer gap), so the frames on either side of an
  edge fall in different splits. The split does not depend on the training
  seed, so runs with different seeds share it.

The trainers and the benchmark all go through :func:`split_dataset`, and the
trainers persist ``split_method`` / ``holdout_fold`` / ratios in the
checkpoint, so the benchmark rebuilds the exact training split.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import math

import h5py
import numpy as np
import torch
from torch.utils.data import Subset

SPLIT_METHODS = ("random", "blocked")

# Val/test order only decides which samples the epoch visualisations and the
# benchmark plots pick first (they take the first N of an unshuffled loader).
# Blocked val/test are permuted with this fixed seed so those picks span
# sessions and time instead of N consecutive, near-identical frames.
_BLOCKED_ORDER_SEED = 0


@dataclass
class SampleSplit:
    """Sample indices per split, plus log lines describing how they were made."""

    train: List[int]
    val: List[int]
    test: List[int]
    description: str
    session_lines: List[str] = field(default_factory=list)

    def log_lines(self) -> List[str]:
        return [self.description, *self.session_lines]


def num_blocked_folds(val_ratio: float, test_ratio: float) -> int:
    """Number of folds K for the blocked split, i.e. 1 / (val_ratio + test_ratio).

    The held-out block must tile a session exactly, so K has to be a whole
    number of at least 2 (e.g. 0.05 + 0.15 = 0.2 gives 5 folds).
    """
    held_out = val_ratio + test_ratio
    if held_out <= 0:
        raise ValueError("Blocked split needs val_ratio + test_ratio > 0")
    k = int(round(1.0 / held_out))
    if k < 2 or abs(k * held_out - 1.0) > 1e-6:
        raise ValueError(
            f"Blocked split needs val_ratio + test_ratio to divide a session into whole folds "
            f"(1 / (val + test) an integer >= 2, e.g. 0.05 + 0.15 = 0.2 -> 5 folds); "
            f"got {val_ratio} + {test_ratio} = {held_out:.6g}"
        )
    return k


def validate_split_settings(
    split_method: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    holdout_fold: Optional[int],
) -> None:
    """Raise ValueError if the split settings are inconsistent."""
    if split_method not in SPLIT_METHODS:
        raise ValueError(f"split_method must be one of {SPLIT_METHODS}, got {split_method!r}")
    if holdout_fold is not None and (not isinstance(holdout_fold, (int, np.integer)) or isinstance(holdout_fold, bool)):
        raise ValueError(f"holdout_fold must be an integer or null, got {holdout_fold!r}")
    if split_method == "random" and holdout_fold is not None:
        raise ValueError(
            f"holdout_fold selects a blocked fold and needs split_method='blocked'; random splits use "
            f"training.seed (got holdout_fold={holdout_fold!r})"
        )
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1, got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = {total:.6g}"
        )
    if split_method == "blocked":
        k = num_blocked_folds(val_ratio, test_ratio)
        if holdout_fold is not None and not 0 <= int(holdout_fold) < k:
            raise ValueError(f"holdout_fold must be in [0, {k - 1}] for {k} blocked folds, got {holdout_fold}")


def add_split_args(parser) -> None:
    """Add --split_method / --holdout_fold to a trainer's argument parser."""
    parser.add_argument(
        "--split_method",
        type=str,
        default=None,
        choices=list(SPLIT_METHODS),
        help="Train/val/test assignment: 'random' (seeded random split over frames) or 'blocked' "
        "(blocked k-fold cross-validation: contiguous blocks along time within each session, "
        "K = 1 / (val_ratio + test_ratio)). Default: from config (random).",
    )
    parser.add_argument(
        "--holdout_fold",
        type=int,
        default=None,
        help="Blocked split only: index of the held-out block, 0 = start of each session "
        "(default: last block). Not allowed with split_method random, which uses --seed.",
    )
    parser.add_argument(
        "--allow_split_change",
        action="store_true",
        default=None,
        help="When resuming, continue even though the current split settings differ from the checkpoint's "
        "(a deliberate fine-tune on a new split). Without it such a resume stops with an error.",
    )


def split_args_to_overrides(parser, args) -> dict:
    """Dataset-block overrides from --split_method / --holdout_fold / --allow_split_change.

    Exits via parser.error when they are given without --config: blocked
    setups must come from a JSON config alongside their ratios.
    """
    keys = ("split_method", "holdout_fold", "allow_split_change")
    overrides = {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}
    if overrides and args.config is None:
        parser.error(
            "--split_method / --holdout_fold / --allow_split_change need --config (a JSON config with a dataset block)"
        )
    return overrides


class SplitMismatchError(ValueError):
    """Resuming a checkpoint with split settings that differ from the ones it was trained with."""


def split_signature(settings: dict) -> dict:
    """The settings that decide a split, normalised so that equal splits compare equal.

    *settings* holds split_method / holdout_fold / train_ratio / val_ratio /
    test_ratio / seed as trainers save them in checkpoints (a missing
    split_method is "random", as for checkpoints that predate it). Blocked: the
    fold (null = last fold) and the val/test shares; the seed does not matter.
    Random: the seed and the train/val shares; test is the remainder.
    """
    method = settings.get("split_method") or "random"
    if method == "blocked":
        k = num_blocked_folds(float(settings["val_ratio"]), float(settings["test_ratio"]))
        fold = settings.get("holdout_fold")
        return {
            "split_method": "blocked",
            "folds": k,
            "holdout_fold": k - 1 if fold is None else int(fold),
            "val_ratio": float(settings["val_ratio"]),
            "test_ratio": float(settings["test_ratio"]),
        }
    return {
        "split_method": "random",
        "seed": settings.get("seed"),
        "train_ratio": float(settings["train_ratio"]),
        "val_ratio": float(settings["val_ratio"]),
    }


def check_resume_split(
    stored: Optional[dict], current: dict, checkpoint_path: str, allow_change: bool = False, log=print
) -> None:
    """Refuse to resume a checkpoint on a different split than it was trained on.

    A resumed run that splits differently trains on frames the checkpoint held
    out, and its new checkpoint records the new split, so the benchmark would
    then score it on frames it has seen. Raises ValueError listing the
    differences (SplitMismatchError), unless *allow_change* (dataset.allow_split_change /
    --allow_split_change), in which case it only warns. Checkpoints without
    split settings cannot be checked and only get a warning.
    """
    try:
        old = split_signature(stored or {})
        new = split_signature(current)
    except (KeyError, TypeError, ValueError) as e:
        log(f"WARNING: cannot check that resuming {checkpoint_path} keeps its train/val/test split ({e!r})")
        return

    def same(a, b):
        if isinstance(a, float) or isinstance(b, float):
            return a is not None and b is not None and math.isclose(a, b, abs_tol=1e-9)
        return a == b

    diffs = [
        f"{k}: {old.get(k)!r} -> {new.get(k)!r}"
        for k in dict.fromkeys([*old, *new])
        if not same(old.get(k), new.get(k))
    ]
    if not diffs:
        return
    msg = (
        f"Resuming {checkpoint_path} with a different train/val/test split than it was trained on "
        f"({'; '.join(diffs)}). Frames it held out would be trained on. Use the checkpoint's split settings, "
        "train from scratch, or set dataset.allow_split_change / --allow_split_change to fine-tune on the new "
        "split on purpose."
    )
    if allow_change:
        log(f"WARNING: {msg} Continuing because allow_split_change is set.")
        return
    raise SplitMismatchError(msg)


def fold_tag(split_method: str, holdout_fold: Optional[int], val_ratio: float, test_ratio: float) -> str:
    """``"_fold{k}"`` for a blocked split (null holdout_fold = the last fold), ``""`` for random.

    Appended to every output directory of a run, so the K folds of a
    cross-validation sweep never write into each other's checkpoints and plots.
    """
    if (split_method or "random") != "blocked":
        return ""
    k = num_blocked_folds(val_ratio, test_ratio)
    return f"_fold{k - 1 if holdout_fold is None else int(holdout_fold)}"


def tag_output_dir(path: str, tag: str) -> str:
    """*path* with *tag* appended to its last component (unchanged if it already ends with it)."""
    if not tag or not path:
        return path
    trimmed = path.rstrip("/\\")
    return trimmed if trimmed.endswith(tag) else trimmed + tag


def random_sample_split(n_samples: int, train_ratio: float, val_ratio: float, seed: int) -> SampleSplit:
    """Seeded random split over ``range(n_samples)``; test takes the remainder.

    Reproduces the trainers' original ``torch.utils.data.random_split`` call
    index for index (``random_split`` depends only on the lengths and the
    generator, never on the dataset contents).
    """
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    train, val, test = torch.utils.data.random_split(
        range(n_samples), [n_train, n_val, n_test], generator=torch.Generator().manual_seed(int(seed))
    )
    return SampleSplit(
        train=[int(i) for i in train.indices],
        val=[int(i) for i in val.indices],
        test=[int(i) for i in test.indices],
        description=f"Split: random (seed={seed}) -> {n_train}/{n_val}/{n_test} samples",
    )


def blocked_sample_split(
    session_names: Sequence[str],
    frame_indices: Sequence[int],
    val_ratio: float,
    test_ratio: float,
    holdout_fold: Optional[int] = None,
) -> SampleSplit:
    """Blocked k-fold split along time within each session (see module docstring).

    Args:
        session_names: Session name per sample (HDF5 row).
        frame_indices: Frame index per sample; orders samples in time.
        val_ratio, test_ratio: Shares of each session for validation and test.
        holdout_fold: Which block to hold out (0 = start of each session).
            None holds out the last block.
    """
    k = num_blocked_folds(val_ratio, test_ratio)
    fold = k - 1 if holdout_fold is None else int(holdout_fold)
    if not 0 <= fold < k:
        raise ValueError(f"holdout_fold must be in [0, {k - 1}] for {k} blocked folds, got {holdout_fold}")
    sessions = np.asarray(session_names, dtype=object)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if sessions.shape != frames.shape:
        raise ValueError(f"session_names and frame_indices differ in length: {sessions.shape} vs {frames.shape}")
    val_share = val_ratio / (val_ratio + test_ratio)

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    session_lines = []
    for name in dict.fromkeys(sessions.tolist()):  # first-appearance order
        rows = np.flatnonzero(sessions == name)
        rows = rows[np.argsort(frames[rows], kind="stable")]
        if np.any(np.diff(frames[rows]) == 0):
            raise ValueError(
                f"Session {name!r} repeats frame_idx values; the blocked split needs one row per (session, frame_idx)"
            )
        n = len(rows)
        start, end = fold * n // k, (fold + 1) * n // k
        n_val = int(round((end - start) * val_share))
        train.extend(rows[:start].tolist() + rows[end:].tolist())
        val.extend(rows[start : start + n_val].tolist())
        test.extend(rows[start + n_val : end].tolist())

        def _frames(lo, hi):
            return f"{frames[rows[lo]]}-{frames[rows[hi - 1]]}" if hi > lo else "-"

        session_lines.append(
            f"  {name}: {n} frames | val frames {_frames(start, start + n_val)} | "
            f"test frames {_frames(start + n_val, end)} | train {n - (end - start)}"
        )

    rng = np.random.default_rng(_BLOCKED_ORDER_SEED)
    val = rng.permutation(np.asarray(val, dtype=np.int64)).tolist()
    test = rng.permutation(np.asarray(test, dtype=np.int64)).tolist()
    return SampleSplit(
        train=sorted(train),
        val=val,
        test=test,
        description=(
            f"Split: blocked, fold {fold} of {k} ({len(session_lines)} sessions) -> "
            f"{len(train)}/{len(val)}/{len(test)} samples"
        ),
        session_lines=session_lines,
    )


def read_session_frame_index(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``auxiliary/session_name`` and ``auxiliary/frame_idx`` (one entry per sample)."""
    with h5py.File(hdf5_path, "r") as f:
        if "auxiliary/session_name" not in f or "auxiliary/frame_idx" not in f:
            raise ValueError(
                f"Blocked split needs auxiliary/session_name and auxiliary/frame_idx in {hdf5_path} "
                "(written by the multi-view preprocessors)"
            )
        names = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["auxiliary/session_name"][:]]
        frames = f["auxiliary/frame_idx"][:].astype(np.int64)
    return np.asarray(names, dtype=object), frames


def split_samples(
    n_samples: int,
    *,
    split_method: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    holdout_fold: Optional[int] = None,
    hdf5_path: Optional[str] = None,
) -> SampleSplit:
    """Split ``range(n_samples)`` with the configured method."""
    validate_split_settings(split_method, train_ratio, val_ratio, test_ratio, holdout_fold)
    if split_method == "random":
        return random_sample_split(n_samples, train_ratio, val_ratio, seed)
    if hdf5_path is None:
        raise ValueError("Blocked split needs a multi-view HDF5 dataset (session_name / frame_idx per sample)")
    names, frames = read_session_frame_index(hdf5_path)
    if len(names) != n_samples:
        raise ValueError(f"{hdf5_path} has {len(names)} session_name entries but the dataset has {n_samples} samples")
    return blocked_sample_split(names, frames, val_ratio, test_ratio, holdout_fold)


def expand_to_items(sample_ids: Sequence[int], item_sample_indices: np.ndarray) -> List[int]:
    """Map sample indices to the item indices of all their views.

    Items come out grouped by sample in the order of ``sample_ids`` (views in
    ascending item order within a sample).
    """
    isi = np.asarray(item_sample_indices, dtype=np.int64)
    order = np.argsort(isi, kind="stable")
    isi_sorted = isi[order]
    samples = np.asarray(sample_ids, dtype=np.int64)
    lo = np.searchsorted(isi_sorted, samples, side="left")
    hi = np.searchsorted(isi_sorted, samples, side="right")
    counts = hi - lo
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1])) if len(counts) else np.zeros(0, dtype=np.int64)
    pos = np.repeat(lo - offsets, counts) + np.arange(int(counts.sum()))
    return order[pos].tolist()


def split_dataset(
    dataset,
    *,
    split_method: str = "random",
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    holdout_fold: Optional[int] = None,
) -> Tuple[Tuple[Subset, Subset, Subset], SampleSplit]:
    """Split a dataset into (train, val, test) Subsets at the sample level.

    Multi-view datasets have one item per sample. Single-view-from-multiview
    datasets with ``item_sample_indices`` have one item per view; the split is
    made over samples and each sample's views follow it, so no view of a frame
    leaks across splits.

    Returns:
        ((train_set, val_set, test_set), sample_split)
    """
    item_sample_indices = getattr(dataset, "item_sample_indices", None)
    n_samples = int(dataset.num_samples) if item_sample_indices is not None else len(dataset)
    split = split_samples(
        n_samples,
        split_method=split_method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        holdout_fold=holdout_fold,
        hdf5_path=getattr(dataset, "hdf5_path", None),
    )
    parts = (split.train, split.val, split.test)
    if item_sample_indices is not None:
        if split_method == "random":
            # The original sample-grouped expansion kept items in dataset order.
            parts = tuple(sorted(p) for p in parts)
        parts = tuple(expand_to_items(p, item_sample_indices) for p in parts)
    subsets = tuple(Subset(dataset, list(p)) for p in parts)
    return subsets, split


SPLIT_PARTS = ("train", "val", "test")


def split_dataset_from_settings(dataset, settings: dict) -> Tuple[Tuple[Subset, Subset, Subset], SampleSplit]:
    """:func:`split_dataset` with the settings a trainer stored in a checkpoint.

    For the benchmark and inference, which must reproduce the training split.
    Checkpoints that predate split_method were random. Random: test is the
    remainder (a stored test_ratio may predate the sum check, e.g. 0.85 / 0.005 / 0.1).
    Blocked: the stored test_ratio, which sets the val/test cut exactly as in
    training (an inferred 1 - train - val can round it differently).
    """
    method = settings.get("split_method") or "random"
    train_ratio, val_ratio = settings["train_ratio"], settings["val_ratio"]
    test_ratio = settings["test_ratio"] if method == "blocked" else 1.0 - train_ratio - val_ratio
    return split_dataset(
        dataset,
        split_method=method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=settings["seed"],
        holdout_fold=settings.get("holdout_fold"),
    )


def add_inference_split_arg(parser) -> None:
    """--split for the inference scripts: restrict frames to one part of the checkpoint's split."""
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", *SPLIT_PARTS],
        help="Run only on the frames of this part of the train/val/test split the checkpoint was trained "
        "with (rebuilt from its stored settings, as in the benchmark), in time order; --max_frames then "
        "counts within it. Default: all frames. Use 'test' for held-out overlays and videos.",
    )
