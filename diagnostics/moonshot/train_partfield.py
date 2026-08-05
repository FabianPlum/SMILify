"""Train the part field. Synthetic labels are exact; real labels are fit-derived and weak.

CLASS WEIGHTING IS NOT OPTIONAL HERE. Area-sampled points are 58% body and 0.4-0.8% per
distal leg class (measured, see the data log). An unweighted network reaches ~95% per-point
accuracy by never predicting a distal class at all -- which is the identical failure mode
probe 13 found in the *fitter*, where the pretarsus expects 0.3 of 8000 chamfer samples.
Accuracy is therefore reported alongside mean-IoU and distal-IoU, and only the last two are
meaningful.

MIRROR AUGMENTATION is a flag, not a default, because gate G2 measures bilateral
consistency and training for it would make that gate self-fulfilling. Both models are
trained so the report can state what the architecture achieves unaided (`--mirror_aug` off)
and what the shippable model achieves (on). Equivariance to a mirror is a genuinely
desirable property to build in -- it is just not evidence about the architecture.

REAL-WEAK FILTERING. Specimens whose scan points sit far from their own fitted mesh do not
have debris, they have a failed fit, and their transferred labels are noise. `--max_debris`
drops them. This keeps the circularity bounded: the network is allowed to imitate fits that
worked, and never sees the ones that did not.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from fitter_3d.partfield import PartFieldNet, part_field_loss  # noqa: E402

DATA = os.path.join(HERE, "partfield")
N_PTS = 4096


def mirror_map(names):
    """Index permutation induced by reflecting the specimen in y (left <-> right)."""
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for n in names:
        if n.endswith("_l"):
            out.append(idx[n[:-2] + "_r"])
        elif n.endswith("_r"):
            out.append(idx[n[:-2] + "_l"])
        else:
            out.append(idx[n])
    return torch.tensor(out, dtype=torch.long)


def iou(pred, true, C, ignore=None):
    """Per-class IoU. Classes absent from both prediction and truth return NaN rather than
    1.0, so a model that never sees a class cannot be credited for it."""
    out = np.full(C, np.nan)
    for c in range(C):
        if ignore is not None and c == ignore:
            continue
        p, t = pred == c, true == c
        u = (p | t).sum()
        if u > 0:
            out[c] = (p & t).sum() / u
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "net.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--mirror_aug", action="store_true")
    ap.add_argument(
        "--real_weight",
        type=float,
        default=0.0,
        help="loss weight on the fit-derived real-weak batches; 0 = synthetic only",
    )
    ap.add_argument(
        "--max_debris",
        type=float,
        default=0.25,
        help="drop real specimens whose scan-to-fit debris fraction exceeds this",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument(
        "--rot_aug_deg",
        type=float,
        default=12.0,
        help="max random rotation at train time; sized from the measured "
        "misalignment spread of the bench scans, not guessed",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    names = json.load(open(os.path.join(DATA, "split.json")))["classes"]
    C = len(names)
    distal = [i for i, n in enumerate(names) if n[1:3] in ("1d", "2d", "3d")]

    tr = np.load(os.path.join(DATA, "synth_train.npz"))
    va = np.load(os.path.join(DATA, "synth_val.npz"))
    Xtr = torch.tensor(tr["pts"])
    Ytr = torch.tensor(tr["lab"])
    Xva = torch.tensor(va["pts"]).to(dev)
    Yva = torch.tensor(va["lab"]).to(dev)
    print(f"[train] synthetic {Xtr.shape[0]} train / {Xva.shape[0]} val, {C} classes")

    Xr = Yr = None
    if args.real_weight > 0:
        rw = np.load(os.path.join(DATA, "real_weak.npz"))
        keep = [i for i in rw["train_idx"] if rw["debris_frac"][i] <= args.max_debris]
        dropped = len(rw["train_idx"]) - len(keep)
        Xr = torch.tensor(rw["pts"][keep])
        Yr = torch.tensor(rw["lab"][keep])
        print(
            f"[train] real-weak {len(keep)} specimens ({dropped} dropped for debris "
            f"> {args.max_debris:.0%}), loss weight {args.real_weight}"
        )

    # inverse-sqrt-frequency class weights, so the 0.5% distal classes are learnable at all
    cnt = np.bincount(tr["lab"].ravel(), minlength=C).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    print("[train] class weights: " + ", ".join(f"{n}={float(x):.2f}" for n, x in zip(names, w)))

    mm = mirror_map(names).to(dev)
    net = PartFieldNet(n_classes=C, width=args.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(Xtr) // args.batch), pct_start=0.2
    )

    def rand_rot(B, dev, max_deg):
        """Per-sample random rotation, Rodrigues, magnitude ~ U(0, max_deg).

        Sized from the data, not guessed: PCA-1 of the bench scans deviates from the body
        axis by a median 1.0 deg in yaw but has a 90th percentile of 17.3 deg in pitch, so
        the +/-3 deg baked into the generator is far too tight. The distribution stays
        centred on zero, so the network still learns to *use* the canonical alignment
        (which G2 requires) while tolerating the spread that is actually present.
        """
        if max_deg <= 0:
            return None
        ax = torch.randn(B, 3, device=dev)
        ax = ax / ax.norm(dim=1, keepdim=True)
        th = torch.rand(B, 1, device=dev) * (max_deg * np.pi / 180.0)
        K = torch.zeros(B, 3, 3, device=dev)
        K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
        K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
        I = torch.eye(3, device=dev).expand(B, 3, 3)
        s, c = th.sin().view(B, 1, 1), th.cos().view(B, 1, 1)
        return I + s * K + (1 - c) * (K @ K)

    def batch_aug(x, y):
        """Light train-time augmentation on top of what the generator baked in."""
        x = x + torch.randn_like(x) * 0.003
        x = x * (1 + torch.randn(x.shape[0], 1, 1, device=x.device) * 0.03)
        R = rand_rot(x.shape[0], x.device, args.rot_aug_deg)
        if R is not None:
            x = torch.bmm(x, R.transpose(1, 2))
        if args.mirror_aug:
            f = torch.rand(x.shape[0], device=x.device) < 0.5
            x = torch.where(f[:, None, None], x * torch.tensor([1.0, -1.0, 1.0], device=x.device), x)
            y = torch.where(f[:, None], mm[y], y)
        return x, y

    best, hist = -1.0, []
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(Xtr.shape[0])
        tot = nb = 0.0
        for i in range(0, Xtr.shape[0] - args.batch + 1, args.batch):
            idx = perm[i : i + args.batch]
            x, y = Xtr[idx].to(dev, non_blocking=True), Ytr[idx].to(dev, non_blocking=True)
            x, y = batch_aug(x, y)
            loss = part_field_loss(net(x), y, class_weight=w)
            if Xr is not None:
                ridx = torch.randint(0, Xr.shape[0], (max(2, args.batch // 4),))
                pidx = torch.randint(0, Xr.shape[1], (len(ridx), N_PTS))
                rx = torch.gather(Xr[ridx], 1, pidx.unsqueeze(-1).expand(-1, -1, 3)).to(dev)
                ry = torch.gather(Yr[ridx], 1, pidx).to(dev)
                rx = rx - rx.mean(1, keepdim=True)
                rx = rx / rx.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                rx, ry = batch_aug(rx, ry)
                loss = loss + args.real_weight * part_field_loss(net(rx), ry, class_weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1

        net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, Xva.shape[0], 16):
                preds.append(net(Xva[i : i + 16]).argmax(-1))
        pr = torch.cat(preds).cpu().numpy()
        gt = Yva.cpu().numpy()
        acc = float((pr == gt).mean())
        I = iou(pr, gt, C)
        miou = float(np.nanmean(I))
        diou = float(np.nanmean(I[distal]))
        hist.append(dict(ep=ep, loss=tot / max(nb, 1), acc=acc, miou=miou, distal_iou=diou))
        if miou > best:
            best = miou
            torch.save(
                dict(
                    state=net.state_dict(),
                    names=names,
                    width=args.width,
                    args=vars(args),
                    epoch=ep,
                    miou=miou,
                    acc=acc,
                    distal_iou=diou,
                ),
                args.out,
            )
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(
                f"[train] ep{ep:3d} loss {tot / max(nb, 1):.4f}  acc {acc:.4f}  "
                f"mIoU {miou:.4f}  distal-IoU {diou:.4f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    print(f"\n[train] best mIoU {best:.4f} -> {args.out}")
    ck = torch.load(args.out)
    print(
        f"[train] at best: acc {ck['acc']:.4f}  mIoU {ck['miou']:.4f}  "
        f"distal-IoU {ck['distal_iou']:.4f}  (epoch {ck['epoch']})"
    )
    net.load_state_dict(ck["state"])
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, Xva.shape[0], 16):
            preds.append(net(Xva[i : i + 16]).argmax(-1))
    I = iou(torch.cat(preds).cpu().numpy(), Yva.cpu().numpy(), C)
    print("[train] per-class IoU on held-out synthetic:")
    for n, v in zip(names, I):
        print(f"        {n:>8s}  {v:.4f}" if np.isfinite(v) else f"        {n:>8s}  absent")
    json.dump(
        dict(
            hist=hist, per_class_iou=[None if not np.isfinite(v) else float(v) for v in I], names=names, args=vars(args)
        ),
        open(args.out.replace(".pt", "_hist.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
