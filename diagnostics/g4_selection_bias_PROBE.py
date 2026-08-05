import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/home/fabi/dev/SMILify")
torch.set_num_threads(os.cpu_count())
from fitter_3d.partfield import PartFieldNet

D = "/home/fabi/dev/SMILify/diagnostics/moonshot/partfield"
ck = torch.load(os.path.join(D, "net_B_mirror.pt"), map_location="cpu")
net = PartFieldNet(n_classes=len(ck["names"]), width=ck.get("width", 1.0))
net.load_state_dict(ck["state"])
net.eval()
print("ckpt epoch", ck["epoch"], "acc", ck["acc"], "miou", ck["miou"], "diou", ck["distal_iou"])

va = np.load(os.path.join(D, "synth_val.npz"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
X = torch.tensor(va["pts"][:N])
Y = torch.tensor(va["lab"][:N])
t0 = time.time()
accs = []
with torch.no_grad():
    for k in range(0, N, 8):
        p = net(X[k : k + 8]).argmax(-1)
        accs.append((p == Y[k : k + 8]).float().mean(-1).numpy())
        print(f"  {k + 8}/{N} {time.time() - t0:.0f}s", flush=True)
a = np.concatenate(accs)
print("n clouds", len(a))
print("overall acc %.4f" % a.mean())
print("per-cloud acc: std %.4f  min %.4f  max %.4f" % (a.std(ddof=1), a.min(), a.max()))
print("SE of mean over 500 clouds = %.5f" % (a.std(ddof=1) / np.sqrt(500)))
np.save("/tmp/claude-1000/-home-fabi-dev-SMILify/6c5a615e-788c-4441-a87e-20e1d0723794/scratchpad/percloud.npy", a)
