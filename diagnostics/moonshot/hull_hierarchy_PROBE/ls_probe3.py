import time, numpy as np, trimesh
from pygel3d import graph as gr, hmesh
rng=np.random.default_rng(0)
def to_gel(tm):
    m=hmesh.Manifold()
    for tri in tm.faces: m.add_face(tm.vertices[tri])
    hmesh.stitch(m); return m
def build():
    parts=[trimesh.creation.icosphere(subdivisions=4,radius=1.0)]
    g=trimesh.creation.icosphere(subdivisions=4,radius=1.0); g.vertices*=[1.6,.9,.9]; g.apply_translation([2.6,0,0]); parts.append(g)
    h=trimesh.creation.icosphere(subdivisions=3,radius=.6); h.apply_translation([-1.5,0,0]); parts.append(h)
    for x in (-.4,0.,.4):
        for s in (1,-1):
            leg=trimesh.creation.cylinder(radius=.09,height=1.8,sections=16)
            leg.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
            leg.apply_translation([x,s*.95,0.]); parts.append(leg)
    return trimesh.boolean.union(parts)
base=build()
def stats(tm,label,sampling=False,msls=None):
    m=to_gel(tm); G=gr.from_mesh(m)
    t0=time.time()
    skel,mp = (gr.MSLS_skeleton_and_map(G,msls) if msls else gr.LS_skeleton_and_map(G,sampling))
    dt=time.time()-t0
    mp=np.array(list(mp)); nodes=list(skel.nodes())
    val=np.array([len(list(skel.neighbors(n,'n'))) for n in nodes])
    print(f"  {label}: V={len(tm.vertices)} wt={tm.is_watertight} comps={tm.body_count} -> skel={len(nodes)} leaves={(val==1).sum()} junc={(val>=3).sum()} unassigned={(mp<0).sum()} t={dt:.1f}s")
    return (val==1).sum(), mp, skel
print("== branch (leaf) recovery under perturbation, LS sampling=False ==")
for trial in range(4):
    tm=base.copy(); tm.vertices += rng.normal(0,0.004,tm.vertices.shape)  # scan-like noise
    stats(tm, f"noise trial {trial}")
print("== decimation (different surface samplings) ==")
for frac in (0.6,0.4):
    tm=base.simplify_quadric_decimation(int(len(base.faces)*frac))
    stats(tm, f"decimated {frac}")
print("== holes (non-watertight) + debris ==")
tm=base.copy(); keep=np.ones(len(tm.faces),bool); keep[rng.choice(len(tm.faces),300,replace=False)]=False
tm.update_faces(keep)
stats(tm,"300 random faces removed")
deb=trimesh.creation.icosphere(subdivisions=1,radius=.05); deb.apply_translation([0,0,2.0])
tm2=trimesh.util.concatenate([tm,deb]); stats(tm2,"holes + 1 debris blob")
print("== MSLS speed/quality on same mesh ==")
for gt in (32,64,128):
    stats(base,f"MSLS grow_thresh={gt}",msls=gt)
