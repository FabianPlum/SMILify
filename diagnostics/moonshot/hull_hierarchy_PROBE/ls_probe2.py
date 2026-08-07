import time, numpy as np, trimesh
from pygel3d import graph as gr, hmesh
trimesh.util.log.setLevel(50)

def to_gel(tm):
    m = hmesh.Manifold()
    for tri in tm.faces: m.add_face(tm.vertices[tri])
    hmesh.stitch(m); return m

def build(res_body=4, res_leg=16, n_leg_sub=6):
    parts=[trimesh.creation.icosphere(subdivisions=res_body, radius=1.0)]
    g=trimesh.creation.icosphere(subdivisions=res_body, radius=1.0); g.vertices*=[1.6,.9,.9]; g.apply_translation([2.6,0,0]); parts.append(g)
    h=trimesh.creation.icosphere(subdivisions=res_body-1, radius=.6); h.apply_translation([-1.5,0,0]); parts.append(h)
    for x in (-0.4,0.0,0.4):
        for s in (1,-1):
            leg=trimesh.creation.cylinder(radius=.09,height=1.8,sections=res_leg)
            leg.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
            leg.apply_translation([x,s*0.95,0.]); parts.append(leg)
    ant=trimesh.util.concatenate(parts)
    fused=trimesh.boolean.union(parts)
    return ant, fused

ant, fused = build()
print("fused: V=%d F=%d comps=%d watertight=%s" % (len(fused.vertices),len(fused.faces),fused.body_count,fused.is_watertight))

def run(tm, tag, fn, label):
    m=to_gel(tm); G=gr.from_mesh(m)
    t0=time.time(); skel,mp=fn(G); dt=time.time()-t0
    mp=np.array(list(mp)); nodes=list(skel.nodes())
    val=np.array([len(list(skel.neighbors(n,'n'))) for n in nodes])
    print(f"  [{tag}] {label}: Vgraph={len(list(G.nodes()))} skel_nodes={len(nodes)} leaves={(val==1).sum()} deg2={(val==2).sum()} junc={(val>=3).sum()} map_min={mp.min()} map_max={mp.max()} uniq={len(np.unique(mp))} t={dt:.2f}s")
    return mp, skel

import sys
mp,skel = run(fused,"FUSED", lambda G: gr.LS_skeleton_and_map(G,True), "LS sampling=True")
mp2,_   = run(fused,"FUSED", lambda G: gr.LS_skeleton_and_map(G,True), "LS sampling=True (rerun)")
print("   determinism (fused, sampling=True): identical=", np.array_equal(mp,mp2))
a,_ = run(fused,"FUSED", lambda G: gr.LS_skeleton_and_map(G,False), "LS sampling=False")
b,_ = run(fused,"FUSED", lambda G: gr.LS_skeleton_and_map(G,False), "LS sampling=False (rerun)")
print("   determinism (fused, sampling=False): identical=", np.array_equal(a,b))
run(fused,"FUSED", lambda G: gr.MSLS_skeleton_and_map(G,64), "MSLS grow=64")
run(ant,"DISCONNECTED", lambda G: gr.LS_skeleton_and_map(G,True), "LS sampling=True")
