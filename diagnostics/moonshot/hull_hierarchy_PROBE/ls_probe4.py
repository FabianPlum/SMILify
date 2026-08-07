import time, numpy as np, trimesh, fast_simplification as fs
from pygel3d import graph as gr, hmesh
rng=np.random.default_rng(1)
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
def run(tm,label,msls=None,sampling=False,post=False):
    m=to_gel(tm); G=gr.from_mesh(m); t0=time.time()
    skel,mp=(gr.MSLS_skeleton_and_map(G,msls) if msls else gr.LS_skeleton_and_map(G,sampling)); dt=time.time()-t0
    def d(s):
        n=list(s.nodes()); v=np.array([len(list(s.neighbors(x,'n'))) for x in n]); return len(n),(v==1).sum(),(v>=3).sum()
    n0,l0,j0=d(skel)
    out=f"  {label}: V={len(tm.vertices)} -> skel={n0} leaves={l0} junc={j0} t={dt:.1f}s"
    if post:
        gr.edge_contract(skel, 0.15*np.mean([np.linalg.norm(np.array(skel.positions()[a])-np.array(skel.positions()[b])) for a,b in [(0,0)]] ) if False else 0.25)
        gr.prune(skel); skel.cleanup(); n1,l1,j1=d(skel)
        out+=f" | after edge_contract(0.25)+prune: skel={n1} leaves={l1} junc={j1}"
    print(out)
print("== two independent decimations (the geodesic stability test analogue), LS sampling=False ==")
for frac in (0.75,0.5,0.35):
    v,f=fs.simplify(base.vertices.astype(np.float32), base.faces.astype(np.int32), target_reduction=1-frac)
    tm=trimesh.Trimesh(v,f,process=True); run(tm,f"decim keep={frac}")
print("== holes / debris ==")
tm=base.copy(); keep=np.ones(len(tm.faces),bool); keep[rng.choice(len(tm.faces),300,replace=False)]=False; tm.update_faces(keep)
run(tm,"300 faces removed (holes)")
deb=trimesh.creation.icosphere(subdivisions=1,radius=.05); deb.apply_translation([0,0,2.0])
run(trimesh.util.concatenate([tm,deb]),"holes + debris blob")
print("== noise + post-processing cleanup ==")
tmn=base.copy(); tmn.vertices+=rng.normal(0,0.004,tmn.vertices.shape)
run(tmn,"noise sigma=0.004",post=True)
run(tmn,"noise sigma=0.004 MSLS64",msls=64,post=True)
