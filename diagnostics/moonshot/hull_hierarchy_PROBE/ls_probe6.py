import numpy as np, trimesh, time
from pygel3d import graph as gr, hmesh
rng=np.random.default_rng(3)
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
def desc(s):
    n=list(s.nodes()); v=np.array([len(list(s.neighbors(x,'n'))) for x in n]); return len(n),int((v==1).sum()),int((v>=3).sum())
print("== HOLES: does gr.saturate() rescue the graph? ==")
tm=base.copy(); keep=np.ones(len(tm.faces),bool); keep[rng.choice(len(tm.faces),300,replace=False)]=False; tm.update_faces(keep)
m=to_gel(tm)
for hops,frac in [(0,0),(2,1.001),(2,1.5),(3,1.5)]:
    G=gr.from_mesh(m)
    if hops: gr.saturate(G,hops,frac,1e300)
    t0=time.time(); skel,mp=gr.MSLS_skeleton_and_map(G,64); dt=time.time()-t0
    n,l,j=desc(skel); print(f"  saturate(hops={hops},dist_frac={frac}): skel={n} leaves={l} junc={j} t={dt:.2f}s")
print("== semantic check on clean mesh ==")
m=to_gel(base); pos=np.array(m.positions()); G=gr.from_mesh(m)
skel,mp=gr.MSLS_skeleton_and_map(G,64); mp=np.array(list(mp))
n,l,j=desc(skel); print(f"  MSLS: skel={n} leaves={l} junc={j}; graph nodes={len(pos)} mesh verts={len(base.vertices)}")
for p in np.unique(mp):
    sel=mp==p; P=pos[sel]; c=P.mean(0); ext=P.max(0)-P.min(0)
    print(f"   part {p:2d}: n={int(sel.sum()):5d} centroid=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) extent=({ext[0]:.2f},{ext[1]:.2f},{ext[2]:.2f})")
