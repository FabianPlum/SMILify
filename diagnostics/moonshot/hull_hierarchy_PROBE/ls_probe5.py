import numpy as np, trimesh, time
from pygel3d import graph as gr, hmesh
rng=np.random.default_rng(2)
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
def go(m,label,msls=64):
    G=gr.from_mesh(m); t0=time.time(); skel,mp=gr.MSLS_skeleton_and_map(G,msls); dt=time.time()-t0
    n,l,j=desc(skel); print(f"  {label}: MSLS -> skel={n} leaves={l} junc={j} t={dt:.2f}s"); return skel,np.array(list(mp))
print("== hole repair via hmesh.close_holes ==")
tm=base.copy(); keep=np.ones(len(tm.faces),bool); keep[rng.choice(len(tm.faces),300,replace=False)]=False; tm.update_faces(keep)
m=to_gel(tm); go(m,"holey, NO repair")
m2=to_gel(tm); hmesh.close_holes(m2); go(m2,"holey, hmesh.close_holes()")
print("== MSLS noise robustness, 4 trials ==")
for t in range(4):
    tn=base.copy(); tn.vertices+=rng.normal(0,0.004,tn.vertices.shape); go(to_gel(tn),f"noise trial {t}")
print("== semantic check: does the partition isolate legs? ==")
m=to_gel(base); skel,mp=go(m,"clean base")
V=np.array(base.vertices)
# graph node i corresponds to mesh vertex i (from_mesh preserves order?) -> check count
print("   n_graph_nodes vs n_mesh_verts:",len(mp),len(V))
pos=np.array(m.positions())
for p in np.unique(mp):
    sel=mp==p; c=pos[sel].mean(0); ext=pos[sel].ptp(0)
    print(f"   part {p:3d}: n={sel.sum():5d} centroid=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) bbox=({ext[0]:.2f},{ext[1]:.2f},{ext[2]:.2f})")
