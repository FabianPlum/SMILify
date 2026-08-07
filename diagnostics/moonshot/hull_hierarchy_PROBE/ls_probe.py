import time, numpy as np, trimesh
from pygel3d import graph as gr, hmesh

def to_gel(tm):
    m = hmesh.Manifold()
    f = tm.faces
    for tri in f:
        m.add_face(tm.vertices[tri])
    hmesh.stitch(m)
    return m

# --- synthetic "ant": gaster + thorax + head + 6 legs pressed flat against thorax
parts=[]
parts.append(trimesh.creation.icosphere(subdivisions=3, radius=1.0).apply_transform(trimesh.transformations.scale_matrix(1.0)))
g = trimesh.creation.icosphere(subdivisions=3, radius=1.0); g.vertices*= [1.6,0.9,0.9]; g.apply_translation([2.6,0,0]); parts.append(g)
h = trimesh.creation.icosphere(subdivisions=2, radius=0.6); h.apply_translation([-1.5,0,0]); parts.append(h)
for i,x in enumerate([-0.4,0.0,0.4]):
    for s in (1,-1):
        leg = trimesh.creation.cylinder(radius=0.09, height=1.8, sections=12)
        T = trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]); leg.apply_transform(T)
        leg.apply_translation([x, s*0.95, 0.0])   # starts INSIDE the body sphere -> touching
        parts.append(leg)
ant = trimesh.util.concatenate(parts)
print("concatenated ant: V=%d F=%d components=%d watertight=%s" % (len(ant.vertices), len(ant.faces), ant.body_count, ant.is_watertight))

try:
    fused = trimesh.boolean.union(parts)
    print("BOOLEAN UNION ok: V=%d F=%d components=%d watertight=%s" % (len(fused.vertices), len(fused.faces), fused.body_count, fused.is_watertight))
except Exception as e:
    fused=None; print("boolean union unavailable:", type(e).__name__, e)

for name, tm in [("touching-not-fused", ant)] + ([("FUSED (welded legs)", fused)] if fused is not None else []):
    m = to_gel(tm)
    g_ = gr.from_mesh(m)
    n_in = len(list(g_.nodes()))
    for label, fn in [("LS(sampling=True)", lambda G: gr.LS_skeleton_and_map(G, True)),
                      ("MSLS(grow=64)", lambda G: gr.MSLS_skeleton_and_map(G, 64))]:
        G = gr.from_mesh(m)
        t0=time.time(); skel, mp = fn(G); dt=time.time()-t0
        mp = np.array(list(mp))
        nodes=list(skel.nodes())
        val=[len(list(skel.neighbors(n,'n'))) for n in nodes]
        leaves=sum(1 for v in val if v==1); junc=sum(1 for v in val if v>=3)
        print(f"  [{name}] {label}: graph_nodes={n_in} skel_nodes={len(nodes)} leaves={leaves} junctions={junc} map_len={len(mp)} uniq_parts={len(np.unique(mp))} t={dt:.2f}s")
    # determinism check on the same graph
    G1=gr.from_mesh(m); s1,m1=gr.LS_skeleton_and_map(G1,True)
    G2=gr.from_mesh(m); s2,m2=gr.LS_skeleton_and_map(G2,True)
    a1,a2=np.array(list(m1)),np.array(list(m2))
    print(f"  [{name}] two LS runs identical partition: {np.array_equal(a1,a2)}  n_parts {len(np.unique(a1))} vs {len(np.unique(a2))}")
