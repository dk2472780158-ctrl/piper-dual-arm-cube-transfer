import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("/home/databall/VLA/cube_mujoco/sim/scene.xml")
for name in ["left_gripper_base","left_link7","left_link8","left_link6","left_link5"]:
    b=m.body(name)
    for g in range(m.body_geomadr[b.id], m.body_geomadr[b.id]+m.body_geomnum[b.id]):
        if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH: continue
        mh=m.geom_dataid[g]; a=m.mesh_vertadr[mh]; n=m.mesh_vertnum[mh]
        v=m.mesh_vert[a:a+n]
        print(name, "verts",n,"min",np.round(v.min(0),3),"max",np.round(v.max(0),3))
