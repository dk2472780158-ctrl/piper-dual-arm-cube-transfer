#!/usr/bin/env python3
"""Generate scene.xml: two Piper arms, the cube-transfer table, and three cameras.

The arm kinematics come straight from the official AgileX URDF, so joint axes,
link lengths and masses are not re-invented here. Everything that belongs to the
cell (table, mats, cube, camera mounts) is described by the CELL dict below and
is expected to be tuned against the recorded dataset frames.

Usage:  python build_scene.py [--out scene.xml]
"""
from __future__ import annotations

import os
import math
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
URDF = ASSETS / "piper_description.urdf"

# --- cell layout -------------------------------------------------------------
CELL = {
    "table_height": 0.712,       # top surface, metres
    "table_size": (0.55, 0.75, 0.02),  # half extents x, y and half thickness
    # The arm mounts follow from the recording: at the grasp frame the
    # fingertips sit on the purple mat, at the release frame on the yellow mat
    # (solve_cell.py). The arm base ends up level with the table top once the
    # mats are given their real thickness. x, y, z in metres, yaw in radians.
    # The solve above put the right fingertips on the purple mat *centre*, which
    # is where the cube would be if a script had placed it. It was put down by
    # hand, cube_on_purple away from that centre, and the recorded joints close on
    # the cube. So the right base carries the same offset; without it the gesture
    # descends 1.8 cm beside the cube and closes on air.
    "right_mount": (-0.3565, -0.3050, 0.7120, 0.026180),
    # The left arm mirrors the right one on the table, so it carries the same
    # correction; the left-to-right pass is the right one reflected in y.
    "left_mount": (-0.3565, 0.3050, 0.7120, -0.026180),
    "yellow_mat": (0.0, 0.1110),
    "purple_mat": (0.0, -0.1110),
    # Mat radius and cube edge are fitted to the recorded frame together with
    # the middle camera (fit_middle.py): the cube-to-mat size ratio in the image
    # pins them, and 30.8 mm agrees with the 29 mm the gripper closes to.
    "mat_radius": 0.0694,
    # The recorded mats are flat cut-outs: no side wall is visible at any
    # camera angle. A thick cylinder shows one in every wrist view and raises
    # the cube off the table, so the mats are 3 mm.
    "mat_thickness": 0.003,
    "cube_size": 0.0154,      # half edge
    "cube_mass": 0.030,
    # The cube was set down by hand, not centred: 2.0 cm along the table and 20
    # deg off axis. Fitted to the recorded frame by moving it until its colour
    # mask overlapped the recorded one (fit_cube_yaw.py; overlap 0.45 -> 0.87).
    "cube_on_purple": (0.020, -0.003, 0.0),  # offset from the purple mat centre
    "cube_yaw": 20.0,
    # Fitted to the recorded frame 0 (fit_middle.py): the camera sits between
    # the two arm bases, 0.47 m from the mats, looking over them down at 36 deg.
    "middle_cam": (-0.4712, -0.0031, 1.1373),
    "middle_dir": (0.8009, 0.0754, -0.5940),
    "middle_up": (0.5948, 0.0131, 0.8037),
    # Wrist cameras live in the gripper_base frame (+z runs flange -> fingertips).
    # fit_wrist.py picks each mount from the recorded start frame by overlapping
    # the mat, cube and gripper silhouettes; the two sides are fitted
    # independently because the brackets are not exact mirrors.
    "wrist_cams": {
        "left": {"pos": (-0.0823, 0.0159, 0.0209),
                 "tilt": 24.558, "azim": -78.789, "roll": -76.680,
                 "fovy": 36.148},
        "right": {"pos": (-0.0884, -0.0155, 0.0084),
                  "tilt": 25.425, "azim": -96.888, "roll": -107.341,
                  "fovy": 32.186},
    },
    # 产品形态（AGILEX PiPER 双臂）：两臂中间竖一根立柱，柱顶朝前下方斜挂
    # 中置相机模组。两者都是纯视觉件（contype/conaffinity=0），不参与碰撞，
    # 标定出来的 middle_cam / middle_dir / middle_up 全部保持原值。
    # 柱轴与柱顶高度由 middle_cam 反推：模组背面接柱顶，镜头面落在 middle_cam。
    "rig_column": {
        "x": -0.5072, "y": -0.0065,
        "half": (0.0300, 0.0300),   # 60 x 60 mm 方管
        "z_top": 1.1090,
    },
    "rig_cam_housing": {
        "half": (0.0320, 0.0320, 0.0450),   # 64 x 64 x 90 mm 模组
    },
    "middle_fovy": 37.373,
    "wrist_fovy": 52.77,   # fallback for a side without its own fovy
}


def rad(degrees: float) -> float:
    """The compiler runs in radians because the URDF rpy values are radians."""
    import math
    return math.radians(degrees)


def cam_xyaxes(direction, up) -> str:
    """Build MuJoCo's xyaxes string: camera x and y axes, world coords.

    A camera looks along its own -z, so its third axis is -direction. Its x axis
    is the viewer's right, which is direction x up; taking up x (-direction)
    instead mirrors the rendered image left to right.
    """
    import numpy as np
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    z = -d
    u = np.asarray(up, dtype=float)
    x = np.cross(d, u)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return fmt([*x, *y], 6)

RADIANS = "radian"


def wrist_camera(name: str, side: str) -> str:
    """Wrist camera expressed in the gripper_base frame.

    The mount is stored the way fit_wrist.py searches for it -- a tilt off the
    flange axis, an azimuth around it, a roll about the view direction -- and
    the quaternion is built with the same helper the fit uses, so the generated
    camera is the one that was fitted. Handing MuJoCo an equivalent set of axes
    instead was not equivalent: the fit builds its axes as matrix rows while
    MuJoCo reads such a matrix column-major, and the two cameras ended up 30 to
    50 degrees apart.
    """
    import mujoco
    import numpy as np
    spec = CELL["wrist_cams"][side]
    tilt, azim, roll = (np.radians(spec[key]) for key in ("tilt", "azim", "roll"))
    direction = np.array([np.sin(tilt) * np.cos(azim),
                          np.sin(tilt) * np.sin(azim),
                          np.cos(tilt)])
    x = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    y = np.cross(x, direction)
    cr, sr = np.cos(roll), np.sin(roll)
    x, y = cr * x + sr * y, -sr * x + cr * y
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([x, y, -direction]).reshape(-1))
    return (f'<camera name="{name}" pos="{fmt(spec["pos"], 5)}" '
            f'quat="{fmt(quat, 8)}" fovy="{spec["fovy"]}"/>')

# Collision masks (collide when (con1 & aff2) || (con2 & aff1)):
#   bit 1  cube only
#   bit 2  table, mats, floor
#   bit 4  arm links
#   bit 8  gripper fingers
# The cube rests on the cell and is grasped by the fingers; the fingers ignore
# the cell; the arm links ignore the cube. The STL files are render meshes, so
# same-arm body pairs are also excluded in the <contact> section.
ARM_COLLIDE = os.environ.get("CUBE_SIM_ARM_COLLIDE", "0") == "1"
_ARM_MASK = (4, 4) if ARM_COLLIDE else (0, 0)
COLLIDE_MASK = {
    "link1": _ARM_MASK, "link2": _ARM_MASK, "link3": _ARM_MASK,
    "link4": _ARM_MASK, "link5": _ARM_MASK, "link6": _ARM_MASK,
    "gripper_base": (0, 0), "link7": (8, 1), "link8": (8, 1),
}

# The Piper SDK reports joints 2..5 with the opposite sign to the URDF axes.
# Every recorded pose and every policy action is in SDK units, so the model
# negates those four joints and mirrors their limits to match.
JOINT_SIGN = {"joint2": -1.0, "joint3": -1.0, "joint4": -1.0, "joint5": -1.0}

ARM_BODIES = ["mount", "link1", "link2", "link3", "link4", "link5", "link6",
              "gripper_base", "link7", "link8"]

# Joint servo gains. The real arm is commanded through MotionCtrl at speed 50,
# which behaves like a stiff position servo; kp here is tuned for 500 Hz.
JOINT_KP = {"joint1": 1200, "joint2": 1200, "joint3": 1000, "joint4": 800,
            "joint5": 600, "joint6": 400}
JOINT_KV = {"joint1": 40, "joint2": 40, "joint3": 35, "joint4": 25,
            "joint5": 20, "joint6": 15}
# The URDF advertises 100 N.m for every arm joint; without that headroom the
# simulated arm droops 0.1 rad under its own weight where the real servo holds.
JOINT_FORCE = {"joint1": 100, "joint2": 100, "joint3": 100, "joint4": 100,
               "joint5": 100, "joint6": 100}
GRIPPER_KP, GRIPPER_KV, GRIPPER_FORCE = 3000, 60, 10
# The hardware never reaches its commanded opening while it holds the cube:
# the command reads 0.0293 and the joint reads 0.0334, and that 4.1 mm residue
# is the grip. A positive value widens the aperture and the touch point moves
# by twice the amount. The flat pads below already put the touch where the
# hardware has it, so nothing is added here; the knob stays for re-measuring.
FINGER_APERTURE = 0.0
# The flat pad on each finger, in the gripper_base frame at zero opening: the
# face lies on the plane the two fingers meet on, and the plate is thin in y.
FINGER_PAD = {
    "centre": (0.010, 0.0, 0.154),
    "half": (0.010, 0.0, 0.013),
    "thickness": 0.002,
}


def fmt(values, digits=6):
    return " ".join(f"{float(v):.{digits}g}" for v in values)


def rpy_to_quat_values(rpy):
    """URDF roll/pitch/yaw (extrinsic xyz) to a MuJoCo quaternion (w x y z).

    MuJoCo's `euler` attribute is intrinsic XYZ, which is not the same
    composition as URDF's fixed-axis rpy, so the rotation is converted once
    here and emitted as `quat`.
    """
    import numpy as np

    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    # R = Rz(yaw) Ry(pitch) Rx(roll)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yq = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, yq, z)


def rpy_to_quat(rpy) -> str:
    return fmt(rpy_to_quat_values(rpy), 9)


def quat_to_matrix(q):
    import numpy as np

    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def parse_urdf(path: Path):
    root = ET.parse(path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    children = {}          # parent link -> [(joint elem, child link)]
    child_of = {}          # child link -> joint elem
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children.setdefault(parent, []).append(joint)
        child_of[child] = joint
    return links, children, child_of


def finger_pad(child, body_xyz, body_rpy):
    """A flat pad on each finger, in that finger's own frame (XML line).

    The finger mesh alone meets its opposite on a small chamfer near the top of
    the jaw, so the cube is held on a corner and squeezes out mid-swing. The
    real fingers carry a flat rubber pad; this is that pad, a thin plate whose
    face lies on the plane the two fingers close onto, placed in the base frame
    and converted into the finger's frame here.
    """
    import numpy as np

    sign = -1.0 if child == "link7" else 1.0
    centre = np.array([FINGER_PAD["centre"][0],
                       sign * FINGER_PAD["thickness"] / 2,
                       FINGER_PAD["centre"][2]])
    half = (FINGER_PAD["half"][0], FINGER_PAD["thickness"] / 2,
            FINGER_PAD["half"][2])
    quat = rpy_to_quat_values(body_rpy)
    R = quat_to_matrix(quat)
    local = R.T @ (centre - np.array([float(v) for v in body_xyz.split()]))
    # the plate is axis aligned in the base frame, so undo the finger rotation
    w, x, y, z = quat
    out = f"{w:g} {-x:g} {-y:g} {-z:g}"
    return (f'<geom type="box" pos="{fmt(local, 9)}" quat="{out}" '
            f'size="{half[0]:g} {half[1]:g} {half[2]:g}" '
            f'group="3" contype="8" conaffinity="1" rgba="0.15 0.15 0.15 0.3"/>')


def emit_link_body(link_name, links, children, prefix, indent, out, cams=None,
                   own_xyz=None, own_rpy=None):
    pad = "  " * indent
    link = links[link_name]
    visual = link.find("visual")
    inertial = link.find("inertial")
    if inertial is not None:
        origin = inertial.find("origin")
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        mass = inertial.find("mass").get("value")
        inertia = inertial.find("inertia")
        full = [inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        out.append(f'{pad}<inertial pos="{xyz}" mass="{mass}" fullinertia="{fmt(full)}"/>')
    if link_name in ("link7", "link8") and own_xyz is not None:
        out.append(pad + finger_pad(link_name, own_xyz, own_rpy))
    # visual mesh
    mesh_name = link_name
    material = "piper"
    out.append(f'{pad}<geom type="mesh" mesh="{mesh_name}" material="{material}" '
               f'group="1" contype="0" conaffinity="0"/>')
    # collision mesh: only where the mask says so
    con, aff = COLLIDE_MASK.get(link_name, (0, 0))
    out.append(f'{pad}<geom type="mesh" mesh="{mesh_name}" group="3" '
               f'contype="{con}" conaffinity="{aff}" rgba="0.2 0.2 0.2 0.2"/>')
    for cam_name in (cams or {}).get(link_name, ()):
        side = cam_name
        out.append(pad + wrist_camera(name=cam_name, side=side))

    for joint in children.get(link_name, []):
        name = joint.get("name")
        kind = joint.get("type")
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        if child in ("link7", "link8"):
            # The fingers slide apart along this frame's y, link7 on the minus
            # side; widening the aperture moves each one further out.
            shift = FINGER_APERTURE if child == "link7" else -FINGER_APERTURE
            x, y, z = (float(v) for v in xyz.split())
            xyz = f"{x:g} {y + shift:.6f} {z:g}"
        out.append(f'{pad}<body name="{prefix}{child}" pos="{xyz}" '
                   f'quat="{rpy_to_quat(rpy.split())}">')
        if kind != "fixed":
            axis = joint.find("axis").get("xyz")
            limit = joint.find("limit")
            jtype = "hinge" if kind == "revolute" else "slide"
            sign = JOINT_SIGN.get(name, 1.0)
            ax = [float(v) * sign for v in axis.split()]
            lo, hi = float(limit.get("lower")), float(limit.get("upper"))
            if sign < 0:
                lo, hi = -hi, -lo
            out.append(f'{pad}  <joint name="{prefix}{name}" type="{jtype}" '
                       f'axis="{ax[0]:g} {ax[1]:g} {ax[2]:g}" '
                       f'range="{lo:g} {hi:g}" damping="0.5" armature="0.01"/>')
        emit_link_body(child, links, children, prefix, indent + 1, out, cams,
                       own_xyz=xyz, own_rpy=rpy.split())
        out.append(f'{pad}</body>')


# make_textures.py centres its maps here, so a material that carries one is
# scaled by 255 / TEX_MEAN: the map multiplies the albedo, and the mean colour
# stays where the calibration left it.
TEX_MEAN = 250
TEX_GAIN = 255.0 / TEX_MEAN


def albedo(rgb) -> str:
    """rgba string for a textured material, compensated for the map's mean."""
    return " ".join(f"{min(1.0, channel * TEX_GAIN):.4f}" for channel in rgb) + " 1"


def build(xml_path: Path, realism: bool = False) -> None:
    links, children, _ = parse_urdf(URDF)
    compiler = (f'  <compiler angle="{RADIANS}" autolimits="true" '
                f'meshdir="../assets/meshes"')
    if realism:
        compiler += ' texturedir="textures"'
    out = ['<mujoco model="piper_cube">',
           compiler + '/>',
           '  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>',
           # A bigger shadow map is what keeps the 3 mm mat edges from turning
           # into a ring of acne once the key light casts, and multisampling
           # takes the stair-steps off their rims.
           ('  <visual><global offwidth="1280" offheight="960"/>'
            '<quality shadowsize="8192" offsamples="8"/></visual>' if realism else
            '  <visual><global offwidth="1280" offheight="960"/></visual>'),
           '  <asset>']
    for name, link in links.items():
        if link.find("visual") is None:
            continue
        out.append(f'    <mesh name="{name}" file="{name}.STL"/>')
    if realism:
        # Texture maps only modulate the calibrated albedo: they are near-white
        # with a few percent of noise, so the mean colour stays where
        # calibrate_colors.py put it and only the plastic flatness goes away.
        # They are centred on TEX_MEAN (make_textures.py) rather than 255, so the
        # albedo is scaled by the same factor to keep the mean unchanged. The
        # cube has no map: at 30 mm it covers about thirty pixels, where a
        # texture is nothing but aliasing across the box's flat faces.
        out += ['    <texture name="tex_table" type="2d" file="table.png"/>',
                '    <material name="piper" rgba="0.22 0.23 0.25 1" specular="0.4" shininess="0.3"/>',
                f'    <material name="tabletop" rgba="{albedo((0.316, 0.350, 0.347))}" '
                'specular="0.1" shininess="0.05" texture="tex_table" '
                'texrepeat="6 6" texuniform="true"/>',
                # The mats, the cube and the arms stay flat: in the recording the
                # mats have no visible weave at this distance, and any map small
                # enough to be honest only aliases into a dotted rim.
                '    <material name="mat_yellow" rgba="0.579 0.531 0.347 1" specular="0"/>',
                '    <material name="mat_purple" rgba="0.481 0.408 0.565 1" specular="0"/>',
                '    <material name="cube_pink" rgba="0.752 0.376 0.478 1" specular="0"/>',
                '    <material name="backdrop" rgba="0.86 0.87 0.85 1"/>',
                # 立柱与模组：黑色哑光，产品渲染里就是这个观感
                '    <material name="rig_black" rgba="0.075 0.075 0.082 1" '
                'specular="0.15" shininess="0.2"/>']
    else:
        out += ['    <material name="piper" rgba="0.22 0.23 0.25 1" specular="0.4" shininess="0.3"/>',
                # Albedo taken from the recorded middle frame (calibrate_colors.py):
                # the key light adds about a third, so these sit below the recorded rgb.
                '    <material name="tabletop" rgba="0.316 0.350 0.347 1" specular="0.1" shininess="0.05"/>',
                # MuJoCo defaults specular to 0.5, which puts a white highlight on
                # these when a wrist camera looks down the light; the real mats and
                # the wooden cube are matte.
                '    <material name="mat_yellow" rgba="0.579 0.531 0.347 1" specular="0"/>',
                '    <material name="mat_purple" rgba="0.481 0.408 0.565 1" specular="0"/>',
                '    <material name="cube_pink" rgba="0.752 0.376 0.478 1" specular="0"/>',
                '    <material name="backdrop" rgba="0.86 0.87 0.85 1"/>',
                # 立柱与模组：黑色哑光，产品渲染里就是这个观感
                '    <material name="rig_black" rgba="0.075 0.075 0.082 1" '
                'specular="0.15" shininess="0.2"/>']
    # The key light casts; the fill exists to keep the shadow side readable and
    # would only wash the contact shadow out if it cast as well. A shadow map
    # with the light straight down sits at the same depth as every flat top
    # surface, which floods the mats with acne; a slight tilt clears it and
    # costs under two percent of the brightness on those surfaces.
    shadow = "true" if realism else "false"
    key_dir = "-0.18 0.12 -1" if realism else "0 0 -1"
    out += ['  </asset>', '  <worldbody>',
            f'    <light name="key" pos="0.1 0.0 2.0" dir="{key_dir}" directional="true" '
            'diffuse="0.50 0.50 0.50" ambient="0.20 0.20 0.20" specular="0.08 0.08 0.08" '
            f'castshadow="{shadow}"/>',
            '    <light name="fill" pos="-1.0 1.0 1.2" dir="0.6 -0.6 -0.8" directional="true" '
            'diffuse="0.22 0.22 0.22" ambient="0.10 0.10 0.10" castshadow="false"/>']

    h = CELL["table_height"]
    sx, sy, sz = CELL["table_size"]
    out += [f'    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 0" '
            f'material="backdrop" contype="2" conaffinity="5"/>']
    # Three light walls so the wrist and middle cameras see a room, not black.
    for name, pos, size in (("wall_far", "0 1.1 1.0", "1.1 0.04 1.0"),
                            ("wall_left", "-1.1 0 1.0", "0.04 1.1 1.0"),
                            ("wall_right", "1.1 0 1.0", "0.04 1.1 1.0")):
        out.append(f'    <geom name="{name}" type="box" size="{size}" pos="{pos}" '
                   f'material="backdrop" contype="0" conaffinity="0"/>')
    out += [
            f'    <geom name="table" type="box" size="{sx} {sy} {sz}" '
            f'pos="0 0 {h - sz}" material="tabletop" contype="2" conaffinity="5"/>']

    mt = CELL["mat_thickness"]
    for name, (mx, my), mat in (("yellow", CELL["yellow_mat"], "mat_yellow"),
                                ("purple", CELL["purple_mat"], "mat_purple")):
        out.append(f'    <geom name="{name}_mat" type="cylinder" '
                   f'size="{CELL["mat_radius"]} {mt / 2}" pos="{mx} {my} {h + mt / 2}" '
                   f'material="{mat}" contype="2" conaffinity="5"/>')

    cube_off = CELL["cube_on_purple"]
    px, py = CELL["purple_mat"]
    c = CELL["cube_size"]
    half_yaw = math.radians(CELL["cube_yaw"]) / 2
    out += [f'    <body name="cube" pos="{px + cube_off[0]} {py + cube_off[1]} {h + mt + c}" '
            f'quat="{math.cos(half_yaw):.6f} 0 0 {math.sin(half_yaw):.6f}">',
            '      <freejoint name="cube_free"/>',
            f'      <geom name="cube_geom" type="box" size="{c} {c} {c}" '
            f'mass="{CELL["cube_mass"]}" material="cube_pink" '
            f'friction="1.4 0.05 0.001" contype="1" conaffinity="2"/>',
            '    </body>']

    for side in ("left", "right"):
        mx, my, mz, yaw = CELL[f"{side}_mount"]
        out.append(f'    <body name="{side}_mount" pos="{mx:.4f} {my:.4f} {mz:.4f}" '
                   f'euler="0 0 {yaw:.6f}">')
        emit_link_body("base_link", links, children, f"{side}_", 3, out,
                       cams={"gripper_base": [side]})
        out.append('    </body>')

    # --- 中央立柱与中置相机模组（纯视觉件，不参与碰撞） -------------
    rig = CELL["rig_column"]
    hx, hy, hz = CELL["rig_cam_housing"]["half"]
    col_h = rig["z_top"] - h
    out.append(f'    <geom name="rig_column" type="box" '
               f'size="{rig["half"][0]:.4f} {rig["half"][1]:.4f} {col_h / 2:.4f}" '
               f'pos="{rig["x"]:.4f} {rig["y"]:.4f} {h + col_h / 2:.4f}" '
               f'material="rig_black" contype="0" conaffinity="0"/>')
    cam_pos = CELL["middle_cam"]
    zc = [-v for v in CELL["middle_dir"]]        # 相机自身 +z 背离视线
    ctr = [cam_pos[i] + zc[i] * hz for i in range(3)]
    out.append(f'    <geom name="rig_cam_housing" type="box" '
               f'size="{hx:.4f} {hy:.4f} {hz:.4f}" pos="{fmt(ctr, 5)}" '
               f'xyaxes="{cam_xyaxes(CELL["middle_dir"], CELL["middle_up"])}" '
               f'material="rig_black" contype="0" conaffinity="0"/>')

    out.append(f'    <camera name="middle" pos="{fmt(CELL["middle_cam"], 5)}" '
               f'xyaxes="{cam_xyaxes(CELL["middle_dir"], CELL["middle_up"])}" '
               f'fovy="{CELL["middle_fovy"]}"/>')

    out.append('  </worldbody>')
    out.append('  <contact>')
    for side in ("left", "right"):
        for i in range(len(ARM_BODIES)):
            for j in range(i + 1, len(ARM_BODIES)):
                out.append(f'    <exclude body1="{side}_{ARM_BODIES[i]}" '
                           f'body2="{side}_{ARM_BODIES[j]}"/>')
    out.append('  </contact>')
    out.append('  <actuator>')
    for side in ("left", "right"):
        for index in range(1, 7):
            key = f"joint{index}"
            joint = f"{side}_{key}"
            out.append(f'    <position name="{side}_act_{index}" joint="{joint}" '
                       f'kp="{JOINT_KP[key]}" kv="{JOINT_KV[key]}" ctrllimited="false" '
                       f'forcerange="{-JOINT_FORCE[key]} {JOINT_FORCE[key]}"/>')
        out.append(f'    <position name="{side}_act_7" joint="{side}_joint7" '
                   f'kp="{GRIPPER_KP}" kv="{GRIPPER_KV}" ctrllimited="false" '
                   f'forcerange="{-GRIPPER_FORCE} {GRIPPER_FORCE}"/>')
        out.append(f'    <position name="{side}_act_8" joint="{side}_joint8" '
                   f'kp="{GRIPPER_KP}" kv="{GRIPPER_KV}" ctrllimited="false" '
                   f'forcerange="{-GRIPPER_FORCE} {GRIPPER_FORCE}"/>')
    out.append('  </actuator>')
    out.append('  <sensor>')
    for side in ("left", "right"):
        for index in range(1, 7):
            out.append(f'    <jointactuatorfrc name="{side}_force_{index}" '
                       f'joint="{side}_joint{index}"/>')
        out.append(f'    <jointactuatorfrc name="{side}_force_gripper" joint="{side}_joint7"/>')
    out.append('  </sensor>')
    out.append('</mujoco>')

    xml_path.write_text("\n".join(out) + "\n")
    print("wrote", xml_path, len(out), "lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "scene.xml"))
    ap.add_argument("--realism", action="store_true",
                    help="textured surfaces and contact shadows; the geometry "
                         "and the dynamics are unchanged")
    parser = ap.parse_args()
    build(Path(parser.out), realism=parser.realism)
