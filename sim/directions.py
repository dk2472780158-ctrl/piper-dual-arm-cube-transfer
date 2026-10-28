"""The two passes over the cell, written once so every driver agrees on them.

There is a recording of the right-to-left pass only. The left-to-right pass is
that recording reflected in y: the arm bases, the mats and the cube are all
placed as mirror images, so reflecting the pose of one arm gives a pose of the
other. `diag_mirror.py` finds which joint signs that reflection reverses and
checks it against the fitted scene (0.0 mm and 0.0 deg on every frame tried).

    from directions import DIRECTION, mirror, spawn
"""
import math

import numpy as np

# One recorded right-arm pose read as a left-arm pose: a reflection about y=0
# reverses exactly the joints whose axis is not perpendicular to it. The SDK
# sign convention is the same on both arms, so this map applies to the action
# vector directly.
MIRROR_SIGN = (-1.0, 1.0, 1.0, -1.0, 1.0, -1.0)

YELLOW = (0.0, 0.1110)
PURPLE = (0.0, -0.1110)
MAT_RADIUS = 0.0694
MAT_TOP = 0.712 + 0.003            # table top plus mat thickness
CUBE_HALF = 0.0154
CUBE_OFF = (0.020, -0.003)         # from the purple mat centre; the cube was
CUBE_YAW = 20.0                    # set down by hand, not centred

# Which arm moves, where the cube starts and where it has to end up.
DIRECTION = {
    "r2l": dict(worker="right", start=PURPLE, target=YELLOW),
    "l2r": dict(worker="left", start=YELLOW, target=PURPLE),
}


def mirror(action):
    """The same pass with the two arms swapped: right-to-left becomes left-to-right.

    Reflecting the cell in y swaps the arm bases, swaps the mats and turns the
    hands into each other's mirror image, so the two halves of the action vector
    swap and the joints in MIRROR_SIGN reverse. The grippers are unaffected:
    they open and close the same way on both arms.
    """
    out = np.array(action, dtype=float, copy=True)
    out[0:6] = MIRROR_SIGN * action[7:13]
    out[6] = action[13]
    out[7:13] = MIRROR_SIGN * action[0:6]
    out[13] = action[6]
    return out


def stream(actions, direction):
    """The action column for one direction, from the recorded one."""
    if direction == "r2l":
        return actions
    return np.stack([mirror(a) for a in actions])


def spawn(direction):
    """Where the cube has to sit for this pass to grab it, as free-joint qpos.

    The right arm's pass starts at the purple mat; the left arm's pass is that
    pass reflected, so it starts at the yellow mat with the offset and the yaw
    reflected too.
    """
    sign = 1.0 if direction == "r2l" else -1.0
    mat = DIRECTION[direction]["start"]
    x = mat[0] + CUBE_OFF[0]
    y = mat[1] + sign * CUBE_OFF[1]
    yaw = math.radians(sign * CUBE_YAW) / 2
    return x, y, MAT_TOP + CUBE_HALF, math.cos(yaw), 0.0, 0.0, math.sin(yaw)
