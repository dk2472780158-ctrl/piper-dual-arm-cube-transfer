#!/usr/bin/env python3
"""MuJoCo stand-in for lerobot's PIPERDual.

The observation/action contract is copied from `lerobot.robots.piper_dual`:
  - action: 14 values, left joint_1..joint_6 + gripper, then the same for right
  - observation.state: 28 values, per motor `pos` then `effort`
  - three RGB cameras, 480x640, named left / middle / right

Only the transport changes: instead of CAN frames and USB cameras the values come
from a MuJoCo model. Everything above this class (record_loop, the ACT policy,
the reset sequence) stays byte-identical to the hardware pipeline.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / "scene.xml"

MOTORS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
SIDES = ("left", "right")

# Where each arm sits when an episode begins: the first recorded frame of the
# cube_r2l demonstration set. The auto-reset target in reset_piper_pose.py is a
# different pose -- the operator moves the arms after the reset and before the
# recording starts, so the episodes themselves begin here.
START_POSE = np.asarray([
    0.0093, 0.0049, -0.0024, -0.0582, 0.3185, 0.0973, 0.0003,
    0.0631, 0.0048, -0.0027, -0.0766, 0.3512, 0.0202, 0.0,
])

MAX_GRIPPER = 0.07

# The Piper SDK reports joints 2..5 with the opposite sign to the URDF axes
# (build_scene.JOINT_SIGN mirrors the model). Everything outside this module --
# datasets, policies, the reset poses -- speaks SDK units, so the conversion
# happens here and nowhere else.
JOINT_SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}


def _motor_keys(side: str) -> list[str]:
    return [f"{side}_{motor}" for motor in MOTORS]


class SimArmBus:
    """Looks like PiperMotorsBus to the reset code: read() and write()."""

    def __init__(self, sim, side: str):
        self._sim = sim
        self._side = side
        self.piper = None  # the SDK object is never touched in simulation

    @property
    def is_connected(self) -> bool:
        return self._sim.is_connected

    def connect(self, enable: bool = True) -> bool:
        return True

    def apply_calibration(self) -> None:
        return None

    def read(self) -> dict:
        # The reset code polls feedback in a tight loop and never calls
        # get_observation, so every poll also steps the physics forward;
        # otherwise the arms would never move and the tracking gate would trip.
        self._sim._advance()
        values = self._sim._read_side(self._side)
        self._sim._trace(self._side, values)
        return values

    def write(self, target_joint) -> None:
        values = [float(v) for v in target_joint]
        if len(values) != 7:
            raise ValueError(f"expected 7 targets, got {len(values)}")
        self._sim._write_side(self._side, values)


class PiperDualSim:
    name = "piper_dual_sim"
    robot_type = "piper_dual"
    config_class = None

    def __init__(self, config=None, *, scene: str | os.PathLike = DEFAULT_SCENE,
                 camera_size=(480, 640), render: bool | None = None,
                 realtime: bool | None = None):
        import mujoco

        self.config = config
        self.read_only = bool(getattr(config, "read_only", False))
        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(scene))
        self._data = mujoco.MjData(self._model)
        self._camera_height, self._camera_width = camera_size
        self._render_enabled = (os.environ.get("CUBE_SIM_RENDER", "1") != "0"
                                if render is None else render)
        self._realtime = (os.environ.get("CUBE_SIM_REALTIME", "0") == "1"
                          if realtime is None else realtime)
        self._renderers = {}
        # Camera effects the physics does not produce: distortion, vignetting,
        # softness and sensor noise. Off by default because the shipped
        # checkpoint was trained on the sharp renders.
        self._lens = os.environ.get("CUBE_SIM_LENS", "0") == "1"
        self._lens_frame = 0
        self._trace_enabled = os.environ.get("CUBE_SIM_TRACE", "0") == "1"
        self._static_dir = os.environ.get("CUBE_SIM_STATIC_IMAGES", "")
        # Diagnostic only: fixed 14-value effort vector, comma separated.
        raw_effort = os.environ.get("CUBE_SIM_FIXED_EFFORT", "")
        self._fixed_effort = ([float(v) for v in raw_effort.split(",")]
                              if raw_effort else None)
        # The model's gravity load is several times what the real servos
        # report, so the torque channel is corrected per joint with the fit
        # calibrate_effort.py writes. Without the file the raw model torque is
        # reported, which is far outside the range the policy was trained on.
        self._effort_gain_offset = {}
        calibration_path = HERE / "effort_calibration.json"
        if calibration_path.exists():
            import json
            with open(calibration_path, encoding="utf-8") as handle:
                for key, value in json.load(handle).items():
                    self._effort_gain_offset[key] = (
                        float(value["gain"]), float(value["offset"]))
        # Sensor model: a smooth per-channel gain field per camera, fitted to
        # the recorded takes (fit_render.py). The renders are otherwise too
        # bright and too flat -- no fall-off towards the frame edge -- and the
        # policy was trained on the recorded response.
        import render_sensor
        self._render_sensor = render_sensor
        self._render_model = render_sensor.load(
            os.environ.get("CUBE_SIM_RENDER_MODEL"))
        self._trace_count = 0
        self._connected = False
        self._calibrated = False
        self._control_period = 1.0 / 30.0
        self._clock = 0.0
        self._step_accumulator = 0.0
        self._targets = {}
        self._actuator_ids = {}
        for side in SIDES:
            for index in range(1, 7):
                self._actuator_ids[(side, index)] = self._model.actuator(
                    f"{side}_act_{index}").id
            self._actuator_ids[(side, 7)] = self._model.actuator(f"{side}_act_7").id
            self._actuator_ids[(side, 8)] = self._model.actuator(f"{side}_act_8").id

        self.left_bus = SimArmBus(self, "left")
        self.right_bus = SimArmBus(self, "right")
        self.cameras = {name: _CameraStub(self._camera_height, self._camera_width)
                        for name in ("left", "middle", "right")}
        self._reset_state(apply_ctrl=True)

    # --- lifecycle ---------------------------------------------------------
    def _reset_state(self, apply_ctrl: bool = False) -> None:
        mujoco = self._mujoco
        data, model = self._data, self._model
        mujoco.mj_resetData(model, data)
        for side in SIDES:
            offset = 0 if side == "left" else 7
            for index in range(1, 7):
                joint = model.joint(f"{side}_joint{index}")
                value = JOINT_SIGN[index] * float(START_POSE[offset + index - 1])
                data.qpos[int(np.ravel(joint.qposadr)[0])] = value
                if apply_ctrl:
                    data.ctrl[self._actuator_ids[(side, index)]] = value
            for index, (finger, sign) in enumerate(((7, 1.0), (8, -1.0)), start=7):
                data.qpos[int(np.ravel(model.joint(f"{side}_joint{index}").qposadr)[0])] = 0.0
                if apply_ctrl:
                    data.ctrl[self._actuator_ids[(side, index)]] = 0.0
        mujoco.mj_forward(model, data)
        self._clock = 0.0
        self._step_accumulator = 0.0

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("sim already connected")
        # apply_ctrl matters: mj_resetData zeroes ctrl, so without it the
        # servos would drive every joint toward zero on the first poll.
        self._reset_state(apply_ctrl=True)
        self._connected = True
        self._calibrated = True
        print("SimRobot: dual Piper on MuJoCo, three offscreen cameras", flush=True)

    def disconnect(self) -> None:
        for renderer in self._renderers.values():
            try:
                renderer.close()
            except Exception:
                pass
        self._renderers.clear()
        self._connected = False

    def calibrate(self) -> None:
        self._reset_state(apply_ctrl=True)
        self._calibrated = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def has_camera(self) -> bool:
        return True

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    def configure(self, **kwargs) -> None:
        return None

    # --- features ----------------------------------------------------------
    @property
    def camera_features(self) -> dict:
        return {f"observation.images.{name}": {
            "shape": (self._camera_height, self._camera_width, 3),
            "names": ["height", "width", "channels"], "info": None}
            for name in self.cameras}

    @property
    def motor_features(self) -> dict:
        action_names = [f"{side}_{motor}" for side in SIDES for motor in MOTORS]
        state_names = []
        for side in SIDES:
            for motor in MOTORS:
                state_names += [f"{side}_{motor}.pos", f"{side}_{motor}.effort"]
        return {
            "action": {"dtype": "float32", "shape": (len(action_names),),
                       "names": action_names},
            "observation.state": {"dtype": "float32", "shape": (len(state_names),),
                                  "names": state_names},
        }

    @property
    def action_features(self) -> dict:
        return {f"{side}_{motor}.pos": float for side in SIDES for motor in MOTORS}

    @property
    def observation_features(self) -> dict:
        features = {}
        for side in SIDES:
            for motor in MOTORS:
                features[f"{side}_{motor}.pos"] = float
                features[f"{side}_{motor}.effort"] = float
        features.update({name: (self._camera_height, self._camera_width, 3)
                         for name in self.cameras})
        return features

    # --- physics -----------------------------------------------------------
    def _advance(self) -> None:
        """Run physics up to now: fixed control period, or wall clock if enabled."""
        if self._realtime:
            drift = time.monotonic() - self._clock
            self._clock = time.monotonic()
            steps = max(1, int(drift / self._model.opt.timestep))
        else:
            self._clock += self._control_period
            steps = max(1, int(round(self._control_period / self._model.opt.timestep)))
        steps = min(steps, 2000)
        for _ in range(steps):
            self._mujoco.mj_step(self._model, self._data)

    def _write_side(self, side: str, values: list[float]) -> None:
        for index in range(1, 7):
            self._data.ctrl[self._actuator_ids[(side, index)]] = (
                JOINT_SIGN[index] * values[index - 1])
        grip = float(np.clip(values[6], 0.0, MAX_GRIPPER))
        self._data.ctrl[self._actuator_ids[(side, 7)]] = grip / 2.0
        self._data.ctrl[self._actuator_ids[(side, 8)]] = -grip / 2.0
        self._targets[side] = values

    def _calibrate_effort(self, side: str, name: str, torque: float) -> float:
        gain, offset = self._effort_gain_offset.get(f"{side}_{name}", (1.0, 0.0))
        return gain * torque + offset

    def _read_side(self, side: str) -> dict:
        model, data = self._model, self._data
        result = {}
        for index in range(1, 7):
            joint = model.joint(f"{side}_joint{index}")
            result[f"joint_{index}_pos"] = JOINT_SIGN[index] * float(
                data.qpos[int(np.ravel(joint.qposadr)[0])])
            # The joint axis was mirrored for joints 2..5, so the torque about
            # it flips with the position; the SDK reports both in its own
            # convention. `diag_effort.py` compares the result with the
            # recorded channel.
            result[f"joint_{index}_effort"] = self._calibrate_effort(
                side, f"j{index}",
                JOINT_SIGN[index] * float(data.qfrc_actuator[joint.dofadr[0]]))
        # The real SDK reports the opening in metres, 0 .. 0.07.
        finger_a = float(data.qpos[int(np.ravel(model.joint(f"{side}_joint7").qposadr)[0])])
        finger_b = float(data.qpos[int(np.ravel(model.joint(f"{side}_joint8").qposadr)[0])])
        result["gripper_pos"] = float(np.clip(finger_a - finger_b, 0.0, MAX_GRIPPER))
        result["gripper_effort"] = self._calibrate_effort(
            side, "grip", float(data.qfrc_actuator[
                model.joint(f"{side}_joint7").dofadr[0]]))
        if self._fixed_effort is not None:
            base = 0 if side == "left" else 7
            for index in range(6):
                result[f"joint_{index + 1}_effort"] = self._fixed_effort[base + index]
            result["gripper_effort"] = self._fixed_effort[base + 6]
        return result

    def _trace(self, side: str, values: dict) -> None:
        """CUBE_SIM_TRACE=1 logs every bus poll; reset debugging only."""
        if not self._trace_enabled:
            return
        self._trace_count += 1
        joints = " ".join(f"{values[f'joint_{i}_pos']:+.4f}" for i in range(1, 7))
        ctrl = " ".join(
            f"{JOINT_SIGN[i] * self._data.ctrl[self._actuator_ids[(side, i)]]:+.4f}"
            for i in range(1, 7))
        print(f"TRACE {self._trace_count:5d} {side:5s} q={joints} | "
              f"ctrl={ctrl} | grip={values['gripper_pos']:.5f} "
              f"gctrl={self._data.ctrl[self._actuator_ids[(side, 7)]]:.5f} "
              f"ncon={self._data.ncon}", flush=True)

    def _static_image(self, name: str):
        """Diagnostic only: CUBE_SIM_STATIC_IMAGES=<dir> replays real frames.

        Used to tell an image-domain gap apart from a state/contract bug; never
        set this for a normal sim run.
        """
        import imageio.v2 as imageio
        path = Path(self._static_dir) / f"{name}.png"
        image = imageio.imread(path)
        return np.asarray(image)[:, :, :3].astype(np.uint8)

    def _render(self, name: str):
        if self._static_dir:
            return self._static_image(name)
        mujoco = self._mujoco
        renderer = self._renderers.get(name)
        if renderer is None:
            renderer = mujoco.Renderer(self._model, self._camera_height,
                                       self._camera_width)
            self._renderers[name] = renderer
        renderer.update_scene(self._data, camera=name)
        image = self._render_sensor.apply(self._render_model, name,
                                          renderer.render())
        if not self._lens:
            return image
        import render_lens
        self._lens_frame += 1
        return render_lens.apply(image, self._lens_frame)

    # --- the two calls record_loop makes -----------------------------------
    def get_observation(self) -> dict:
        if not self._connected:
            raise RuntimeError("sim not connected")
        self._advance()
        observation = {}
        for side in SIDES:
            for key, value in self._read_side(side).items():
                name, suffix = key.rsplit("_", 1)
                observation[f"{side}_{name}.{suffix}"] = value
        if self._render_enabled:
            for name in self.cameras:
                observation[name] = self._render(name)
        else:
            for name in self.cameras:
                observation[name] = np.zeros((self._camera_height,
                                              self._camera_width, 3), np.uint8)
        return observation

    def send_action(self, action: dict) -> dict:
        if not self._connected:
            raise RuntimeError("sim not connected")
        if self.read_only:
            return action
        for side in SIDES:
            self._write_side(side, [float(action[f"{side}_{motor}.pos"])
                                    for motor in MOTORS])
        return action

    # --- helpers the launchers use ----------------------------------------
    def hold_pose(self, pose14) -> None:
        """Drive both arms straight to a 14-value pose (used by the reset)."""
        pose = np.asarray(pose14, dtype=float)
        self._write_side("left", pose[:7].tolist())
        self._write_side("right", pose[7:].tolist())

    def settled(self) -> bool:
        return True


class _CameraStub:
    """Only shape metadata is needed; frames come from the renderer."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.channels = 3
        self.is_connected = True

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None
