"""Build side-by-side montages of a recorded frame and the replayed render.

Usage:  python compare_frames.py [camera] [frames...]
"""
import glob
import sys

import numpy as np
import imageio.v2 as imageio

ROOT = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust")
camera = sys.argv[1] if len(sys.argv) > 1 else "middle"
frames = [int(v) for v in sys.argv[2:]] or [150, 270, 300, 330]
reader = imageio.get_reader(
    glob.glob(f"{ROOT}/videos/observation.images.{camera}/*/*.mp4")[0])
rows = []
for frame in frames:
    real = reader.get_data(frame)[:, :, :3]
    sim = imageio.imread(f"/tmp/replay/f{frame:04d}_{camera}.png")[:, :, :3]
    rows.append(np.concatenate([real, sim], axis=1))
imageio.imwrite(f"/tmp/cmp_{camera}.png", np.concatenate(rows, axis=0))
print("wrote", f"/tmp/cmp_{camera}.png", np.array(rows).shape)
