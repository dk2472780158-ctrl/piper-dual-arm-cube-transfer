"""Side-by-side of the recorded wrist views and the simulated ones at the start.

Usage:  python compare_wrist.py [frame] [out]
"""
import glob
import sys

import numpy as np
import imageio.v2 as imageio

ROOT = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust")
frame = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cmp_wrist.png"
rows = []
for name in ("left", "middle", "right"):
    path = glob.glob(f"{ROOT}/videos/observation.images.{name}/*/*.mp4")[0]
    real = imageio.get_reader(path).get_data(frame)[:, :, :3]
    sim = imageio.imread(f"/tmp/sim_frame0_{name}.png")[:, :, :3]
    rows.append(np.concatenate([real, sim], axis=1))
imageio.imwrite(out, np.concatenate(rows, axis=0))
print("wrote", out)
