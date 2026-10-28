"""Compare the first observation.state of a sim run against a real episode."""
import glob
import sys

import numpy as np
import pandas as pd

sim_glob = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/databall/.cache/huggingface/lerobot/local/eval_sim_cube_r2l_*/data/chunk-000/file-000.parquet")
real_glob = ("/home/databall/.cache/huggingface/lerobot/local/cube_r2l_dataset_v2_robust"
             "/data/chunk-*/file-*.parquet")

sim_path = sorted(glob.glob(sim_glob))[-1]
real_path = sorted(glob.glob(real_glob))[0]
ds = pd.read_parquet(sim_path)
dr = pd.read_parquet(real_path)
dr = dr[dr.episode_index == 0]
ss = np.stack(ds["observation.state"].values)
sr = np.stack(dr["observation.state"].values)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("sim file", sim_path)
print("sim  pos   ", ss[0][0::2])
print("real pos   ", sr[0][0::2])
print("sim  effort", ss[0][1::2])
print("real effort", sr[0][1::2])
print("sim  action[0]", np.stack(ds["action"].values)[0])
print("real action[0]", np.stack(dr["action"].values)[0])
