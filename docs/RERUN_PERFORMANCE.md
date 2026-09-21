# Why Rerun was slowing the robot down, and what `launch/fast/` does about it

## The symptom

On hardware, runs *with* Rerun telemetry felt noticeably worse than runs without it. "Worse" was not a small constant overhead — the arm would move acceptably for a while, then become sluggish, then freeze for up to a second in the middle of a trajectory.

## The measurements

Step timings taken from the loop logs' own `time` field (ms per step, and the control rate that follows from it):

| Date | Rerun? | Stages / steps | ms per step | Rate | p95 | Worst single step |
|---|---|---|---|---|---|---|
| 9/17 | no | 18 / 7048 | 38.1 | 26.2 Hz | ~57 ms | 185 ms |
| 9/18 | **yes** | 20 / 7685 | 39.0 | 25.6 Hz | ~55 ms | 189 ms |
| 9/20 | **yes** | 7 / 1853 | 62.0 | 16.1 Hz | 270 ms | **1144 ms** |

Row two is the row that rules out the obvious explanation. On 9/18 the instrumented run was **as fast as the uninstrumented one**. So the cost is not a fixed per-step overhead that Rerun adds.

Row three is what the complaint was actually about: p95 inflated from ~55 ms to 270 ms, and a single step that took 1.1 seconds — 34 control periods. That is the freeze.

The shape is *degradation*, not *slowness*.

## Root cause

`rr.log()` executes on the calling thread, and it blocks when the SDK's send channel is full.

Tracing the call:

```
rr.log()                                    rerun/_log.py:212
  → ChunkBatcherInner::send_cmd
    → re_quota_channel::sync::Sender::send_impl
      → crossbeam_channel::Sender::send    (the BLOCKING variant)
```

The SDK binary carries the matching message: *"Channel byte budget (N) exceeded. Blocking until space is available."* And in SDK 0.31.3, `ChunkBatcherConfig` exposes only four flush-related fields — there is **no "drop instead of block" switch**. If the consumer cannot keep up, the producer waits. There is no configuration that changes this.

Now the part that makes it a control problem rather than a telemetry problem: **where the call sits.**

```
piper_cube_record.py:645   send_action()          ← the arm is commanded here
piper_cube_record.py:651   dataset write
piper_cube_record.py:653   log_rerun_data()       ← can block here
piper_cube_record.py:694   precise_sleep()        ← remaining budget
```

The Rerun call is between the action being sent and the sleep that pads out the 30 Hz period. Any time it loses comes straight out of the control period; it is never recovered.

So the causal chain is:

```
viewer/server falls behind
  → the SDK's send-channel byte budget fills
    → rr.log() blocks on the control thread
      → the control period stretches, then stalls outright
```

## Why the server fell behind

Four contributing causes, all measured or read directly out of the code:

**1. Uncompressed frames during reset.** `piper_cube_record.py:931-944` calls the reset path without passing `display_compressed_images`, so it takes the `compress_images=False` branch. Three cameras at 480×640×3 = **2.76 MB per step**, uncompressed, for the whole reset phase.

**2. Camera entities are overwrite-only, sent 30 times a second.** `visualization_utils.py:93-94` logs each camera image with `static=True`, and `image_ext.py` sets up the entity accordingly. An overwrite-only entity has exactly one useful state: the latest frame. At 30 Hz that means 29 of every 30 frames are pure waste — the server processes and stores data that is discarded before anyone can see it.

**3. The flush threshold was set 131× below the SDK default.** `visualization_utils.py:37-38` sets `RERUN_FLUSH_NUM_BYTES` to **8000 bytes**, against an SDK default of **1 MiB**. Each step's images are tens of kilobytes, so a single step triggered several sends; at 30 Hz that is hundreds of small messages per second instead of a handful.

**4. The server was running with a hard memory cap and continuous eviction.** The viewer process was started with `--server-memory-limit 64MiB --newest-first`, i.e. it constantly drops the oldest data and compacts. Measured behaviour: the resident set rolled over roughly every 20 seconds. This is what makes the problem *time-dependent* rather than constant — it explains why 9/18 was fine and 9/20 was not.

Worth stating plainly: cause 4 was the platform's configuration, not a bug in it. A 64 MiB cap is a reasonable setting for a viewer. The mistake was putting a blocking call in a hard real-time path and then blaming the thing on the other end of it.

## The fix

`launch/fast/` adds an overlay. It **modifies nothing** — not LeRobot's source, not the platform's viewer, not the original launchers.

### 1. Get `log_rerun_data` off the control thread — the actual fix

`cube_fastlog.py` replaces the logging function with one that, on the control thread, only:

- decides whether this frame should be logged (scalars always, images per `CUBE_RERUN_IMG_HZ`),
- copies the numpy arrays,
- does a non-blocking `put_nowait` into a bounded queue, dropping the **oldest** entry when full.

A background thread drains the queue and calls the *original* function. The control thread never blocks — not when the server is slow, not when the network is gone.

Two properties make this safe:

- **The scalar curves keep full 30 Hz resolution.** Only camera frames are downsampled, and they are overwrite-only anyway, so nothing observable is lost. This was a hard requirement of the fix.
- **`rr.init()` uses `make_default=True`**, so it installs a global default recording. Calling `rr.log()` from a background thread lands on the same recording as the control thread would have. No recording plumbing is needed.

Exceptions in the background thread are swallowed and reported **once** — telemetry must never take down the control loop. The queue depth defaults to 3 and is tunable; the worst case is a few dropped frames, never a stalled robot.

### 2. Camera frames to 5 Hz

`CUBE_RERUN_IMG_HZ` defaults to `5`. Since camera entities are overwrite-only, this cuts the byte volume by 5/6 with no loss of visible information. Set it to `0` to log no images at all — the 42 scalar curves, which are the actual diagnostic value, are unaffected.

### 3. Flush threshold back to the SDK default

`fast_env.sh` sets `RERUN_FLUSH_NUM_BYTES=1048576`. That is an environment variable, not a code change — the smallest possible intervention for one of the four causes.

A useful side effect: the wrapper always calls the original with `compress_images=True`, so the reset phase's 2.76 MB-per-step uncompressed frames become JPEG too.

## How it is installed

`sitecustomize.py` is imported by the interpreter at start-up, before any application module, and patches `lerobot.utils.visualization_utils.log_rerun_data`.

Timing is not a detail here. `piper_cube_record.py:156` does:

```python
from ...visualization_utils import init_rerun, log_rerun_data
```

That binds the function **by name** into the module's namespace. Patching `visualization_utils` afterwards would change nothing. `sitecustomize.py` therefore patches `viz` *and* re-patches the already-imported `piper_cube_record` module if it happens to be loaded, covering both orders.

Getting `sitecustomize.py` to run at all requires the overlay directory to be on `PYTHONPATH` ahead of everything else. Both launchers **append** to `PYTHONPATH` rather than replacing it, so prepending from `fast_env.sh` survives — a fragile-looking invariant that is actually guaranteed by the launchers' own code.

If patching fails for any reason, the whole thing is wrapped in `try/except`: it prints one warning line and the run continues with the original behaviour. There is no path where broken telemetry stops a robot.

## Rollback

One prefix:

```bash
CUBE_RERUN_FAST=off bash launch/cube_loop_rerun_fast.sh
```

`CUBE_RERUN_FAST=off` skips the patch entirely at interpreter start-up. Behaviour is then identical to running the original launcher.

## What is verified, and what is not

**Verified.**

- The root cause above is established by static evidence: SDK source, the blocking-primitive chain, the specific call site and its position in the control path, and the four server-side contributors. Every file and line referenced here was read.
- The measurements in the table are real, taken from the loop logs of three hardware runs.
- `probe_fastlog.py unit` — the logic-level self-check of the queue, the drop-oldest policy and the frame-decimation counter — **passes**.

**Not verified.**

- **The fix has never run on hardware.** All three `*_fast.sh` launchers are untested against real arms.
- The decisive experiment has been written but **not executed**: a `blackhole` sink (a TCP listener that accepts and never reads) with the original logger should show the *original* `log_rerun_data` blocking for hundreds of milliseconds, and the fast version returning in well under a millisecond with a non-zero drop count. That experiment would close the loop on both the diagnosis and the fix. It is the first thing to run before trusting this overlay.
- `probe_fastlog.py`'s `timing`, `file` and `compare` modes are likewise written but unexecuted.

So: read this document as **a well-evidenced diagnosis with a plausible, untested-in-production fix** — not as a validated improvement. The claim "this makes the loop smoother" has not been demonstrated.

## Reproducing the missing verification

No hardware, no robot, no cameras:

```bash
cd launch/fast

# 1) queue / drop-oldest / frame-decimation logic — does not even import rerun
python -B probe_fastlog.py unit

# 2) THE decisive experiment: point the SDK at a TCP listener that accepts and
#    never reads, then compare the two loggers.  --variant original should show
#    call times climbing into the hundreds of ms; --variant fast should stay
#    sub-millisecond and report its drop count.
python -B probe_fastlog.py blackhole --variant original --seconds 60
python -B probe_fastlog.py blackhole --variant fast     --seconds 60

# 3) step-timing under synthetic load against your own viewer on 9877
#    (this never touches a viewer someone else is using)
python -B probe_fastlog.py timing --port 9877 --variant fast --seconds 180

# 4) content equivalence: write both variants to disk and compare entity names
python -B probe_fastlog.py file --variant original --out /tmp/pf_orig.rrd
python -B probe_fastlog.py file --variant fast     --out /tmp/pf_fast.rrd
python -B probe_fastlog.py compare --a /tmp/pf_orig.rrd --b /tmp/pf_fast.rrd
# -> exits 0 when the entity-name sets match
```

`blackhole`, `timing` and `file` import Rerun, which on this project took 10–20 s the first time. `unit` does not.

Then, and only then, one hardware run:

```bash
bash launch/cube_loop_rerun_fast.sh
# expect: STAGE n/20 progressing normally, a control rate back near 26 Hz,
#         the fast-logger confirmation line at start-up, and a summary on exit.
```

## Lessons worth carrying to other projects

- **Instrumentation that runs in a real-time path must be non-blocking by construction.** "It was fast last time" is not a guarantee — it held on 9/18 and did not hold on 9/20.
- **A synchronous logging API with no drop policy is a latency bug waiting for a slow consumer.** Before putting one on a control thread, find out what it does when the buffer is full.
- **Measure over time, not just the mean.** Mean step time was 62 ms; the p95 was 270 ms and the max was 1144 ms. The mean alone would have suggested a mild slowdown rather than occasional one-second freezes.
- **Distrust an untouched baseline.** The comparison that mattered was instrumented-vs-uninstrumented on a *different day*, which is what disproved the "Rerun is just slow" theory.
