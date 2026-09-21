"""Three RGB cameras in Rerun; no CAN/robot SDK, ten minute lifetime."""
import signal
import threading
import time
import cv2
import rerun as rr
import rerun.blueprint as rrb

stop = threading.Event()
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, lambda *_: stop.set())

rr.init('Piper three cameras - view only', spawn=False)
rr.serve_grpc(grpc_port=9877, server_memory_limit='64MiB',
    default_blueprint=rrb.Blueprint(rrb.Horizontal(
        *[rrb.Spatial2DView(name=name.upper(), origin=f'camera/{name}')
          for name in ('left', 'middle', 'right')])) )
rr.serve_web_viewer(web_port=9081, open_browser=False)
started = time.monotonic()


def capture(name):
    cap = cv2.VideoCapture('/dev/camera_' + name, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open ' + name)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        last = 0.0
        frames = failures = 0
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 30:
                    raise RuntimeError('Read failed: ' + name)
                continue
            failures = 0
            now = time.monotonic()
            if now - last < 0.1:
                continue
            last = now
            rr.set_time('camera_time', duration=now-started)
            rr.log('camera/' + name,
                   rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=85))
            frames += 1
            if frames == 1:
                print(f'FIRST_FRAME_OK {name} shape={frame.shape}', flush=True)
    except Exception as error:
        print(f'CAMERA_ERROR {name}: {error}', flush=True)
        stop.set()
    finally:
        cap.release()
        print('RELEASED ' + name, flush=True)


threads = [threading.Thread(target=capture, args=(name,))
           for name in ('left', 'middle', 'right')]
for thread in threads:
    thread.start()
stop.wait(600)
stop.set()
for thread in threads:
    thread.join()
