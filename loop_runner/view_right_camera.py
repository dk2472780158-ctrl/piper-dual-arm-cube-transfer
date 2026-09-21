"""Read only the right RGB camera, with a private Rerun viewer; no robot SDK."""
import signal
import time
import cv2
import rerun as rr
import rerun.blueprint as rrb

running = True


def stop(*_):
    global running
    running = False


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, stop)

rr.init('Piper right camera - view only', spawn=False)
rr.serve_grpc(grpc_port=9877, server_memory_limit='64MiB',
              default_blueprint=rrb.Blueprint(rrb.Spatial2DView(origin='camera/right')))
rr.serve_web_viewer(web_port=9081, open_browser=False)
cap = cv2.VideoCapture('/dev/camera_right', cv2.CAP_V4L2)
try:
    if not cap.isOpened():
        raise RuntimeError('Cannot open /dev/camera_right')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    started = time.monotonic()
    last = 0.0
    failures = 0
    frames = 0
    print('RIGHT_CAMERA_OPEN; automatic release after 10 minutes', flush=True)
    while running and time.monotonic() - started < 600:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > 30:
                raise RuntimeError('Camera read repeatedly failed')
            continue
        failures = 0
        now = time.monotonic()
        if now - last < 0.1:
            continue
        last = now
        rr.set_time('camera_time', duration=now-started)
        rr.log('camera/right', rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=85))
        frames += 1
        if frames == 1:
            print(f'FIRST_FRAME_OK shape={frame.shape}', flush=True)
finally:
    cap.release()
    print('RIGHT_CAMERA_RELEASED', flush=True)
