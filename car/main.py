#!/usr/bin/env python3

import time, threading, socket, struct, pickle, json
import cv2, afb

from modules.steering_module import SteeringAnalyzer
from modules.fsm_module import LaneFSM
from modules.drive_module import (
    get_current_mode, get_current_speed, BASE_ANGLE
)

WIDTH, HEIGHT, FPS = 640, 480, 30
JPEG_QUALITY = 85
SEND_FPS = 1

LAPTOP_IP   = "192.168.137.1"
LAPTOP_PORT = 8485

PI_LISTEN_HOST = "0.0.0.0"
PI_LISTEN_PORT = 8587

PRINT_DEBUG = True

latest_frame = None
frame_lock = threading.Lock()
selected_label = None
key_start_flag = False

analyzer = SteeringAnalyzer(
    base_angle= 85,
    max_offset=48.5,
    angle_limits=(30.0, 150.0),
    y_bottom_ratio=0.70,
    y_top_ratio=0.40,
    k_head=0.60
)
fsm = LaneFSM()


def compute_servo_target(mode: str, angle: float | None) -> float:
    m = (mode or "").upper()
    if m == "LEFT":
        return 30.0
    if m == "RIGHT":
        return 150.0
    return float(BASE_ANGLE)


def camera_thread():
    global latest_frame
    while True:
        f = afb.camera.get_image()
        if f is not None:
            with frame_lock:
                latest_frame = f
        time.sleep(0.0)


def sender_thread():
    sock = None
    while True:
        try:
            if sock is None:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((LAPTOP_IP, LAPTOP_PORT))
                sock = s
                if PRINT_DEBUG:
                    print(f"[TX] connected to {LAPTOP_IP}:{LAPTOP_PORT}")
            with frame_lock:
                f = latest_frame
            if f is None:
                time.sleep(0.05); continue
            ok, jpg = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                time.sleep(1.0/max(1,SEND_FPS)); continue
            data = pickle.dumps(jpg, protocol=4)
            sock.sendall(struct.pack(">L", len(data)) + data)
            time.sleep(1.0/max(1,SEND_FPS))
        except (ConnectionError, OSError):
            try:
                if sock: sock.close()
            except: pass
            sock = None
            if PRINT_DEBUG:
                print("[TX] reconnecting in 2s ...")
            time.sleep(2)


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk: raise ConnectionError("closed")
        buf += chunk
    return buf


def receiver_thread():
    global selected_label
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((PI_LISTEN_HOST, PI_LISTEN_PORT))
    srv.listen(1)
    if PRINT_DEBUG:
        print(f"[RX] listening on {PI_LISTEN_HOST}:{PI_LISTEN_PORT}")
    while True:
        try:
            conn, addr = srv.accept()
            if PRINT_DEBUG:
                print(f"[RX] connected from {addr}")
            with conn:
                while True:
                    size = struct.unpack(">L", recv_exact(conn, 4))[0]
                    blob = recv_exact(conn, size)
                    msg = json.loads(blob.decode("utf-8"))
                    selected_label = msg.get("selected")
                    if PRINT_DEBUG:
                        print(f"[RX] label={selected_label} raw={msg}")
        except Exception as e:
            if PRINT_DEBUG:
                print(f"[RX] error: {e}")
            time.sleep(0.5)


def key_thread():
    global key_start_flag
    try:
        while True:
            ch = input().strip().lower()
            if ch == "s":
                key_start_flag = True
                if PRINT_DEBUG:
                    print("[KEY] start trigger ON")
    except Exception:
        pass


def drive_loop():
    global key_start_flag
    last_log_t = 0.0
    while True:
        with frame_lock:
            f = latest_frame
        if f is None:
            time.sleep(0.01); continue

        result = analyzer.analyze_frame(f)
        angle = result["angle"]

        fsm.update(angle, selected_label, key_start_flag)
        key_start_flag = False

        now = time.time()
        if PRINT_DEBUG and (now - last_log_t) >= 0.10:
            mode = get_current_mode()
            speed = get_current_speed()
            servo = compute_servo_target(mode, angle)
            state_val = getattr(fsm, "state", "?")
            print(f"[DBG] FSM:{state_val} | MODE:{mode} | SPEED:{speed} | SERVO:{servo:.1f} | ANGLE:{angle:.1f} | LABEL:{selected_label}")
            last_log_t = now

        time.sleep(1.0 / max(1, FPS))


def main():
    afb.gpio.init()
    afb.camera.init(WIDTH, HEIGHT, FPS)

    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=sender_thread, daemon=True).start()
    threading.Thread(target=receiver_thread, daemon=True).start()
    threading.Thread(target=key_thread, daemon=True).start()

    try:
        drive_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
