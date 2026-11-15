import afb, cv2, time, os, threading
import numpy as np
import config_preprocessing as cfg_pre
import config_vehicle as cfg_vehicle
from tensorflow.keras.models import load_model
from collections import deque

MODEL_PATH = "steering_classification_lightcnn.h5"
cnn_model = load_model(MODEL_PATH)
input_shape = cnn_model.input_shape
CNN_INPUT_SIZE = (input_shape[1], input_shape[2])
CNN_CLASSES = ["FORWARD", "LEFT", "RIGHT"]

afb.gpio.init()
afb.camera.init(cfg_vehicle.IMG_WIDTH, cfg_vehicle.IMG_HEIGHT, cfg_vehicle.FPS)


def get_next_filename(base_name="drive", ext=".avi", folder="."):
    i = 1
    while True:
        filename = f"{base_name}_{i}{ext}"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            return filepath
        i += 1


raw_video_name = get_next_filename("drive_raw")
overlay_video_name = get_next_filename("drive_overlay")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
raw_out = cv2.VideoWriter(raw_video_name, fourcc, cfg_vehicle.FPS, (cfg_vehicle.IMG_WIDTH, cfg_vehicle.IMG_HEIGHT))
overlay_out = cv2.VideoWriter(overlay_video_name, fourcc, cfg_vehicle.FPS, (cfg_vehicle.IMG_WIDTH, cfg_vehicle.IMG_HEIGHT))

M = cv2.getPerspectiveTransform(
    np.float32(cfg_pre.SRC_POINTS),
    np.float32(cfg_pre.DST_POINTS)
)


def preprocess(frame):
    bev = cv2.warpPerspective(frame, M, cfg_pre.BEV_SIZE)
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    if cfg_pre.USE_BLUR:
        gray = cv2.GaussianBlur(gray, cfg_pre.BLUR_KERNEL, 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def cnn_predict(frame):
    binary = preprocess(frame)
    resized = cv2.resize(binary, CNN_INPUT_SIZE)
    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)
    predictions = cnn_model.predict(input_data, verbose=0)
    return CNN_CLASSES[np.argmax(predictions)]


def draw_overlay(frame, mode, steer_angle):
    overlay = frame.copy()
    cv2.putText(overlay, f"Mode: {mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(overlay, f"Steer: {steer_angle:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return overlay


lock = threading.Lock()
current_speed = 0
current_angle = cfg_vehicle.BASE_ANGLE
prev_speed, prev_angle = None, None
running = True


def motor_control():
    global prev_speed, prev_angle
    while running:
        with lock:
            if current_speed != prev_speed:
                afb.gpio.motor(current_speed)
                prev_speed = current_speed
            if current_angle != prev_angle:
                afb.gpio.servo(current_angle)
                prev_angle = current_angle
        time.sleep(0.05)


frame_queue = deque()


def capture_thread():
    while running:
        frame = afb.camera.get_image()
        if frame is not None:
            frame_queue.append(frame)
        time.sleep(0.01)


def process_thread():
    global current_speed, current_angle
    while running:
        if frame_queue:
            frame = frame_queue.popleft()
            cnn_mode = cnn_predict(frame)

            with lock:
                if cnn_mode == "FORWARD":
                    current_speed = cfg_vehicle.MOTOR_SPEED
                    current_angle = cfg_vehicle.BASE_ANGLE
                elif cnn_mode == "LEFT":
                    current_speed = cfg_vehicle.MOTOR_SPEED
                    current_angle = cfg_vehicle.LEFT_ANGLE
                elif cnn_mode == "RIGHT":
                    current_speed = cfg_vehicle.MOTOR_SPEED
                    current_angle = cfg_vehicle.RIGHT_ANGLE

            overlay = draw_overlay(frame, cnn_mode, current_angle)
            raw_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            overlay_out.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        time.sleep(0.01)


threads = [
    threading.Thread(target=capture_thread, daemon=True),
    threading.Thread(target=process_thread, daemon=True),
    threading.Thread(target=motor_control, daemon=True)
]
for t in threads:
    t.start()

try:
    print("[INFO] driving (Ctrl+C to stop)")
    while running:
        time.sleep(0.1)
except KeyboardInterrupt:
    running = False
finally:
    running = False
    for t in threads: t.join()
    with lock: current_speed = 0
    time.sleep(0.1)
    afb.gpio.stop_all()
    afb.camera.release_camera()
    raw_out.release()
    overlay_out.release()
    print(f"[INFO] recorded {raw_video_name}, {overlay_video_name}")
