#!/usr/bin/env python3
"""Single scan row steering, tried on still images before the two-row steering of car/modules.

One row of the warped, thresholded image; the lane center is the middle of the two lines, or one line
shifted by half of an assumed lane width when only one line crosses the row. The angle is proportional
to the offset of that center from the image center. The vehicle values are those of config_vehicle.py,
copied here because that file needs the kit's afb package.

usage: python cnn/single_row_test.py IMAGE_DIR
"""
import os
import sys

import cv2
import numpy as np

import config_preprocessing as cfg_pre

IMG_WIDTH, IMG_HEIGHT = 640, 480
SCAN_Y = 240
CENTER_X = IMG_WIDTH // 2
KP = 0.3
BASE_ANGLE = 78.5
ANGLE_LIMITS = (40, 140)
LANE_WIDTH = 300
previous_angle = BASE_ANGLE

M = cv2.getPerspectiveTransform(np.float32(cfg_pre.SRC_POINTS), np.float32(cfg_pre.DST_POINTS))


def get_lane_points(binary_img):
    scan_line = binary_img[SCAN_Y, :]
    edges = list(np.where(scan_line[:-1] != scan_line[1:])[0])
    if scan_line[0] == 255:
        edges.insert(0, 0)
    if scan_line[-1] == 255:
        edges.append(len(scan_line) - 1)
    return edges


def calculate_lane_center(edges):
    if len(edges) == 4:
        return (edges[1] + edges[-2]) // 2
    elif len(edges) == 2:
        left_edge, right_edge = edges[0], edges[1]
        if left_edge < CENTER_X:
            return left_edge + LANE_WIDTH // 2
        else:
            return right_edge - LANE_WIDTH // 2
    else:
        return None


def calculate_steer_angle(lane_center):
    offset = lane_center - CENTER_X
    steer_angle = BASE_ANGLE - (offset * KP)
    return max(ANGLE_LIMITS[0], min(ANGLE_LIMITS[1], steer_angle))


def process_single_image(img_path):
    global previous_angle
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[ERROR] cannot read {img_path}")
        return
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    bev_frame = cv2.warpPerspective(frame, M, cfg_pre.BEV_SIZE)
    gray = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2GRAY)
    if cfg_pre.USE_BLUR:
        gray = cv2.GaussianBlur(gray, cfg_pre.BLUR_KERNEL, 0)

    if cfg_pre.THRESHOLD_TYPE == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif cfg_pre.THRESHOLD_TYPE == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            cfg_pre.ADAPTIVE_BLOCK_SIZE, cfg_pre.ADAPTIVE_C
        )

    edges = get_lane_points(binary)
    lane_center = calculate_lane_center(edges)
    steer_angle = previous_angle
    overlay_original = frame.copy()
    overlay_bev = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    if lane_center is not None:
        steer_angle = calculate_steer_angle(lane_center)
        previous_angle = steer_angle
        for img in (overlay_original, overlay_bev):
            cv2.line(img, (CENTER_X, SCAN_Y - 20), (CENTER_X, SCAN_Y + 20), (255, 0, 0), 2)
            cv2.line(img, (lane_center, SCAN_Y - 20), (lane_center, SCAN_Y + 20), (0, 255, 0), 2)
            for e in edges:
                cv2.circle(img, (e, SCAN_Y), 5, (0, 0, 255), -1)

    for img in (overlay_original, overlay_bev):
        cv2.putText(img, f"Steer Angle: {steer_angle:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    print(f"[RESULT] {img_path} | steer angle: {steer_angle:.1f}")

    combined = cv2.hconcat([overlay_original, overlay_bev])
    cv2.imshow("Lane Detection (Left: Original | Right: BEV Binary+Overlay)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_folder_images(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    for img_path in sorted(image_paths):
        process_single_image(img_path)


if __name__ == "__main__":
    process_folder_images(sys.argv[1])
