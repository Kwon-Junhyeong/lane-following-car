#!/usr/bin/env python3

import numpy as np
import cv2
from modules.preprocessing_module import PreprocessingModule


class SteeringAnalyzer:
    def __init__(self,
                 base_angle: float = 78.5,
                 max_offset: float = 48.5,
                 angle_limits: tuple = (30.0, 150.0),
                 y_bottom_ratio: float = 0.70,
                 y_top_ratio: float = 0.40,
                 k_head: float = 0.60):
        self.base_angle = float(base_angle)
        self.max_offset = float(max_offset)
        self.angle_limits = (float(angle_limits[0]), float(angle_limits[1]))
        self.yb_ratio = float(y_bottom_ratio)
        self.yt_ratio = float(y_top_ratio)
        self.k_head = float(k_head)
        self.pre = PreprocessingModule()

    def _compute_scan_points(self, binary: np.ndarray):
        if binary is None or binary.ndim != 2:
            return None

        h, w = binary.shape
        yb = int(np.clip(h * self.yb_ratio, 0, h - 1))
        yt = int(np.clip(h * self.yt_ratio, 0, h - 1))

        def center_on_row(y: int):
            line = binary[y, :]
            xs = np.flatnonzero(line > 0)
            if xs.size < 2:
                return None
            return int((xs[0] + xs[-1]) // 2)

        c_bot = center_on_row(yb)
        c_top = center_on_row(yt)

        return {
            "yb": yb, "yt": yt,
            "c_bot": c_bot, "c_top": c_top,
            "w": w, "h": h,
            "center_x": w // 2
        }

    def _estimate_angle(self, binary: np.ndarray, pts: dict | None) -> float:
        if binary is None or binary.ndim != 2 or not pts:
            return float(np.clip(self.base_angle, *self.angle_limits))

        yb, yt = pts["yb"], pts["yt"]
        c_bot, c_top = pts["c_bot"], pts["c_top"]
        w = pts["w"]

        if c_bot is None or c_top is None:
            return float(np.clip(self.base_angle, *self.angle_limits))

        dx_lat_norm = (c_bot - (w // 2)) / (w / 2)

        dy = max(1, abs(yb - yt))
        dx = (c_top - c_bot)
        heading_deg = float(np.degrees(np.arctan2(dx, dy)))

        angle = self.base_angle + dx_lat_norm * self.max_offset + self.k_head * heading_deg
        return float(np.clip(angle, *self.angle_limits))

    def analyze_binary(self, binary: np.ndarray) -> dict:
        pts = self._compute_scan_points(binary)
        angle = self._estimate_angle(binary, pts)
        return {"angle": angle, "points": pts}

    def analyze_frame(self, frame_bgr) -> dict:
        binary = self.pre.process(frame_bgr)
        pts = self._compute_scan_points(binary)
        angle = self._estimate_angle(binary, pts)

        src = self.pre.src_points.copy()
        dst = self.pre.dst_points.copy()
        M = self.pre.M.copy()
        Minv = cv2.getPerspectiveTransform(dst, src)

        return {
            "angle": angle,
            "points": pts,
            "homography": {
                "src_points": src,
                "dst_points": dst,
                "M": M,
                "Minv": Minv
            },
            "binary": binary
        }
