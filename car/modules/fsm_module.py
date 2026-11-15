#!/usr/bin/env python3

import time

from modules.drive_module import (
    BASE_ANGLE, forward, turn_left, turn_right, stop
)

THETA0 = 3.0
THETA2 = 11.0


def _drive_by_angle(angle):
    if angle is None:
        stop(); return "STOP"
    e = float(angle) - float(BASE_ANGLE)
    if abs(e) <= THETA0:
        forward(angle=angle); return "FORWARD"
    if e <= -THETA2:
        turn_left(); return "LEFT"
    if e >= THETA2:
        turn_right(); return "RIGHT"
    else:
        forward(angle=angle); return "FORWARD"


class LaneFSM:
    def __init__(self):
        self.state = 1
        self._t0 = None
        self._seq_done = False

    def reset_timer(self):
        self._t0 = time.time()
        self._seq_done = False

    def elapsed(self):
        if self._t0 is None: return 0.0
        return time.time() - self._t0

    def update(self, angle, selected_label, key_start=False):
        s = self.state

        if s == 1:
            stop()
            if key_start:
                self.state = 2
                self.reset_timer()
            return self.state

        if s == 2:
            _drive_by_angle(angle-5)
            if selected_label == "smallhill":
                self.state = 3
                self.reset_timer()
            return self.state

        if s == 3:
            t = self.elapsed()
            if t < 2.5:
                forward(speed=120, angle=angle-5)
            else:
                _drive_by_angle(angle-5)
                if selected_label == "rightsign":
                    self.state = 4
                    self.reset_timer()
                return self.state

        if s == 4:
            t = self.elapsed()
            if t < 5.0:
                turn_right()
            else:
                _drive_by_angle(angle-5)
                if selected_label == "gateclosed":
                    self.state = 5
                    self.reset_timer()
                return self.state

        if s == 5:
            stop()
            if selected_label == "gateopen" or selected_label == "None":
                self.state = 6
                self.reset_timer()
            return self.state

        if s == 6:
            t = self.elapsed()
            _drive_by_angle(angle-5)
            if selected_label == "staticb":
                self.state = 7
                self.reset_timer()
            return self.state

        if s == 7:
            t = self.elapsed()
            if t < 3.0:
                turn_left()
            else:
                _drive_by_angle(angle-5)
                if selected_label == "staticy":
                    self.state = 8
                    self.reset_timer()
                return self.state

        if s == 8:
            t = self.elapsed()
            if t < 3.0:
                turn_right()
            else:
                _drive_by_angle(angle-5)
                if selected_label == "leftlight":
                    self.state = 9
                    self.reset_timer()
                return self.state

        if s == 9:
            stop()
            t = self.elapsed()
            if selected_label == "greensign" or t > 8.0:
                self.state = 10
                self.reset_timer()
            return self.state

        if s == 10:
            t = self.elapsed()
            if t < 5.0:
                turn_left()
            else:
                _drive_by_angle(angle-5)
            return self.state
