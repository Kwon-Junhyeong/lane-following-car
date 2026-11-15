#!/usr/bin/env python3

try:
    import afb
    _HAS_AFB = True
except Exception:
    _HAS_AFB = False

BASE_ANGLE = 78.5
BASE_SPEED = 60

current_mode = "STOP"
current_speed = 0


def get_current_mode():
    return current_mode


def get_current_speed():
    return current_speed


def forward(speed=None, angle=None):
    global current_mode, current_speed
    v = BASE_SPEED if speed is None else speed
    a = BASE_ANGLE if angle is None else angle
    if _HAS_AFB:
        afb.gpio.servo(a)
        afb.gpio.motor(80)
    current_mode = "FORWARD"
    current_speed = v


def turn_left():
    global current_mode, current_speed
    if _HAS_AFB:
        afb.gpio.servo(33)
        afb.gpio.motor(60)
    current_mode = "LEFT"
    current_speed = 50


def turn_right():
    global current_mode, current_speed
    if _HAS_AFB:
        afb.gpio.servo(147)
        afb.gpio.motor(60)
    current_mode = "RIGHT"
    current_speed = 50


def stop():
    global current_mode, current_speed
    if _HAS_AFB:
        afb.gpio.motor(0)
    current_mode = "STOP"
    current_speed = 0
