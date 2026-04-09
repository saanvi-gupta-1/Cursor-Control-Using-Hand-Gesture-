"""
Advanced Gesture Cursor Control
================================
Control your computer in real time using hand gestures.

Uses the One Euro Filter for adaptive cursor smoothing (CHI 2012),
a stabilized control point blending index tip and DIP,
velocity-proportional scroll and zoom, gesture hysteresis,
and an adaptive dead-zone.

Gesture Reference
-----------------
  Index finger only       -> Move cursor
  Pinch (thumb+index)     -> Left click
  Two quick pinches       -> Double click
  Pinch + middle up       -> Right click
  2 fingers + move up/dn  -> Scroll up / down
  3 fingers + move up     -> Zoom in  (Ctrl ++)
  3 fingers + move down   -> Zoom out (Ctrl --)
  Closed fist (hold)      -> Drag and drop
  Open palm (hold ~0.2s)  -> Screenshot
  Thumb + pinky (shaka)   -> Pause / freeze cursor
  Q                       -> Quit
"""

import cv2
import pyautogui
import numpy as np
import time
import os
import sys
from collections import deque
from hand_tracker import HandTracker
from gesture import GestureDetector
from filters import OneEuroFilter2D

# -- Config -------------------------------------------------------------------
WEBCAM_ID        = 0
FRAME_W, FRAME_H = 640, 480

# One Euro Filter tuning
OEF_FREQ         = 30.0     # expected camera FPS
OEF_MIN_CUTOFF   = 1.5      # Hz - lower = smoother at rest
OEF_BETA         = 0.05     # speed coefficient - higher = less lag at speed
OEF_D_CUTOFF     = 1.0      # derivative filter cutoff

# Screen mapping
MARGIN           = 50       # ignore-zone near frame edges (pixels)

# Scroll / zoom
SCROLL_BASE      = 3        # base scroll lines per tick
SCROLL_MAX       = 20       # max scroll lines per tick
SCROLL_GAIN      = 8.0      # velocity to scroll-lines multiplier
ZOOM_INTERVAL    = 0.10     # min seconds between zoom key presses

# Dead zone (adaptive)
DEAD_ZONE_MIN    = 1        # minimum dead zone (px) during fast movement
DEAD_ZONE_MAX    = 5        # maximum dead zone (px) when nearly still
SPEED_THRESHOLD  = 50       # cursor speed (px/frame) dividing fast vs slow

GESTURE_LOG_LEN  = 6
SCREENSHOT_DIR   = os.path.expanduser("~/Desktop")

# Tip + DIP blend weight for stable control point
TIP_WEIGHT       = 0.75     # 1.0 = pure tip (precise but jittery)

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

# -- Color palette (BGR) ------------------------------------------------------
COLORS = {
    "move":         (0,   255, 120),
    "click":        (0,   200, 255),
    "double_click": (0,   120, 255),
    "right_click":  (180,  60, 255),
    "drag_start":   (255, 140,   0),
    "drag_move":    (255, 180,  40),
    "drag_end":     (255, 220, 100),
    "scroll_up":    (100, 255, 200),
    "scroll_down":  (100, 200, 255),
    "zoom_in":      (0,   255, 255),
    "zoom_out":     (0,   180, 200),
    "screenshot":   (255, 255,   0),
    "pause":        (80,   80, 255),
    "none":         (160, 160, 160),
}

LABELS = {
    "move":         "MOVE",
    "click":        "CLICK",
    "double_click": "DBL CLICK",
    "right_click":  "RT CLICK",
    "drag_start":   "DRAG START",
    "drag_move":    "DRAGGING",
    "drag_end":     "DRAG END",
    "scroll_up":    "SCROLL UP",
    "scroll_down":  "SCROLL DOWN",
    "zoom_in":      "ZOOM IN",
    "zoom_out":     "ZOOM OUT",
    "screenshot":   "SCREENSHOT",
    "pause":        "PAUSED",
    "none":         "...",
}


# -- Helpers -------------------------------------------------------------------

def map_to_screen(x, y, frame_w, frame_h, screen_w, screen_h, margin=MARGIN):
    """Map webcam coords (with margin buffer) to full screen coordinates."""
    x = np.clip(x, margin, frame_w - margin)
    y = np.clip(y, margin, frame_h - margin)
    sx = np.interp(x, [margin, frame_w - margin], [0, screen_w])
    sy = np.interp(y, [margin, frame_h - margin], [0, screen_h])
    return float(sx), float(sy)


def take_screenshot():
    """Save screenshot to Desktop with timestamp."""
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"gesture_screenshot_{ts}.png")
    try:
        img = pyautogui.screenshot()
        img.save(path)
        return path
    except Exception as e:
        return f"Error: {e}"


def draw_gesture_log(frame, log):
    """Draw the last N gestures as a fading sidebar."""
    x_start = frame.shape[1] - 200
    for i, (g, ts) in enumerate(reversed(list(log))):
        age   = time.time() - ts
        alpha = max(0, 1.0 - age / 3.0)
        color = COLORS.get(g, (160, 160, 160))
        faded = tuple(int(c * alpha) for c in color)
        label = LABELS.get(g, g.upper())
        y     = 40 + i * 22
        cv2.putText(frame, label, (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, faded, 1, cv2.LINE_AA)


def draw_hud(frame, gesture, paused, sx, sy, fps, dragging, log,
             scroll_active, zoom_active):
    """Draw HUD overlay on the camera frame."""
    h, w = frame.shape[:2]

    # Top bar with dark overlay
    overlay = frame.copy()
    for row in range(80):
        cv2.rectangle(overlay, (0, row), (w, row + 1), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = COLORS.get(gesture, (255, 255, 255))
    label = LABELS.get(gesture, gesture.upper().replace("_", " "))

    # Gesture indicator dot
    cv2.circle(frame, (22, 30), 8, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (22, 30), 10, (255, 255, 255), 1, cv2.LINE_AA)

    # Gesture name
    cv2.putText(frame, label, (40, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    # Mode indicators
    mode_y = 58
    if scroll_active:
        cv2.putText(frame, "[SCROLL MODE]", (40, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 200), 1, cv2.LINE_AA)
        mode_y += 16
    if zoom_active:
        cv2.putText(frame, "[ZOOM MODE]", (40, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        mode_y += 16

    # Status line
    if paused:
        cv2.putText(frame, "CURSOR FROZEN", (40, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA)
    elif dragging:
        cv2.putText(frame, f"DRAG -> ({sx}, {sy})", (40, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 40), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Cursor ({sx}, {sy})", (40, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # FPS top-right
    if fps >= 24:
        fps_color = (100, 255, 100)
    elif fps >= 15:
        fps_color = (0, 200, 255)
    else:
        fps_color = (0, 80, 255)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 95, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 1, cv2.LINE_AA)

    # Filter label
    cv2.putText(frame, "1-EURO", (w - 95, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 200), 1, cv2.LINE_AA)

    # Gesture log sidebar
    draw_gesture_log(frame, log)

    # Bottom help bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 28), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame,
                "Q:Quit | Pinch:Click | 2-Finger:Scroll | 3-Finger:Zoom | Fist:Drag | Palm:Screenshot",
                (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1, cv2.LINE_AA)

    # Drag ring
    if dragging:
        cx = int(sx * w / pyautogui.size()[0])
        cy = int(sy * h / pyautogui.size()[1])
        cv2.circle(frame, (cx, cy), 20, (255, 140, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 24, (255, 200, 80), 1, cv2.LINE_AA)


def draw_cursor_dot(frame, tip, gesture):
    """Draw a dot at the finger tip with gesture-coloured ring."""
    if tip is None:
        return
    color = COLORS.get(gesture, (255, 255, 255))
    cv2.circle(frame, tip, 7, color, -1, cv2.LINE_AA)
    cv2.circle(frame, tip, 11, (255, 255, 255), 1, cv2.LINE_AA)
    outer = (color[0], color[1], max(color[2] // 2, 40))
    cv2.circle(frame, tip, 16, outer, 1, cv2.LINE_AA)


# -- Main ---------------------------------------------------------------------

def main():
    screen_w, screen_h = pyautogui.size()

    def open_camera():
        for cam_id in [WEBCAM_ID, 0, 1, 2]:
            print(f"  Trying camera ID {cam_id}...")
            c = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                c.set(cv2.CAP_PROP_FPS, 30)
                # Warm up the camera - discard first few frames
                for _ in range(5):
                    c.read()
                print(f"  Camera ID {cam_id} opened successfully.")
                return c
            c.release()
        return None

    cap = open_camera()
    if cap is None:
        print("  Could not open any webcam. Make sure no other app is using it.")
        sys.exit(1)

    tracker  = HandTracker(max_hands=1)
    detector = GestureDetector()

    # One Euro Filter for cursor smoothing
    cursor_filter = OneEuroFilter2D(
        freq=OEF_FREQ,
        min_cutoff=OEF_MIN_CUTOFF,
        beta=OEF_BETA,
        d_cutoff=OEF_D_CUTOFF,
    )

    # State
    prev_sx = prev_sy   = 0
    dragging            = False
    last_screenshot_msg = ""
    last_screenshot_ts  = 0
    gesture_log         = deque(maxlen=GESTURE_LOG_LEN)
    frame_fail_count    = 0
    MAX_FRAME_FAILS     = 30

    # Timing
    last_zoom_time      = 0
    fps_buffer          = deque(maxlen=30)
    prev_time           = time.time()

    print(__doc__)
    print("  Starting... Press Q in the webcam window to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            frame_fail_count += 1
            if frame_fail_count >= MAX_FRAME_FAILS:
                print("  Too many frame failures. Reopening camera...")
                cap.release()
                time.sleep(1)
                cap = open_camera()
                if cap is None:
                    print("  Camera lost. Exiting.")
                    break
                frame_fail_count = 0
            time.sleep(0.05)
            continue
        frame_fail_count = 0

        # FPS
        now       = time.time()
        fps_buffer.append(1.0 / max(now - prev_time, 1e-6))
        fps       = np.mean(fps_buffer)
        prev_time = now

        frame          = cv2.flip(frame, 1)
        frame, results = tracker.find_hands(frame)
        landmarks      = tracker.get_landmarks(results, frame.shape)
        fingers        = tracker.fingers_up(landmarks)

        gesture, tip, extra = detector.detect(landmarks, fingers)

        sx, sy  = prev_sx, prev_sy
        paused  = gesture == GestureDetector.GESTURE_PAUSE

        # Log gesture (avoid spamming none / move / drag_move)
        if gesture not in ("none", "move", "drag_move"):
            if not gesture_log or gesture_log[-1][0] != gesture:
                gesture_log.append((gesture, time.time()))

        # -- Execute actions ---------------------------------------------------
        if tip and not paused:
            # Stabilized control point: blend tip + DIP
            stable_tip = tracker.get_stable_control_point(landmarks, TIP_WEIGHT)
            if stable_tip is None:
                stable_tip = tip

            # Map to screen coordinates
            raw_sx, raw_sy = map_to_screen(
                stable_tip[0], stable_tip[1],
                FRAME_W, FRAME_H, screen_w, screen_h,
            )

            # One Euro Filter
            filtered_x, filtered_y = cursor_filter(raw_sx, raw_sy, now)
            sx = int(np.clip(filtered_x, 0, screen_w - 1))
            sy = int(np.clip(filtered_y, 0, screen_h - 1))

            # Adaptive dead-zone
            speed = np.hypot(sx - prev_sx, sy - prev_sy)
            dead_zone = int(np.interp(speed,
                                      [0, SPEED_THRESHOLD],
                                      [DEAD_ZONE_MAX, DEAD_ZONE_MIN]))
            if abs(sx - prev_sx) < dead_zone and abs(sy - prev_sy) < dead_zone:
                sx, sy = prev_sx, prev_sy

            # -- Gesture actions -----------------------------------------------
            if gesture == GestureDetector.GESTURE_MOVE:
                pyautogui.moveTo(sx, sy)

            elif gesture == GestureDetector.GESTURE_CLICK:
                pyautogui.click(sx, sy)

            elif gesture == GestureDetector.GESTURE_DOUBLE_CLICK:
                pyautogui.doubleClick(sx, sy)

            elif gesture == GestureDetector.GESTURE_RIGHT_CLICK:
                pyautogui.rightClick(sx, sy)

            elif gesture == GestureDetector.GESTURE_DRAG_START:
                pyautogui.moveTo(sx, sy)
                pyautogui.mouseDown()
                dragging = True

            elif gesture == GestureDetector.GESTURE_DRAG_MOVE:
                pyautogui.moveTo(sx, sy)

            elif gesture == GestureDetector.GESTURE_DRAG_END:
                pyautogui.mouseUp()
                dragging = False

            elif gesture in (GestureDetector.GESTURE_SCROLL_UP,
                             GestureDetector.GESTURE_SCROLL_DOWN):
                vel = extra.get("velocity", 1.0)
                amount = int(np.clip(SCROLL_BASE + vel * SCROLL_GAIN,
                                     SCROLL_BASE, SCROLL_MAX))
                if gesture == GestureDetector.GESTURE_SCROLL_UP:
                    pyautogui.scroll(amount)
                else:
                    pyautogui.scroll(-amount)

            elif gesture == GestureDetector.GESTURE_ZOOM_IN:
                if now - last_zoom_time > ZOOM_INTERVAL:
                    pyautogui.hotkey("ctrl", "+")
                    last_zoom_time = now

            elif gesture == GestureDetector.GESTURE_ZOOM_OUT:
                if now - last_zoom_time > ZOOM_INTERVAL:
                    pyautogui.hotkey("ctrl", "-")
                    last_zoom_time = now

            elif gesture == GestureDetector.GESTURE_SCREENSHOT:
                path = take_screenshot()
                last_screenshot_msg = f"Saved: {os.path.basename(path)}"
                last_screenshot_ts  = time.time()

            prev_sx, prev_sy = sx, sy

        elif landmarks is None:
            cursor_filter.reset()

        # Safety: release mouse if hand disappears while dragging
        if landmarks is None and dragging:
            pyautogui.mouseUp()
            dragging = False

        # -- Draw --------------------------------------------------------------
        draw_cursor_dot(frame, tip, gesture)
        draw_hud(frame, gesture, paused, sx, sy, fps, dragging, gesture_log,
                 detector._scroll_active, detector._zoom_active)

        # Screenshot flash notification
        if last_screenshot_msg and time.time() - last_screenshot_ts < 2.5:
            flash_overlay = frame.copy()
            cv2.rectangle(flash_overlay,
                          (FRAME_W // 2 - 165, FRAME_H // 2 - 20),
                          (FRAME_W // 2 + 165, FRAME_H // 2 + 10),
                          (0, 80, 80), -1)
            cv2.addWeighted(flash_overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, last_screenshot_msg,
                        (FRAME_W // 2 - 150, FRAME_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Gesture Cursor Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("  Exited cleanly.")


if __name__ == "__main__":
    main()