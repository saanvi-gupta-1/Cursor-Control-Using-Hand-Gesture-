"""
Advanced Gesture Cursor Control
================================
Control your entire computer in real time using hand gestures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GESTURE CHEAT SHEET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ☝️  Index finger only       → Move cursor
  🤏  Pinch (thumb+index)    → Left click
  🤏🤏 Two quick pinches      → Double click
  🤏✌  Pinch + middle up     → Right click
  ✌️  Two fingers + move up  → Scroll up
  ✌️  Two fingers + move dn  → Scroll down
  🤙  Thumb + pinky spread   → Zoom in  (Ctrl ++)
  🤙  Thumb + pinky close    → Zoom out (Ctrl --)
  ✊  Closed fist (hold)     → Drag & drop
  🖐  Open palm (all 5 up)   → Screenshot
  🖖  3 fingers up           → Pause / freeze cursor
  Q                          → Quit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import pyautogui
import numpy as np
import time
import os
from collections import deque
from hand_tracker import HandTracker
from gesture import GestureDetector

# ── Config ──────────────────────────────────────────────────────────────────
WEBCAM_ID        = 0
FRAME_W, FRAME_H = 640, 480
SMOOTH_FACTOR    = 6        # 1-10: higher = smoother but slightly laggier
MARGIN           = 80       # ignore-zone near frame edges (pixels)
SCROLL_AMOUNT    = 3        # lines per scroll tick
GESTURE_LOG_LEN  = 6        # how many past gestures to display in HUD
SCREENSHOT_DIR   = os.path.expanduser("~/Desktop")   # where screenshots go

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0       # remove pyautogui default delay

# ── Color palette ───────────────────────────────────────────────────────────
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

EMOJI = {
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
    "screenshot":   "SCREENSHOT!",
    "pause":        "PAUSED",
    "none":         "...",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def map_to_screen(x, y, frame_w, frame_h, screen_w, screen_h, margin=MARGIN):
    """Map webcam coords (with margin buffer) → full screen coordinates."""
    x  = np.clip(x, margin, frame_w - margin)
    y  = np.clip(y, margin, frame_h - margin)
    sx = np.interp(x, [margin, frame_w - margin], [0, screen_w])
    sy = np.interp(y, [margin, frame_h - margin], [0, screen_h])
    return int(sx), int(sy)


def take_screenshot():
    """Save screenshot to Desktop with timestamp."""
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"gesture_screenshot_{ts}.png")
    try:
        img  = pyautogui.screenshot()
        img.save(path)
        return path
    except Exception as e:
        return f"Error: {e}"


def draw_gesture_log(frame, log):
    """Draw the last N gestures as a fading sidebar."""
    x_start = frame.shape[1] - 200
    for i, (g, ts) in enumerate(reversed(list(log))):
        age     = time.time() - ts
        alpha   = max(0, 1.0 - age / 3.0)  # fade over 3 seconds
        color   = COLORS.get(g, (160, 160, 160))
        faded   = tuple(int(c * alpha) for c in color)
        label   = EMOJI.get(g, g.upper())
        y       = 40 + i * 22
        cv2.putText(frame, label, (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, faded, 1)


def draw_hud(frame, gesture, paused, sx, sy, fps, dragging, log):
    """Draw full HUD overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = COLORS.get(gesture, (255, 255, 255))
    label = EMOJI.get(gesture, gesture.upper().replace("_", " "))

    # Gesture name (big)
    cv2.putText(frame, label, (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Status line
    if paused:
        status = "  CURSOR FROZEN"
    elif dragging:
        status = f"  DRAG -> ({sx}, {sy})"
    else:
        status = f"  Cursor -> ({sx}, {sy})"
    cv2.putText(frame, status, (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # FPS top-right
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

    # Gesture log sidebar
    draw_gesture_log(frame, log)

    # Bottom help bar
    cv2.rectangle(frame, (0, h - 24), (w, h), (10, 10, 20), -1)
    cv2.putText(frame, "Q:Quit | Pinch:Click | 2-Finger:Scroll | Fist:Drag | Palm:Screenshot",
                (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    # Drag indicator ring around cursor when dragging
    if dragging:
        cx = int(sx * frame.shape[1] / pyautogui.size()[0])
        cy = int(sy * frame.shape[0] / pyautogui.size()[1])
        cv2.circle(frame, (cx, cy), 18, (255, 140, 0), 2)


def draw_cursor_dot(frame, tip, gesture):
    """Draw a dot at the finger tip with gesture-coloured ring."""
    if tip is None:
        return
    color = COLORS.get(gesture, (255, 255, 255))
    cv2.circle(frame, tip, 8,  color,            -1)
    cv2.circle(frame, tip, 12, (255, 255, 255),   1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    screen_w, screen_h = pyautogui.size()
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency

    tracker  = HandTracker(max_hands=1)
    detector = GestureDetector()

    # State
    smooth_x = smooth_y = 0.0
    prev_sx = prev_sy   = 0
    dragging            = False
    last_screenshot_msg = ""
    last_screenshot_ts  = 0
    gesture_log         = deque(maxlen=GESTURE_LOG_LEN)

    # FPS tracking
    fps_buffer = deque(maxlen=30)
    prev_time  = time.time()

    print(__doc__)
    print("✅  Starting... Press Q in the webcam window to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # FPS
        now       = time.time()
        fps_buffer.append(1.0 / max(now - prev_time, 1e-6))
        fps       = np.mean(fps_buffer)
        prev_time = now

        frame          = cv2.flip(frame, 1)
        frame, results = tracker.find_hands(frame)
        landmarks      = tracker.get_landmarks(results, frame.shape)
        fingers        = tracker.fingers_up(landmarks)

        gesture, tip, _extra = detector.detect(landmarks, fingers)

        sx, sy  = prev_sx, prev_sy
        paused  = gesture == GestureDetector.GESTURE_PAUSE

        # Log gesture (avoid spamming none/move)
        if gesture not in ("none", "move", "drag_move"):
            if not gesture_log or gesture_log[-1][0] != gesture:
                gesture_log.append((gesture, time.time()))

        # ── Execute actions ────────────────────────────────────────────────
        if tip and not paused:
            raw_sx, raw_sy = map_to_screen(
                tip[0], tip[1], FRAME_W, FRAME_H, screen_w, screen_h
            )

            # Exponential smoothing
            smooth_x += (raw_sx - smooth_x) / SMOOTH_FACTOR
            smooth_y += (raw_sy - smooth_y) / SMOOTH_FACTOR
            sx, sy    = int(smooth_x), int(smooth_y)

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

            elif gesture == GestureDetector.GESTURE_SCROLL_UP:
                pyautogui.scroll(SCROLL_AMOUNT)

            elif gesture == GestureDetector.GESTURE_SCROLL_DOWN:
                pyautogui.scroll(-SCROLL_AMOUNT)

            elif gesture == GestureDetector.GESTURE_ZOOM_IN:
                pyautogui.hotkey("ctrl", "+")

            elif gesture == GestureDetector.GESTURE_ZOOM_OUT:
                pyautogui.hotkey("ctrl", "-")

            elif gesture == GestureDetector.GESTURE_SCREENSHOT:
                path = take_screenshot()
                last_screenshot_msg = f"Saved: {os.path.basename(path)}"
                last_screenshot_ts  = time.time()

            prev_sx, prev_sy = sx, sy

        # Safety: release mouse if hand disappears while dragging
        if landmarks is None and dragging:
            pyautogui.mouseUp()
            dragging = False

        # ── Draw ──────────────────────────────────────────────────────────
        draw_cursor_dot(frame, tip, gesture)
        draw_hud(frame, gesture, paused, sx, sy, fps, dragging, gesture_log)

        # Screenshot flash notification
        if last_screenshot_msg and time.time() - last_screenshot_ts < 2.5:
            cv2.putText(frame, last_screenshot_msg,
                        (FRAME_W // 2 - 150, FRAME_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Advanced Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("👋  Exited cleanly.")


if __name__ == "__main__":
    main()