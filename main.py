"""
Advanced Gesture Cursor Control
================================
Control your entire computer in real time using hand gestures.

Features
--------
* Double-exponential (Holt) cursor smoothing – buttery movement, no jitter
* Dead-zone filtering – micro-tremors are ignored
* Frame-confirmed gestures – no accidental clicks or screenshots
* Adaptive pinch threshold – works at any hand-to-camera distance
* Professional HUD with gesture history, FPS, and status indicators

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
  🖐  Open palm (hold ~0.2s) → Screenshot
  🖖  3 fingers up           → Pause / freeze cursor
  Q                          → Quit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ── Config ──────────────────────────────────────────────────────────────────
WEBCAM_ID        = 0
FRAME_W, FRAME_H = 640, 480

# Smoothing
SMOOTH_ALPHA     = 0.35      # Holt smoother – level weight  (0-1: lower = smoother)
SMOOTH_BETA      = 0.15      # Holt smoother – trend weight  (0-1: lower = smoother)
DEAD_ZONE        = 4         # ignore cursor moves smaller than this (px)

MARGIN           = 100       # ignore-zone near frame edges (pixels)
SCROLL_AMOUNT    = 3         # lines per scroll tick

GESTURE_LOG_LEN  = 6         # how many past gestures to display in HUD
SCREENSHOT_DIR   = os.path.expanduser("~/Desktop")

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.008   # 8ms – prevents flooding OS with mouse events

# ── Color palette (BGR) ────────────────────────────────────────────────────
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


# ── Holt Double-Exponential Smoother ────────────────────────────────────────

class HoltSmoother:
    """
    Double-exponential (Holt) smoother for 2-D cursor positions.

    Predicts the *next* position using both the current level and trend,
    giving butter-smooth movement without the lag of a simple average.
    """

    def __init__(self, alpha=SMOOTH_ALPHA, beta=SMOOTH_BETA):
        self.alpha  = alpha
        self.beta   = beta
        self.level  = None   # (x, y) smoothed position
        self.trend  = None   # (dx, dy) velocity estimate
        self._initialized = False

    def update(self, raw_x, raw_y):
        """Feed a new raw position and return the smoothed position."""
        if not self._initialized:
            self.level = np.array([raw_x, raw_y], dtype=np.float64)
            self.trend = np.array([0.0, 0.0])
            self._initialized = True
            return int(raw_x), int(raw_y)

        raw       = np.array([raw_x, raw_y], dtype=np.float64)
        prev_lvl  = self.level.copy()
        self.level = self.alpha * raw + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - prev_lvl) + (1 - self.beta) * self.trend

        return int(self.level[0]), int(self.level[1])

    def reset(self):
        """Reset when hand disappears."""
        self._initialized = False


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
        label = EMOJI.get(g, g.upper())
        y     = 40 + i * 22
        cv2.putText(frame, label, (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, faded, 1, cv2.LINE_AA)


def draw_hud(frame, gesture, paused, sx, sy, fps, dragging, log):
    """Draw full professional HUD overlay."""
    h, w = frame.shape[:2]

    # ── Top bar with gradient ────────────────────────────────────────────
    overlay = frame.copy()
    # Draw a gradient bar (dark at top, fading out)
    for row in range(80):
        alpha_row = 0.75 - (row / 80) * 0.35
        cv2.rectangle(overlay, (0, row), (w, row + 1), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = COLORS.get(gesture, (255, 255, 255))
    label = EMOJI.get(gesture, gesture.upper().replace("_", " "))

    # Gesture indicator dot
    cv2.circle(frame, (22, 30), 8, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (22, 30), 10, (255, 255, 255), 1, cv2.LINE_AA)

    # Gesture name
    cv2.putText(frame, label, (40, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    # Status line
    if paused:
        status = "CURSOR FROZEN"
        cv2.putText(frame, status, (40, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA)
    elif dragging:
        status = f"DRAG -> ({sx}, {sy})"
        cv2.putText(frame, status, (40, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 40), 1, cv2.LINE_AA)
    else:
        status = f"Cursor ({sx}, {sy})"
        cv2.putText(frame, status, (40, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # FPS top-right with color coding
    fps_color = (100, 255, 100) if fps >= 24 else (0, 200, 255) if fps >= 15 else (0, 80, 255)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 95, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 1, cv2.LINE_AA)

    # Gesture log sidebar
    draw_gesture_log(frame, log)

    # ── Bottom help bar ──────────────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 28), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame,
                "Q:Quit | Pinch:Click | 2-Finger:Scroll | Fist:Drag | Palm(hold):Screenshot",
                (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1, cv2.LINE_AA)

    # Drag ring around cursor on the frame
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
    cv2.circle(frame, tip, 7,  color,            -1, cv2.LINE_AA)
    cv2.circle(frame, tip, 11, (255, 255, 255),   1, cv2.LINE_AA)
    # Subtle outer glow
    cv2.circle(frame, tip, 16, (*color[:2], max(color[2] // 2, 40)), 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    screen_w, screen_h = pyautogui.size()

    # ── Open webcam with retry ───────────────────────────────────────────
    cap = None
    for attempt in range(3):
        cap = cv2.VideoCapture(WEBCAM_ID)
        if cap.isOpened():
            break
        print(f"⚠  Webcam not found (attempt {attempt + 1}/3), retrying...")
        time.sleep(1)

    if cap is None or not cap.isOpened():
        print("❌  Could not open webcam. Exiting.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency

    tracker  = HandTracker(max_hands=1)
    detector = GestureDetector()
    smoother = HoltSmoother()

    # State
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
            print("⚠  Frame read failed, retrying...")
            time.sleep(0.05)
            continue

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

        # Log gesture (avoid spamming none / move / drag_move)
        if gesture not in ("none", "move", "drag_move"):
            if not gesture_log or gesture_log[-1][0] != gesture:
                gesture_log.append((gesture, time.time()))

        # ── Execute actions ────────────────────────────────────────────────
        if tip and not paused:
            raw_sx, raw_sy = map_to_screen(
                tip[0], tip[1], FRAME_W, FRAME_H, screen_w, screen_h
            )

            # Holt double-exponential smoothing
            sx, sy = smoother.update(raw_sx, raw_sy)

            # Dead-zone: skip micro-movements to prevent jitter
            if abs(sx - prev_sx) < DEAD_ZONE and abs(sy - prev_sy) < DEAD_ZONE:
                sx, sy = prev_sx, prev_sy

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

        elif landmarks is None:
            # Hand disappeared → reset smoother so it doesn't "jump" on return
            smoother.reset()

        # Safety: release mouse if hand disappears while dragging
        if landmarks is None and dragging:
            pyautogui.mouseUp()
            dragging = False

        # ── Draw ──────────────────────────────────────────────────────────
        draw_cursor_dot(frame, tip, gesture)
        draw_hud(frame, gesture, paused, sx, sy, fps, dragging, gesture_log)

        # Screenshot flash notification
        if last_screenshot_msg and time.time() - last_screenshot_ts < 2.5:
            # Semi-transparent flash background
            flash_overlay = frame.copy()
            cv2.rectangle(flash_overlay,
                          (FRAME_W // 2 - 165, FRAME_H // 2 - 20),
                          (FRAME_W // 2 + 165, FRAME_H // 2 + 10),
                          (0, 80, 80), -1)
            cv2.addWeighted(flash_overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, last_screenshot_msg,
                        (FRAME_W // 2 - 150, FRAME_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

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