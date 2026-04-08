"""
Gesture Cursor Control
======================
Control your mouse cursor in real time using hand gestures.

Gestures:
  ☝️  Index finger up       → Move cursor
  🤏  Pinch (thumb+index)  → Left click
  🤏🤏 Double pinch         → Double click
  ✌️  Index + Middle up    → Pause (freeze cursor)
  Q                        → Quit
"""

import cv2
import pyautogui
import numpy as np
from hand_tracker import HandTracker
from gesture import GestureDetector

# ── Config ──────────────────────────────────────────────────────────────────
WEBCAM_ID        = 0
FRAME_W, FRAME_H = 640, 480
SMOOTH_FACTOR    = 5       # higher = smoother but slightly laggier (1–10)
MARGIN           = 80      # pixel margin to ignore near frame edges
pyautogui.FAILSAFE = False  # disable corner-escape failsafe

# ── Helpers ──────────────────────────────────────────────────────────────────

def map_to_screen(x, y, frame_w, frame_h, screen_w, screen_h, margin=MARGIN):
    """Map webcam coords (with margin) to full screen coordinates."""
    x = np.clip(x, margin, frame_w - margin)
    y = np.clip(y, margin, frame_h - margin)
    sx = np.interp(x, [margin, frame_w - margin], [0, screen_w])
    sy = np.interp(y, [margin, frame_h - margin], [0, screen_h])
    return int(sx), int(sy)


def draw_hud(frame, gesture, paused, sx, sy):
    """Draw gesture label and status onto the frame."""
    color_map = {
        "move":         (0, 255, 100),
        "click":        (0, 200, 255),
        "double_click": (0, 100, 255),
        "pause":        (60, 60, 255),
        "none":         (180, 180, 180),
    }
    color = color_map.get(gesture, (255, 255, 255))
    label = f"Gesture: {gesture.upper().replace('_', ' ')}"
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    status = "PAUSED" if paused else f"Cursor → ({sx}, {sy})"
    cv2.putText(frame, status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, "Q: Quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    screen_w, screen_h = pyautogui.size()
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    tracker  = HandTracker(max_hands=1)
    detector = GestureDetector()

    # Smoothing buffers
    smooth_x = smooth_y = 0
    prev_sx   = prev_sy = 0

    print("✅ Gesture Cursor Control started. Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural feel
        frame, results = tracker.find_hands(frame)
        landmarks = tracker.get_landmarks(results, frame.shape)
        fingers   = tracker.fingers_up(landmarks)

        gesture, tip = detector.detect(landmarks, fingers)

        sx, sy  = prev_sx, prev_sy
        paused  = gesture == GestureDetector.GESTURE_PAUSE

        if tip and not paused:
            raw_sx, raw_sy = map_to_screen(tip[0], tip[1], FRAME_W, FRAME_H, screen_w, screen_h)

            # Exponential smoothing
            smooth_x = smooth_x + (raw_sx - smooth_x) / SMOOTH_FACTOR
            smooth_y = smooth_y + (raw_sy - smooth_y) / SMOOTH_FACTOR
            sx, sy = int(smooth_x), int(smooth_y)

            if gesture == GestureDetector.GESTURE_MOVE:
                pyautogui.moveTo(sx, sy)

            elif gesture == GestureDetector.GESTURE_CLICK:
                pyautogui.click(sx, sy)

            elif gesture == GestureDetector.GESTURE_DOUBLE_CLICK:
                pyautogui.doubleClick(sx, sy)

            prev_sx, prev_sy = sx, sy

        draw_hud(frame, gesture, paused, sx, sy)
        cv2.imshow("Gesture Cursor Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Exited.")


if __name__ == "__main__":
    main()
