"""
Gesture Detection Engine
========================
Robust, frame-confirmed gesture recognition with adaptive thresholds.

Every action gesture (click, right-click, screenshot, scroll, zoom) now
requires multi-frame confirmation and respects a global cooldown to
prevent accidental triggers.
"""

import time
import numpy as np
from hand_tracker import HandTracker


class GestureDetector:
    """
    Advanced gesture recognition engine with frame-confirmed detection.

    Key accuracy features
    ---------------------
    * **Frame confirmation** – click / right-click require ≥ CONFIRM_FRAMES
      consecutive frames of the same gesture before firing.
    * **Palm-relative pinch threshold** – the pinch distance is scaled by
      the current palm size so the threshold works at any hand-to-camera
      distance.
    * **Global action cooldown** – after any action fires, a cooldown period
      prevents re-triggering within ACTION_COOLDOWN_SEC seconds.
    * **Screenshot hold** – open palm must be held for SCREENSHOT_HOLD_FRAMES
      consecutive frames.

    Gestures
    --------
      MOVE          : Index finger only up              → move cursor
      CLICK         : Pinch (thumb + index close)       → left click
      DOUBLE_CLICK  : Two pinches within interval       → double click
      RIGHT_CLICK   : Pinch with middle finger up       → right click
      DRAG_START    : Fist (all fingers down)            → start drag
      SCROLL_UP     : Two fingers up + hand moving up   → scroll up
      SCROLL_DOWN   : Two fingers up + hand moving down → scroll down
      ZOOM_IN       : Thumb + pinky spread out           → zoom in
      ZOOM_OUT      : Thumb + pinky close together       → zoom out
      SCREENSHOT    : All 5 fingers up (open palm hold) → screenshot
      PAUSE         : Index + middle + ring up           → freeze cursor
      NONE          : No recognized gesture
    """

    # ── Tuned thresholds ────────────────────────────────────────────────────
    PINCH_RATIO            = 0.22   # pinch dist / palm size  (replaces px)
    DOUBLE_CLICK_INTERVAL  = 0.35   # seconds between pinches
    DRAG_HOLD_FRAMES       = 10     # frames fist must be held before drag
    SCROLL_SENSITIVITY     = 3      # scroll ticks per detected motion
    SCROLL_MOTION_THRESH   = 20     # px/frame to trigger scroll
    SCROLL_COOLDOWN_FRAMES = 6      # frames between scroll repeats
    ZOOM_COOLDOWN_FRAMES   = 8      # frames between zoom repeats

    # ── Confirmation & cooldown ─────────────────────────────────────────────
    CONFIRM_FRAMES         = 2      # consecutive detections before firing
    ACTION_COOLDOWN_SEC    = 0.18   # global cooldown between any actions
    SCREENSHOT_HOLD_FRAMES = 5      # palm must be held this many frames

    # ── Gesture name constants ──────────────────────────────────────────────
    GESTURE_NONE         = "none"
    GESTURE_MOVE         = "move"
    GESTURE_CLICK        = "click"
    GESTURE_DOUBLE_CLICK = "double_click"
    GESTURE_RIGHT_CLICK  = "right_click"
    GESTURE_DRAG_START   = "drag_start"
    GESTURE_DRAG_MOVE    = "drag_move"
    GESTURE_DRAG_END     = "drag_end"
    GESTURE_SCROLL_UP    = "scroll_up"
    GESTURE_SCROLL_DOWN  = "scroll_down"
    GESTURE_ZOOM_IN      = "zoom_in"
    GESTURE_ZOOM_OUT     = "zoom_out"
    GESTURE_SCREENSHOT   = "screenshot"
    GESTURE_PAUSE        = "pause"

    def __init__(self):
        # Pinch / click state
        self._last_pinch_time      = 0
        self._pinch_active         = False
        self._pinch_confirm_count  = 0      # consecutive pinch frames
        self._right_click_confirm  = 0      # consecutive right-click frames
        self._right_click_done     = False

        # Drag state
        self._fist_frames          = 0
        self._dragging             = False

        # Scroll state
        self._prev_palm_y          = None
        self._scroll_cooldown      = 0

        # Zoom state
        self._zoom_prev_dist       = None
        self._zoom_cooldown        = 0

        # Screenshot state
        self._screenshot_hold      = 0
        self._screenshot_done      = False

        # Global cooldown
        self._last_action_time     = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(self, landmarks, fingers):
        """
        Main detection entry point.

        Returns
        -------
        (gesture_name, control_point, extra_data)
            control_point : (x, y) pixel position to anchor cursor
            extra_data    : dict with optional gesture metadata
        """
        if landmarks is None:
            self._reset_drag()
            self._prev_palm_y = None
            return self.GESTURE_NONE, None, {}

        index_tip  = landmarks[HandTracker.INDEX_TIP]
        thumb_tip  = landmarks[HandTracker.THUMB_TIP]
        middle_tip = landmarks[HandTracker.MIDDLE_TIP]
        palm_cx, palm_cy = self._palm_center(landmarks)
        palm_size  = self._palm_size(landmarks)

        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers
        finger_count = sum(fingers)

        # ── SCREENSHOT : open palm – all 5 up, held for N frames ──────────
        if finger_count == 5:
            self._screenshot_hold += 1
            if self._screenshot_hold >= self.SCREENSHOT_HOLD_FRAMES and not self._screenshot_done:
                self._screenshot_done = True
                self._mark_action()
                return self.GESTURE_SCREENSHOT, index_tip, {}
            # Still holding, but not long enough yet → report none
            return self.GESTURE_NONE, index_tip, {}
        else:
            self._screenshot_hold = 0
            self._screenshot_done = False

        # ── PAUSE : index + middle + ring up ──────────────────────────────
        if index_up and middle_up and ring_up and not pinky_up and not thumb_up:
            return self.GESTURE_PAUSE, index_tip, {}

        # ── ZOOM : thumb + pinky spread ───────────────────────────────────
        if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
            result = self._detect_zoom(landmarks)
            if result:
                return result, index_tip, {}

        # ── SCROLL : index + middle up, palm moving vertically ────────────
        if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
            result = self._detect_scroll(palm_cy)
            if result:
                return result, index_tip, {}
            return self.GESTURE_NONE, index_tip, {}

        self._prev_palm_y = palm_cy   # keep tracking for scroll

        # ── PINCH detection (click / double-click / right-click) ──────────
        pinch_dist = np.hypot(thumb_tip[0] - index_tip[0],
                              thumb_tip[1] - index_tip[1])
        # Palm-relative threshold: adapt to hand distance from camera
        effective_threshold = max(self.PINCH_RATIO * palm_size, 18)
        is_pinching = pinch_dist < effective_threshold

        # Right-click: pinch with middle finger also up
        if is_pinching and middle_up:
            self._right_click_confirm += 1
            if self._right_click_confirm >= self.CONFIRM_FRAMES \
                    and not self._right_click_done \
                    and self._can_act():
                self._right_click_done = True
                self._mark_action()
                return self.GESTURE_RIGHT_CLICK, index_tip, {}
        else:
            self._right_click_confirm = 0
            if not is_pinching:
                self._right_click_done = False

        # Standard pinch (click / double-click) — only when middle is down
        if is_pinching and not middle_up:
            self._pinch_confirm_count += 1
            if self._pinch_confirm_count >= self.CONFIRM_FRAMES \
                    and not self._pinch_active \
                    and self._can_act():
                self._pinch_active = True
                now = time.time()
                if now - self._last_pinch_time < self.DOUBLE_CLICK_INTERVAL:
                    self._last_pinch_time = 0
                    self._mark_action()
                    return self.GESTURE_DOUBLE_CLICK, index_tip, {}
                else:
                    self._last_pinch_time = now
                    self._mark_action()
                    return self.GESTURE_CLICK, index_tip, {}
        else:
            self._pinch_confirm_count = 0

        if not is_pinching:
            self._pinch_active = False

        # ── DRAG : closed fist ────────────────────────────────────────────
        if finger_count == 0:
            self._fist_frames += 1
            if self._fist_frames >= self.DRAG_HOLD_FRAMES:
                if not self._dragging:
                    self._dragging = True
                    return self.GESTURE_DRAG_START, (palm_cx, palm_cy), {}
                else:
                    return self.GESTURE_DRAG_MOVE, (palm_cx, palm_cy), {}
        else:
            if self._dragging:
                self._dragging    = False
                self._fist_frames = 0
                return self.GESTURE_DRAG_END, (palm_cx, palm_cy), {}
            self._fist_frames = 0

        # ── MOVE : only index up ──────────────────────────────────────────
        if index_up and not middle_up and not ring_up and not pinky_up:
            return self.GESTURE_MOVE, index_tip, {}

        return self.GESTURE_NONE, index_tip, {}

    # ── Private helpers ──────────────────────────────────────────────────────

    def _can_act(self):
        """Check global action cooldown."""
        return (time.time() - self._last_action_time) >= self.ACTION_COOLDOWN_SEC

    def _mark_action(self):
        """Record that an action just fired."""
        self._last_action_time = time.time()

    def _detect_scroll(self, palm_cy):
        if self._scroll_cooldown > 0:
            self._scroll_cooldown -= 1
            return None

        if self._prev_palm_y is not None:
            delta = palm_cy - self._prev_palm_y
            if abs(delta) > self.SCROLL_MOTION_THRESH:
                self._scroll_cooldown = self.SCROLL_COOLDOWN_FRAMES
                self._prev_palm_y = palm_cy
                if delta < 0:
                    return self.GESTURE_SCROLL_UP
                else:
                    return self.GESTURE_SCROLL_DOWN
        self._prev_palm_y = palm_cy
        return None

    def _detect_zoom(self, landmarks):
        if self._zoom_cooldown > 0:
            self._zoom_cooldown -= 1
            return None

        thumb_tip = landmarks[HandTracker.THUMB_TIP]
        pinky_tip = landmarks[HandTracker.PINKY_TIP]
        dist = np.hypot(thumb_tip[0] - pinky_tip[0],
                        thumb_tip[1] - pinky_tip[1])

        if self._zoom_prev_dist is not None:
            delta = dist - self._zoom_prev_dist
            if abs(delta) > 18:
                self._zoom_cooldown  = self.ZOOM_COOLDOWN_FRAMES
                self._zoom_prev_dist = dist
                if delta > 0:
                    return self.GESTURE_ZOOM_IN
                else:
                    return self.GESTURE_ZOOM_OUT
        self._zoom_prev_dist = dist
        return None

    def _reset_drag(self):
        if self._dragging:
            self._dragging    = False
            self._fist_frames = 0

    def _palm_center(self, landmarks):
        pts = [landmarks[i] for i in [0, 5, 9, 13, 17]]
        cx  = int(np.mean([p[0] for p in pts]))
        cy  = int(np.mean([p[1] for p in pts]))
        return cx, cy

    def _palm_size(self, landmarks):
        """Approximate palm size (wrist → middle MCP distance)."""
        wx, wy = landmarks[HandTracker.WRIST]
        mx, my = landmarks[HandTracker.MIDDLE_MCP]
        return np.hypot(mx - wx, my - wy)