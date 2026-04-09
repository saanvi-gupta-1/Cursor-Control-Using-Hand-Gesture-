"""
Gesture Detection Engine v2
============================
Production-grade gesture recognition with:

  • **State machine with hysteresis** — enter/exit thresholds are different,
    so gestures don't flicker in and out when detection is borderline.
  • **Velocity-based scroll** — returns scroll *velocity* so main.py can
    apply proportional scroll amounts (slow hand → slow scroll, fast → fast).
  • **Velocity-based zoom** — same proportional approach for zoom.
  • **Relaxed finger conditions** — thumb is allowed in most poses; only
    the distinctive fingers matter for each gesture.
  • **Palm-relative thresholds** — all distance/motion thresholds scale
    with palm size, so gestures work identically at any hand-to-camera
    distance.
"""

import time
import numpy as np
from collections import deque
from hand_tracker import HandTracker
from filters import EMA


class GestureDetector:
    """
    Advanced gesture recognition engine.

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

    # ── Tuned thresholds ────────────────────────────────────────────────────
    PINCH_RATIO             = 0.24   # pinch dist / palm size

    DOUBLE_CLICK_INTERVAL   = 0.40   # seconds between pinches
    CONFIRM_FRAMES          = 2      # consecutive detections before click fires
    ACTION_COOLDOWN_SEC     = 0.12   # global cooldown between discrete actions

    DRAG_HOLD_FRAMES        = 8      # frames fist must be held before drag

    SCREENSHOT_HOLD_FRAMES  = 5      # palm must be held this many frames

    # ── Scroll parameters ───────────────────────────────────────────────────
    SCROLL_VELOCITY_THRESH  = 0.4    # palm-relative velocity threshold
    SCROLL_ENTER_FRAMES     = 2      # frames to ENTER scroll mode
    SCROLL_EXIT_FRAMES      = 6      # frames to EXIT scroll mode (hysteresis!)

    # ── Zoom parameters ─────────────────────────────────────────────────────
    ZOOM_DELTA_RATIO        = 0.04   # palm-relative change to trigger zoom
    ZOOM_ENTER_FRAMES       = 2      # frames to ENTER zoom mode
    ZOOM_EXIT_FRAMES        = 6      # frames to EXIT zoom mode (hysteresis!)

    def __init__(self):
        # ── Pinch / click state ─────────────────────────────────────────────
        self._last_pinch_time      = 0
        self._pinch_active         = False
        self._pinch_confirm_count  = 0
        self._right_click_confirm  = 0
        self._right_click_done     = False

        # ── Drag state ──────────────────────────────────────────────────────
        self._fist_frames          = 0
        self._dragging             = False

        # ── Scroll state (state machine + velocity) ─────────────────────────
        self._scroll_active        = False
        self._scroll_enter_count   = 0    # consecutive "scroll pose" frames
        self._scroll_exit_count    = 0    # consecutive "non-scroll pose" frames
        self._prev_palm_y          = None
        self._prev_scroll_time     = None
        self._scroll_velocity_ema  = EMA(alpha=0.35)   # smoothed velocity

        # ── Zoom state (state machine + velocity) ───────────────────────────
        self._zoom_active          = False
        self._zoom_enter_count     = 0
        self._zoom_exit_count      = 0
        self._zoom_prev_dist       = None
        self._prev_zoom_time       = None
        self._zoom_velocity_ema    = EMA(alpha=0.35)

        # ── Screenshot state ────────────────────────────────────────────────
        self._screenshot_hold      = 0
        self._screenshot_done      = False

        # ── Global cooldown ─────────────────────────────────────────────────
        self._last_action_time     = 0

        # ── Gesture smoothing buffer (majority vote) ────────────────────────
        self._gesture_buffer       = deque(maxlen=3)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def detect(self, landmarks, fingers):
        """
        Main detection entry point.

        Returns
        -------
        (gesture_name, control_point, extra_data)
            control_point : (x, y) pixel position to anchor cursor
            extra_data    : dict – may contain 'velocity' for scroll/zoom
        """
        now = time.time()

        if landmarks is None:
            self._reset_on_hand_lost()
            return self.GESTURE_NONE, None, {}

        index_tip  = landmarks[HandTracker.INDEX_TIP]
        thumb_tip  = landmarks[HandTracker.THUMB_TIP]
        middle_tip = landmarks[HandTracker.MIDDLE_TIP]
        palm_cx, palm_cy = self._palm_center(landmarks)
        palm_size  = self._palm_size(landmarks)

        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers
        finger_count = sum(fingers)

        # ── 1. SCREENSHOT : all 5 up, held for N frames ──────────────────
        if finger_count == 5:
            self._screenshot_hold += 1
            if (self._screenshot_hold >= self.SCREENSHOT_HOLD_FRAMES
                    and not self._screenshot_done):
                self._screenshot_done = True
                self._mark_action()
                return self.GESTURE_SCREENSHOT, index_tip, {}
            return self.GESTURE_NONE, index_tip, {}
        else:
            self._screenshot_hold = 0
            self._screenshot_done = False

        # ── 2. PAUSE : index + middle + ring up, not pinky, not thumb ─────
        if index_up and middle_up and ring_up and not pinky_up and not thumb_up:
            return self.GESTURE_PAUSE, index_tip, {}

        # ── 3. ZOOM (state machine with hysteresis) ───────────────────────
        # Zoom pose: thumb + pinky up.  Other fingers can be loose.
        zoom_pose = thumb_up and pinky_up and finger_count < 5
        zoom_result = self._update_zoom_state(zoom_pose, landmarks, palm_size, now)
        if zoom_result is not None:
            return zoom_result[0], index_tip, zoom_result[1]

        # ── 4. SCROLL (state machine with hysteresis) ─────────────────────
        # Scroll pose: index + middle up, ring + pinky down.  Thumb is free.
        scroll_pose = index_up and middle_up and not ring_up and not pinky_up
        scroll_result = self._update_scroll_state(scroll_pose, palm_cy, palm_size, now)
        if scroll_result is not None:
            return scroll_result[0], index_tip, scroll_result[1]

        # Always update palm Y for scroll tracking (even outside scroll mode)
        self._prev_palm_y    = palm_cy
        self._prev_scroll_time = now

        # ── 5. PINCH detection (click / double-click / right-click) ───────
        pinch_dist = np.hypot(thumb_tip[0] - index_tip[0],
                              thumb_tip[1] - index_tip[1])
        effective_threshold = max(self.PINCH_RATIO * palm_size, 18)
        is_pinching = pinch_dist < effective_threshold

        # Right-click: pinch + middle finger up
        if is_pinching and middle_up:
            self._right_click_confirm += 1
            if (self._right_click_confirm >= self.CONFIRM_FRAMES
                    and not self._right_click_done
                    and self._can_act()):
                self._right_click_done = True
                self._mark_action()
                return self.GESTURE_RIGHT_CLICK, index_tip, {}
        else:
            self._right_click_confirm = 0
            if not is_pinching:
                self._right_click_done = False

        # Standard pinch (click / double-click) — middle must be down
        if is_pinching and not middle_up:
            self._pinch_confirm_count += 1
            if (self._pinch_confirm_count >= self.CONFIRM_FRAMES
                    and not self._pinch_active
                    and self._can_act()):
                self._pinch_active = True
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

        # ── 6. DRAG : closed fist ─────────────────────────────────────────
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

        # ── 7. MOVE : index up (thumb allowed) ───────────────────────────
        if index_up and not middle_up and not ring_up and not pinky_up:
            return self.GESTURE_MOVE, index_tip, {}

        return self.GESTURE_NONE, index_tip, {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SCROLL STATE MACHINE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _update_scroll_state(self, pose_active, palm_cy, palm_size, now):
        """
        State machine for scroll with enter/exit hysteresis.

        Returns None if scroll is inactive, or (gesture, extra_dict).
        """
        if pose_active:
            self._scroll_exit_count = 0
            if not self._scroll_active:
                self._scroll_enter_count += 1
                if self._scroll_enter_count >= self.SCROLL_ENTER_FRAMES:
                    self._scroll_active = True
                    self._prev_palm_y      = palm_cy
                    self._prev_scroll_time = now
                    self._scroll_velocity_ema.reset()
            # Already active → detect scroll direction + velocity
            if self._scroll_active:
                return self._compute_scroll(palm_cy, palm_size, now)
            # Entering but not yet confirmed
            return (self.GESTURE_NONE, {})
        else:
            self._scroll_enter_count = 0
            if self._scroll_active:
                self._scroll_exit_count += 1
                if self._scroll_exit_count >= self.SCROLL_EXIT_FRAMES:
                    self._scroll_active = False
                    self._prev_palm_y   = None
                else:
                    # Still in scroll mode (hysteresis hold)
                    return (self.GESTURE_NONE, {})
            return None

    def _compute_scroll(self, palm_cy, palm_size, now):
        """Compute scroll direction and velocity from palm movement."""
        if self._prev_palm_y is None or self._prev_scroll_time is None:
            self._prev_palm_y      = palm_cy
            self._prev_scroll_time = now
            return (self.GESTURE_NONE, {})

        dt = max(now - self._prev_scroll_time, 1e-4)
        raw_velocity = (palm_cy - self._prev_palm_y) / dt   # px/sec

        # Normalize by palm size → scale-invariant velocity
        norm_velocity = raw_velocity / max(palm_size, 1.0)

        # Smooth with EMA to remove jitter
        smooth_vel = self._scroll_velocity_ema(norm_velocity)

        self._prev_palm_y      = palm_cy
        self._prev_scroll_time = now

        if abs(smooth_vel) > self.SCROLL_VELOCITY_THRESH:
            magnitude = abs(smooth_vel)
            if smooth_vel < 0:
                return (self.GESTURE_SCROLL_UP, {"velocity": magnitude})
            else:
                return (self.GESTURE_SCROLL_DOWN, {"velocity": magnitude})

        return (self.GESTURE_NONE, {})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  ZOOM STATE MACHINE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _update_zoom_state(self, pose_active, landmarks, palm_size, now):
        """
        State machine for zoom with enter/exit hysteresis.

        Returns None if zoom is inactive, or (gesture, extra_dict).
        """
        if pose_active:
            self._zoom_exit_count = 0
            if not self._zoom_active:
                self._zoom_enter_count += 1
                if self._zoom_enter_count >= self.ZOOM_ENTER_FRAMES:
                    self._zoom_active   = True
                    self._zoom_prev_dist = self._thumb_pinky_dist(landmarks)
                    self._prev_zoom_time = now
                    self._zoom_velocity_ema.reset()
            # Already active → detect zoom direction + velocity
            if self._zoom_active:
                return self._compute_zoom(landmarks, palm_size, now)
            return (self.GESTURE_NONE, {})
        else:
            self._zoom_enter_count = 0
            if self._zoom_active:
                self._zoom_exit_count += 1
                if self._zoom_exit_count >= self.ZOOM_EXIT_FRAMES:
                    self._zoom_active    = False
                    self._zoom_prev_dist = None
                else:
                    # Still in zoom mode (hysteresis hold)
                    # Keep updating distance so it doesn't go stale
                    if landmarks:
                        self._zoom_prev_dist = self._thumb_pinky_dist(landmarks)
                    return (self.GESTURE_NONE, {})
            return None

    def _compute_zoom(self, landmarks, palm_size, now):
        """Compute zoom direction and velocity from thumb-pinky spread."""
        dist = self._thumb_pinky_dist(landmarks)

        if self._zoom_prev_dist is None or self._prev_zoom_time is None:
            self._zoom_prev_dist = dist
            self._prev_zoom_time = now
            return (self.GESTURE_NONE, {})

        dt = max(now - self._prev_zoom_time, 1e-4)

        # Normalized rate of spread change (per second, per palm-size)
        raw_delta = (dist - self._zoom_prev_dist) / dt / max(palm_size, 1.0)

        smooth_delta = self._zoom_velocity_ema(raw_delta)

        self._zoom_prev_dist = dist
        self._prev_zoom_time = now

        threshold = self.ZOOM_DELTA_RATIO
        if abs(smooth_delta) > threshold:
            magnitude = abs(smooth_delta)
            if smooth_delta > 0:
                return (self.GESTURE_ZOOM_IN, {"velocity": magnitude})
            else:
                return (self.GESTURE_ZOOM_OUT, {"velocity": magnitude})

        return (self.GESTURE_NONE, {})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _can_act(self):
        """Check global action cooldown."""
        return (time.time() - self._last_action_time) >= self.ACTION_COOLDOWN_SEC

    def _mark_action(self):
        """Record that an action just fired."""
        self._last_action_time = time.time()

    def _thumb_pinky_dist(self, landmarks):
        thumb = landmarks[HandTracker.THUMB_TIP]
        pinky = landmarks[HandTracker.PINKY_TIP]
        return np.hypot(thumb[0] - pinky[0], thumb[1] - pinky[1])

    def _reset_on_hand_lost(self):
        """Reset all state when hand disappears."""
        if self._dragging:
            self._dragging    = False
            self._fist_frames = 0
        self._prev_palm_y       = None
        self._prev_scroll_time  = None
        self._zoom_prev_dist    = None
        self._prev_zoom_time    = None
        self._scroll_active     = False
        self._scroll_enter_count = 0
        self._scroll_exit_count  = 0
        self._zoom_active       = False
        self._zoom_enter_count  = 0
        self._zoom_exit_count   = 0

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