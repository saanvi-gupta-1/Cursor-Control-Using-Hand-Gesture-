"""
Gesture Detection Engine
========================
Gesture recognition with state-machine hysteresis for scroll and zoom,
velocity-based proportional control, and palm-relative adaptive thresholds.

Gesture Mappings
----------------
  1 finger  (index)                -> Move cursor
  Pinch     (thumb + index close)  -> Left click
  Two quick pinches                -> Double click
  Pinch + middle up                -> Right click
  2 fingers (index+middle) + move  -> Scroll up/down
  3 fingers (i+m+ring)   + move    -> Zoom in (up) / out (down)
  Fist      (hold ~0.15s)          -> Drag and drop
  5 fingers (palm, hold)           -> Screenshot
  Thumb + pinky (shaka)            -> Pause / freeze cursor
"""

import time
import numpy as np
from collections import deque
from hand_tracker import HandTracker
from filters import EMA


class GestureDetector:
    """
    Gesture recognition engine with intuitive hand-pose mappings.

    Uses frame-counting for gesture confirmation, cooldowns to prevent
    accidental repeat-fires, and state machines with hysteresis for
    continuous gestures (scroll, zoom).
    """

    # Gesture name constants
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

    # Tuned thresholds
    PINCH_RATIO             = 0.24
    DOUBLE_CLICK_INTERVAL   = 0.40
    CONFIRM_FRAMES          = 2
    ACTION_COOLDOWN_SEC     = 0.12

    DRAG_HOLD_FRAMES        = 4
    SCREENSHOT_HOLD_FRAMES  = 5

    # Scroll parameters
    SCROLL_VELOCITY_THRESH  = 0.4
    SCROLL_ENTER_FRAMES     = 2
    SCROLL_EXIT_FRAMES      = 6

    # Zoom parameters (uses vertical movement, same approach as scroll)
    ZOOM_VELOCITY_THRESH    = 0.35
    ZOOM_ENTER_FRAMES       = 2
    ZOOM_EXIT_FRAMES        = 6

    def __init__(self):
        # Pinch / click state
        self._last_pinch_time      = 0
        self._pinch_active         = False
        self._pinch_confirm_count  = 0
        self._right_click_confirm  = 0
        self._right_click_done     = False

        # Drag state
        self._fist_frames          = 0
        self._dragging             = False

        # Scroll state machine
        self._scroll_active        = False
        self._scroll_enter_count   = 0
        self._scroll_exit_count    = 0
        self._scroll_prev_palm_y   = None
        self._scroll_prev_time     = None
        self._scroll_velocity_ema  = EMA(alpha=0.35)

        # Zoom state machine (vertical movement, like scroll)
        self._zoom_active          = False
        self._zoom_enter_count     = 0
        self._zoom_exit_count      = 0
        self._zoom_prev_palm_y     = None
        self._zoom_prev_time       = None
        self._zoom_velocity_ema    = EMA(alpha=0.35)

        # Screenshot
        self._screenshot_hold      = 0
        self._screenshot_done      = False

        # Global cooldown
        self._last_action_time     = 0

    # -------------------------------------------------------------------------
    #  PUBLIC API
    # -------------------------------------------------------------------------

    def detect(self, landmarks, fingers):
        """
        Main detection entry point.

        Returns (gesture_name, control_point, extra_data)
            control_point : (x, y) pixel position to anchor cursor
            extra_data    : dict, may contain 'velocity' for scroll/zoom
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

        # Priority order:
        #   1. Screenshot  (5 fingers, held)
        #   2. Pause       (thumb + pinky only, shaka sign)
        #   3. Zoom        (3 fingers: index+middle+ring + move up/down)
        #   4. Scroll      (2 fingers: index+middle + move up/down)
        #   5. Pinch/Click (thumb+index close)
        #   6. Drag        (fist, 0 fingers)
        #   7. Move        (index only)

        # 1. SCREENSHOT: all 5 fingers up, held for N frames
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

        # 2. PAUSE: thumb + pinky up, others down (shaka sign)
        if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
            return self.GESTURE_PAUSE, index_tip, {}

        # 3. ZOOM: index + middle + ring up, NOT pinky (thumb free)
        #    Move hand UP = zoom in,  DOWN = zoom out
        zoom_pose = index_up and middle_up and ring_up and not pinky_up
        zoom_result = self._update_zoom_state(zoom_pose, palm_cy, palm_size, now)
        if zoom_result is not None:
            return zoom_result[0], index_tip, zoom_result[1]

        # 4. SCROLL: index + middle up, NOT ring, NOT pinky
        #    (thumb is free, many people can't tuck it)
        scroll_pose = index_up and middle_up and not ring_up and not pinky_up
        scroll_result = self._update_scroll_state(scroll_pose, palm_cy, palm_size, now)
        if scroll_result is not None:
            return scroll_result[0], index_tip, scroll_result[1]

        # Keep palm Y fresh for next frame when neither scroll nor zoom is active
        self._scroll_prev_palm_y = palm_cy
        self._scroll_prev_time   = now
        self._zoom_prev_palm_y   = palm_cy
        self._zoom_prev_time     = now

        # 5. PINCH detection (click / double-click / right-click)
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

        # Standard pinch (click / double-click)
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

        # 6. DRAG: closed fist (0 fingers)
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

        # 7. MOVE: index up (thumb allowed)
        if index_up and not middle_up and not ring_up and not pinky_up:
            return self.GESTURE_MOVE, index_tip, {}

        return self.GESTURE_NONE, index_tip, {}

    # -------------------------------------------------------------------------
    #  SCROLL STATE MACHINE
    # -------------------------------------------------------------------------

    def _update_scroll_state(self, pose_active, palm_cy, palm_size, now):
        """State machine for scroll with enter/exit hysteresis."""
        if pose_active:
            self._scroll_exit_count = 0
            if not self._scroll_active:
                self._scroll_enter_count += 1
                if self._scroll_enter_count >= self.SCROLL_ENTER_FRAMES:
                    self._scroll_active      = True
                    self._scroll_prev_palm_y = palm_cy
                    self._scroll_prev_time   = now
                    self._scroll_velocity_ema.reset()
            if self._scroll_active:
                return self._compute_scroll(palm_cy, palm_size, now)
            return (self.GESTURE_NONE, {})
        else:
            self._scroll_enter_count = 0
            if self._scroll_active:
                self._scroll_exit_count += 1
                if self._scroll_exit_count >= self.SCROLL_EXIT_FRAMES:
                    self._scroll_active = False
                    self._scroll_prev_palm_y = None
                else:
                    return (self.GESTURE_NONE, {})
            return None

    def _compute_scroll(self, palm_cy, palm_size, now):
        """Compute scroll direction and velocity from palm movement."""
        if self._scroll_prev_palm_y is None or self._scroll_prev_time is None:
            self._scroll_prev_palm_y = palm_cy
            self._scroll_prev_time   = now
            return (self.GESTURE_NONE, {})

        dt = max(now - self._scroll_prev_time, 1e-4)
        raw_velocity = (palm_cy - self._scroll_prev_palm_y) / dt
        norm_velocity = raw_velocity / max(palm_size, 1.0)
        smooth_vel = self._scroll_velocity_ema(norm_velocity)

        self._scroll_prev_palm_y = palm_cy
        self._scroll_prev_time   = now

        if abs(smooth_vel) > self.SCROLL_VELOCITY_THRESH:
            magnitude = abs(smooth_vel)
            if smooth_vel < 0:
                return (self.GESTURE_SCROLL_UP, {"velocity": magnitude})
            else:
                return (self.GESTURE_SCROLL_DOWN, {"velocity": magnitude})

        return (self.GESTURE_NONE, {})

    # -------------------------------------------------------------------------
    #  ZOOM STATE MACHINE (vertical movement, same approach as scroll)
    # -------------------------------------------------------------------------

    def _update_zoom_state(self, pose_active, palm_cy, palm_size, now):
        """State machine for zoom with enter/exit hysteresis."""
        if pose_active:
            self._zoom_exit_count = 0
            if not self._zoom_active:
                self._zoom_enter_count += 1
                if self._zoom_enter_count >= self.ZOOM_ENTER_FRAMES:
                    self._zoom_active      = True
                    self._zoom_prev_palm_y = palm_cy
                    self._zoom_prev_time   = now
                    self._zoom_velocity_ema.reset()
            if self._zoom_active:
                return self._compute_zoom(palm_cy, palm_size, now)
            return (self.GESTURE_NONE, {})
        else:
            self._zoom_enter_count = 0
            if self._zoom_active:
                self._zoom_exit_count += 1
                if self._zoom_exit_count >= self.ZOOM_EXIT_FRAMES:
                    self._zoom_active      = False
                    self._zoom_prev_palm_y = None
                else:
                    # Keep updating so we don't build stale deltas
                    self._zoom_prev_palm_y = palm_cy
                    self._zoom_prev_time   = now
                    return (self.GESTURE_NONE, {})
            return None

    def _compute_zoom(self, palm_cy, palm_size, now):
        """
        Compute zoom from vertical hand movement.
        Move hand UP = zoom in,  move hand DOWN = zoom out.
        """
        if self._zoom_prev_palm_y is None or self._zoom_prev_time is None:
            self._zoom_prev_palm_y = palm_cy
            self._zoom_prev_time   = now
            return (self.GESTURE_NONE, {})

        dt = max(now - self._zoom_prev_time, 1e-4)
        raw_velocity = (palm_cy - self._zoom_prev_palm_y) / dt
        norm_velocity = raw_velocity / max(palm_size, 1.0)
        smooth_vel = self._zoom_velocity_ema(norm_velocity)

        self._zoom_prev_palm_y = palm_cy
        self._zoom_prev_time   = now

        if abs(smooth_vel) > self.ZOOM_VELOCITY_THRESH:
            magnitude = abs(smooth_vel)
            if smooth_vel < 0:
                # Hand moving UP in frame -> zoom IN
                return (self.GESTURE_ZOOM_IN, {"velocity": magnitude})
            else:
                # Hand moving DOWN in frame -> zoom OUT
                return (self.GESTURE_ZOOM_OUT, {"velocity": magnitude})

        return (self.GESTURE_NONE, {})

    # -------------------------------------------------------------------------
    #  HELPERS
    # -------------------------------------------------------------------------

    def _can_act(self):
        return (time.time() - self._last_action_time) >= self.ACTION_COOLDOWN_SEC

    def _mark_action(self):
        self._last_action_time = time.time()

    def _reset_on_hand_lost(self):
        """Reset all state when hand disappears."""
        if self._dragging:
            self._dragging    = False
            self._fist_frames = 0
        self._scroll_prev_palm_y = None
        self._scroll_prev_time   = None
        self._zoom_prev_palm_y   = None
        self._zoom_prev_time     = None
        self._scroll_active      = False
        self._scroll_enter_count = 0
        self._scroll_exit_count  = 0
        self._zoom_active        = False
        self._zoom_enter_count   = 0
        self._zoom_exit_count    = 0

    def _palm_center(self, landmarks):
        pts = [landmarks[i] for i in [0, 5, 9, 13, 17]]
        cx  = int(np.mean([p[0] for p in pts]))
        cy  = int(np.mean([p[1] for p in pts]))
        return cx, cy

    def _palm_size(self, landmarks):
        wx, wy = landmarks[HandTracker.WRIST]
        mx, my = landmarks[HandTracker.MIDDLE_MCP]
        return np.hypot(mx - wx, my - wy)