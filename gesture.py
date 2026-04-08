import time
from hand_tracker import HandTracker


class GestureDetector:
    """
    Detects high-level gestures from hand landmarks.
    Gestures:
      - MOVE      : only index finger up
      - CLICK     : pinch (thumb + index close together)
      - DOUBLE_CLICK: two pinches within DOUBLE_CLICK_INTERVAL seconds
      - PAUSE     : index + middle fingers up (freeze cursor)
    """

    PINCH_THRESHOLD = 40        # pixels — distance to count as pinch
    DOUBLE_CLICK_INTERVAL = 0.4  # seconds between pinches

    GESTURE_NONE = "none"
    GESTURE_MOVE = "move"
    GESTURE_CLICK = "click"
    GESTURE_DOUBLE_CLICK = "double_click"
    GESTURE_PAUSE = "pause"

    def __init__(self):
        self._last_pinch_time = 0
        self._pinch_active = False  # prevents re-triggering while held

    def detect(self, landmarks, fingers):
        """
        Returns (gesture_name, index_tip_position).
        index_tip_position is (x, y) pixels from the webcam frame.
        """
        if landmarks is None:
            return self.GESTURE_NONE, None

        index_tip = landmarks[HandTracker.INDEX_TIP]
        thumb_tip = landmarks[HandTracker.THUMB_TIP]

        # Finger states
        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers

        # --- PAUSE: index + middle both up ---
        if index_up and middle_up and not ring_up and not pinky_up:
            return self.GESTURE_PAUSE, index_tip

        # --- PINCH detection ---
        import numpy as np
        pinch_dist = np.hypot(
            thumb_tip[0] - index_tip[0],
            thumb_tip[1] - index_tip[1]
        )
        is_pinching = pinch_dist < self.PINCH_THRESHOLD

        if is_pinching and not self._pinch_active:
            self._pinch_active = True
            now = time.time()
            if now - self._last_pinch_time < self.DOUBLE_CLICK_INTERVAL:
                self._last_pinch_time = 0  # reset so triple tap doesn't misfire
                return self.GESTURE_DOUBLE_CLICK, index_tip
            else:
                self._last_pinch_time = now
                return self.GESTURE_CLICK, index_tip

        if not is_pinching:
            self._pinch_active = False

        # --- MOVE: only index finger up ---
        if index_up and not middle_up:
            return self.GESTURE_MOVE, index_tip

        return self.GESTURE_NONE, index_tip
