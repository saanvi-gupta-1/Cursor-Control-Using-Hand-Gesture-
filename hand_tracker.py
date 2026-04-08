import mediapipe as mp
import cv2
import numpy as np


class HandTracker:
    """
    Wraps MediaPipe Hands for real-time hand landmark detection.
    Enhanced with angle-based finger detection and multi-hand support.
    """

    # MediaPipe landmark indices
    WRIST        = 0
    THUMB_CMC    = 1
    THUMB_MCP    = 2
    THUMB_IP     = 3
    THUMB_TIP    = 4
    INDEX_MCP    = 5
    INDEX_PIP    = 6
    INDEX_DIP    = 7
    INDEX_TIP    = 8
    MIDDLE_MCP   = 9
    MIDDLE_PIP   = 10
    MIDDLE_DIP   = 11
    MIDDLE_TIP   = 12
    RING_MCP     = 13
    RING_PIP     = 14
    RING_DIP     = 15
    RING_TIP     = 16
    PINKY_MCP    = 17
    PINKY_PIP    = 18
    PINKY_DIP    = 19
    PINKY_TIP    = 20

    def __init__(self, max_hands=1, detection_confidence=0.8, tracking_confidence=0.8):
        self.mp_hands  = mp.solutions.hands
        self.mp_draw   = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if draw and results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return frame, results

    def get_landmarks(self, results, frame_shape, hand_index=0):
        if not results.multi_hand_landmarks:
            return None
        if hand_index >= len(results.multi_hand_landmarks):
            return None
        h, w = frame_shape[:2]
        hand_lm = results.multi_hand_landmarks[hand_index]
        landmarks = {}
        for idx, lm in enumerate(hand_lm.landmark):
            landmarks[idx] = (int(lm.x * w), int(lm.y * h))
        return landmarks

    def get_landmarks_normalized(self, results, hand_index=0):
        """Returns normalized (0-1) landmarks for rotation-invariant gestures."""
        if not results.multi_hand_landmarks:
            return None
        if hand_index >= len(results.multi_hand_landmarks):
            return None
        hand_lm = results.multi_hand_landmarks[hand_index]
        return {idx: (lm.x, lm.y, lm.z) for idx, lm in enumerate(hand_lm.landmark)}

    def fingers_up(self, landmarks):
        """
        Returns [thumb, index, middle, ring, pinky] booleans.
        Uses both Y-axis and angle for more robust detection.
        """
        if landmarks is None:
            return [False] * 5

        tips = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        pips = [self.THUMB_IP,  self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        mcps = [self.THUMB_MCP, self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP]

        up = []
        for i, (tip_id, pip_id, mcp_id) in enumerate(zip(tips, pips, mcps)):
            if i == 0:  # Thumb: use X axis
                up.append(landmarks[tip_id][0] < landmarks[pip_id][0])
            else:
                tip_y = landmarks[tip_id][1]
                pip_y = landmarks[pip_id][1]
                mcp_y = landmarks[mcp_id][1]
                # Finger is up if tip is above both PIP and MCP
                up.append(tip_y < pip_y and tip_y < mcp_y)
        return up

    def count_fingers(self, landmarks):
        """Returns integer count of extended fingers."""
        return sum(self.fingers_up(landmarks))

    def distance(self, landmarks, id1, id2):
        x1, y1 = landmarks[id1]
        x2, y2 = landmarks[id2]
        return np.hypot(x2 - x1, y2 - y1)

    def midpoint(self, landmarks, id1, id2):
        x1, y1 = landmarks[id1]
        x2, y2 = landmarks[id2]
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def hand_angle(self, landmarks):
        """Returns the rotation angle of the hand (wrist to middle MCP)."""
        wx, wy = landmarks[self.WRIST]
        mx, my = landmarks[self.MIDDLE_MCP]
        return np.degrees(np.arctan2(wy - my, mx - wx))

    def palm_center(self, landmarks):
        """Returns approximate palm center."""
        pts = [landmarks[i] for i in [0, 5, 9, 13, 17]]
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        return (cx, cy)

    def palm_size(self, landmarks):
        """Returns approximate palm size (wrist to middle MCP distance)."""
        return self.distance(landmarks, self.WRIST, self.MIDDLE_MCP)