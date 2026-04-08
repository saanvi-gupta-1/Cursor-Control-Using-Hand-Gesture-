import mediapipe as mp
import cv2
import numpy as np


class HandTracker:
    """
    Wraps MediaPipe Hands for real-time hand landmark detection.
    """

    # MediaPipe landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_MCP = 9
    RING_TIP = 16
    PINKY_TIP = 20

    def __init__(self, max_hands=1, detection_confidence=0.8, tracking_confidence=0.8):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def find_hands(self, frame, draw=True):
        """
        Process frame, optionally draw landmarks.
        Returns the annotated frame and raw results.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if draw and results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

        return frame, results

    def get_landmarks(self, results, frame_shape, hand_index=0):
        """
        Returns a dict of {landmark_id: (x_px, y_px)} for the given hand.
        Returns None if no hand detected.
        """
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

    def fingers_up(self, landmarks):
        """
        Returns a list of booleans [thumb, index, middle, ring, pinky]
        indicating which fingers are extended.
        """
        if landmarks is None:
            return [False] * 5

        tips = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        mcps = [2, self.INDEX_MCP, self.MIDDLE_MCP, 13, 17]

        up = []
        for tip_id, mcp_id in zip(tips, mcps):
            tip_y = landmarks[tip_id][1]
            mcp_y = landmarks[mcp_id][1]
            up.append(tip_y < mcp_y)  # finger is up when tip is above MCP

        return up

    def distance(self, landmarks, id1, id2):
        """Euclidean distance in pixels between two landmarks."""
        x1, y1 = landmarks[id1]
        x2, y2 = landmarks[id2]
        return np.hypot(x2 - x1, y2 - y1)
