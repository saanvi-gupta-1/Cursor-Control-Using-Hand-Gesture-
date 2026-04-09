# Cursor Control Using Hand Gestures

Control your computer's cursor, clicks, scrolling, drag-and-drop, zoom, and screenshots using nothing but hand gestures captured through your webcam.

Built with MediaPipe, OpenCV, and PyAutoGUI.

---

## Features

| Gesture | Action | How it works |
|---------|--------|--------------|
| Index finger up | Move cursor | Point with your index finger to move the mouse |
| Pinch (thumb + index) | Left click | Bring thumb and index finger together |
| Two quick pinches | Double click | Pinch twice within 0.4 seconds |
| Pinch + middle up | Right click | Pinch while keeping middle finger raised |
| Two fingers + move up | Scroll up | Raise index and middle finger, move hand up |
| Two fingers + move down | Scroll down | Raise index and middle finger, move hand down |
| Three fingers + move up | Zoom in | Raise index, middle, ring finger and move hand up (Ctrl++) |
| Three fingers + move down | Zoom out | Raise index, middle, ring finger and move hand down (Ctrl--) |
| Closed fist (hold) | Drag and drop | Make a fist and hold for a moment to start dragging |
| Open palm (hold) | Screenshot | Open all 5 fingers and hold briefly |
| Thumb + pinky (shaka) | Pause cursor | Raise thumb and pinky only to freeze the cursor |
| Q key | Quit | Press Q in the webcam window to exit |

---

## How It Works

The application captures video from your webcam and uses MediaPipe's hand tracking model to detect 21 hand landmarks in real time. These landmark positions are fed into a gesture classifier that maps specific hand poses and movements to computer actions via PyAutoGUI.

To keep the cursor smooth and responsive, the system applies a One Euro Filter (an adaptive low-pass filter from the CHI 2012 paper by Casiez et al.). It also blends the index fingertip with the more stable DIP joint to reduce jitter, and uses an adaptive dead-zone that shrinks during fast movement and grows when the hand is nearly still.

Scroll and zoom use velocity-proportional control -- slow hand movement gives fine adjustments, fast movement gives big jumps. Both use a state machine with hysteresis (takes 2 frames to enter the mode, 6 frames to exit) so brief detection drop-outs don't interrupt you mid-gesture.

### Architecture

```
Webcam -> Hand Tracker (MediaPipe) -> Gesture Detector -> Action Dispatcher (PyAutoGUI)
                                           |
                                    One Euro Filter
                                    Adaptive Dead-zone
```

### File Structure

| File | What it does |
|------|-------------|
| main.py | Application loop, cursor smoothing, HUD rendering, action dispatch |
| hand_tracker.py | MediaPipe hand detection, landmark extraction, finger-up logic |
| gesture.py | Gesture classification with frame-confirmation, cooldowns, state machines |
| filters.py | One Euro Filter, exponential moving average, low-pass filter |

---

## Tech Stack

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.9+ | Core language |
| OpenCV | >= 4.8.0 | Webcam capture, frame rendering, HUD overlay |
| MediaPipe | 0.10.9 | Real-time hand landmark detection (21 keypoints) |
| PyAutoGUI | >= 0.9.54 | System-level mouse and keyboard control |
| NumPy | >= 1.24.0 | Numerical computations, coordinate math |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- A working webcam

### Steps

```bash
# Clone the repository
git clone https://github.com/saanvi-gupta-1/Cursor-Control-Using-Hand-Gesture-.git
cd Cursor-Control-Using-Hand-Gesture-

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

A webcam window will open showing:
- Your hand skeleton overlay
- The current gesture name (top-left)
- Cursor position (below gesture name)
- FPS counter (top-right)
- Recent gesture history (right sidebar)
- Quick-reference help bar (bottom)

Press Q inside the window to quit.

---

## Configuration

All tunable parameters are at the top of `main.py`:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| OEF_MIN_CUTOFF | 1.5 | One Euro Filter smoothing at rest (lower = smoother) |
| OEF_BETA | 0.05 | One Euro Filter speed coefficient (higher = less lag) |
| DEAD_ZONE_MIN | 1 | Min dead zone in pixels (during fast movement) |
| DEAD_ZONE_MAX | 5 | Max dead zone in pixels (when nearly still) |
| MARGIN | 50 | Edge buffer zone on webcam frame in pixels |
| SCROLL_BASE | 3 | Base scroll lines per tick |
| SCROLL_GAIN | 8.0 | Velocity to scroll-lines multiplier |

Key thresholds in `gesture.py`:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| PINCH_RATIO | 0.24 | Pinch distance / palm size ratio |
| CONFIRM_FRAMES | 2 | Frames a gesture must persist before firing |
| ACTION_COOLDOWN_SEC | 0.12 | Cooldown between actions in seconds |
| SCREENSHOT_HOLD_FRAMES | 5 | Frames open palm must be held |
| DRAG_HOLD_FRAMES | 4 | Frames fist must be held to start drag |

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| Cursor jitters too much | Lower OEF_MIN_CUTOFF (try 0.8) or increase DEAD_ZONE_MAX |
| Cursor feels laggy | Increase OEF_MIN_CUTOFF (try 2.5) or increase OEF_BETA |
| False clicks happening | Decrease PINCH_RATIO (try 0.20) or increase CONFIRM_FRAMES to 3 |
| Clicks won't register | Increase PINCH_RATIO (try 0.30) or sit closer to camera |
| Webcam not found | Change WEBCAM_ID at the top of main.py (try 1 or 2) |
| Scroll/zoom too sensitive | Increase the velocity threshold in gesture.py |
| Drag activates too easily | Increase DRAG_HOLD_FRAMES (try 6 or 8) |

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
