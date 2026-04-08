# 🖐️ Cursor Control Using Hand Gestures

> Control your entire computer — cursor, clicks, scrolling, drag & drop, zoom, and screenshots — using nothing but hand gestures captured through your webcam.

Built with **MediaPipe**, **OpenCV**, and **PyAutoGUI**.

---

## ✨ Features

| Gesture | Action | Description |
|---------|--------|-------------|
| ☝️ Index finger up | **Move cursor** | Point with your index finger to move the mouse |
| 🤏 Pinch (thumb + index) | **Left click** | Bring thumb and index finger together |
| 🤏🤏 Two quick pinches | **Double click** | Pinch twice within 0.35 seconds |
| 🤏 + ✌️ Pinch + middle up | **Right click** | Pinch while keeping middle finger raised |
| ✌️ Two fingers + move up | **Scroll up** | Raise index + middle finger and move hand up |
| ✌️ Two fingers + move down | **Scroll down** | Raise index + middle finger and move hand down |
| 🤙 Thumb + pinky spread | **Zoom in** | Spread thumb and pinky apart (Ctrl++) |
| 🤙 Thumb + pinky close | **Zoom out** | Bring thumb and pinky together (Ctrl−−) |
| ✊ Closed fist (hold) | **Drag & drop** | Make a fist and hold for ~0.3s to start dragging |
| 🖐️ Open palm (hold) | **Screenshot** | Open all 5 fingers and hold for ~0.2s |
| 🖖 3 fingers up | **Pause / freeze** | Raise index + middle + ring to freeze cursor |
| `Q` key | **Quit** | Press Q in the webcam window to exit |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                   main.py                       │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐ │
│  │  Webcam   │→ │   Hand     │→ │  Gesture   │ │
│  │  Capture  │  │  Tracker   │  │  Detector  │ │
│  └───────────┘  └────────────┘  └────────────┘ │
│        ↓              ↓              ↓          │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐ │
│  │   Holt    │← │  Finger    │← │  Frame     │ │
│  │ Smoother  │  │  Analysis  │  │ Confirmed  │ │
│  └───────────┘  └────────────┘  └────────────┘ │
│        ↓                                        │
│  ┌───────────┐  ┌────────────┐                  │
│  │ Dead-zone │→ │ PyAutoGUI  │→  OS Actions    │
│  │  Filter   │  │  Dispatch  │                  │
│  └───────────┘  └────────────┘                  │
└─────────────────────────────────────────────────┘
```

### Key Modules

| File | Responsibility |
|------|----------------|
| `main.py` | Application loop, cursor smoothing (Holt filter), HUD rendering, action dispatch |
| `hand_tracker.py` | MediaPipe hand detection, landmark extraction, finger-up logic |
| `gesture.py` | Gesture classification with frame-confirmation, cooldowns, and adaptive thresholds |

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- A working webcam

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Cursor-Control-Using-Hand-Gesture-.git
cd Cursor-Control-Using-Hand-Gesture-

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py
```

A webcam window titled **"Advanced Gesture Control"** will open showing:
- Your hand skeleton overlay
- Current gesture name (top-left)
- Cursor position (below gesture name)
- FPS counter (top-right)
- Recent gesture history (right sidebar)
- Quick-reference help bar (bottom)

Press **Q** inside the window to quit.

---

## ⚙️ Configuration

All tunable parameters are at the top of `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SMOOTH_ALPHA` | `0.35` | Holt smoother level weight (lower = smoother) |
| `SMOOTH_BETA` | `0.15` | Holt smoother trend weight (lower = smoother) |
| `DEAD_ZONE` | `4` | Ignore cursor moves smaller than this (pixels) |
| `MARGIN` | `100` | Edge buffer zone on webcam frame (pixels) |
| `SCROLL_AMOUNT` | `3` | Lines per scroll tick |

Key thresholds in `gesture.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PINCH_RATIO` | `0.22` | Pinch distance / palm size ratio |
| `CONFIRM_FRAMES` | `2` | Frames a gesture must persist before firing |
| `ACTION_COOLDOWN_SEC` | `0.18` | Global cooldown between actions (seconds) |
| `SCREENSHOT_HOLD_FRAMES` | `5` | Frames open palm must be held |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Cursor jitters too much | Decrease `SMOOTH_ALPHA` (e.g. `0.25`) or increase `DEAD_ZONE` |
| Cursor feels laggy | Increase `SMOOTH_ALPHA` (e.g. `0.45`) |
| False clicks | Decrease `PINCH_RATIO` (e.g. `0.18`) or increase `CONFIRM_FRAMES` to `3` |
| Clicks won't register | Increase `PINCH_RATIO` (e.g. `0.28`) or sit closer to camera |
| Webcam not found | Check `WEBCAM_ID` (try `1` or `2` for external webcams) |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.8.0 | Webcam capture & frame rendering |
| `mediapipe` | 0.10.9 | Hand landmark detection |
| `pyautogui` | ≥ 0.9.54 | System mouse/keyboard control |
| `numpy` | ≥ 1.24.0 | Numerical computations |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
