"""
Signal Processing Filters
==========================
Adaptive signal filters for real-time hand tracking.

Implements the One Euro Filter, an algorithm for reducing jitter in
interactive cursor/pointing systems while preserving low-latency
responsiveness.

Reference: Casiez, Roussel, Vogel. "1 Euro Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems." CHI 2012.

Core idea: adapt the smoothing cutoff based on input speed.
  Slow movement -> low cutoff -> heavy smoothing -> no jitter
  Fast movement -> high cutoff -> light smoothing -> no lag
"""

import numpy as np


# --- Low-Pass Filter (building block) ----------------------------------------

class LowPassFilter:
    """First-order exponential low-pass filter (IIR)."""

    __slots__ = ("_y", "_s", "_initialized")

    def __init__(self):
        self._y = 0.0
        self._s = 0.0
        self._initialized = False

    def reset(self):
        self._initialized = False

    @property
    def prev_raw(self):
        return self._y

    def __call__(self, value: float, alpha: float) -> float:
        if not self._initialized:
            self._s = value
            self._initialized = True
        else:
            self._s = alpha * value + (1.0 - alpha) * self._s
        self._y = value
        return self._s


# --- One Euro Filter ----------------------------------------------------------

class OneEuroFilter:
    """
    One Euro adaptive low-pass filter for a single scalar signal.

    Parameters
    ----------
    freq : float
        Initial estimate of signal frequency (fps). Auto-updates each call.
    min_cutoff : float
        Minimum cutoff frequency (Hz). Lower = more smoothing at low speed.
    beta : float
        Speed coefficient. Higher = less lag during fast movement.
    d_cutoff : float
        Cutoff for the derivative low-pass filter. Usually 1.0.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0):
        self.freq       = freq
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x_filt    = LowPassFilter()
        self._dx_filt   = LowPassFilter()
        self._last_time = None

    def reset(self):
        self._x_filt.reset()
        self._dx_filt.reset()
        self._last_time = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        te  = 1.0 / freq
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, t: float = None) -> float:
        # Auto-update frequency from timestamps
        if t is not None:
            if self._last_time is not None:
                dt = t - self._last_time
                if dt > 1e-6:
                    self.freq = 1.0 / dt
            self._last_time = t

        # Derivative estimate
        prev = self._x_filt.prev_raw if self._x_filt._initialized else x
        dx   = (x - prev) * self.freq

        # Smoothed derivative
        edx = self._dx_filt(dx, self._alpha(self.d_cutoff, self.freq))

        # Adaptive cutoff: faster movement -> higher cutoff -> less smoothing
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter the signal
        return self._x_filt(x, self._alpha(cutoff, self.freq))


# --- 2-D One Euro Filter -----------------------------------------------------

class OneEuroFilter2D:
    """
    Independent One Euro filters for X and Y axes.

    Usage:
        f = OneEuroFilter2D(freq=30, min_cutoff=1.0, beta=0.05)
        sx, sy = f(raw_x, raw_y)          # auto-timestamps
        sx, sy = f(raw_x, raw_y, t=now)   # explicit timestamp
    """

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.fx = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x, y, t=None):
        return self.fx(x, t), self.fy(y, t)


# --- Exponential Moving Average -----------------------------------------------

class EMA:
    """Simple exponential moving average for scalar signals."""

    __slots__ = ("_alpha", "_value", "_initialized")

    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        self._value = 0.0
        self._initialized = False

    def reset(self):
        self._initialized = False

    @property
    def value(self):
        return self._value

    def __call__(self, x: float) -> float:
        if not self._initialized:
            self._value = x
            self._initialized = True
        else:
            self._value = self._alpha * x + (1.0 - self._alpha) * self._value
        return self._value
