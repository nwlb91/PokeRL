"""Learning plateau detection using statistical trend analysis.

Uses OLS linear regression over a sliding window of win rate observations.
A plateau is declared when:
  1. The slope of the regression is not significantly different from zero
     (two-tailed t-test, p > alpha), AND
  2. The coefficient of variation is below a stability threshold, ruling out
     high-variance oscillation that happens to have near-zero slope.

Both conditions must hold for ``patience`` consecutive evaluations before
the detector fires, guarding against transient flat spots.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PlateauInfo:
    """Snapshot of plateau-detection state, returned every evaluation."""

    is_plateau: bool = False
    # How many consecutive evaluations the plateau condition has held.
    streak: int = 0
    # Current window slope (win-rate change per evaluation step).
    slope: float = 0.0
    # Two-tailed p-value for slope == 0.
    p_value: float = 1.0
    # Standard deviation of win rates in the current window.
    std: float = 0.0
    # Coefficient of variation (std / mean) in current window.
    cv: float = 0.0
    # Full history for plotting.
    win_rate_history: List[float] = field(default_factory=list)
    battle_history: List[int] = field(default_factory=list)
    # Indices into history where plateau regions start/end.
    plateau_regions: List[Tuple[int, int]] = field(default_factory=list)


class PlateauDetector:
    """Detect learning plateaus from a stream of win-rate observations.

    Parameters
    ----------
    window : int
        Number of most-recent observations used for the regression.
        Must be >= 5 to have meaningful statistics.
    patience : int
        How many consecutive "flat" evaluations before declaring a plateau.
    alpha : float
        Significance level for the two-tailed t-test on the slope.
        A larger alpha is *stricter* (easier to declare plateau).
    cv_threshold : float
        Maximum coefficient of variation (std/mean) allowed.  If the CV
        exceeds this, the signal is too noisy to call it a plateau.
    """

    def __init__(
        self,
        window: int = 20,
        patience: int = 10,
        alpha: float = 0.05,
        cv_threshold: float = 0.10,
    ):
        if window < 5:
            raise ValueError("window must be >= 5")
        self.window = window
        self.patience = patience
        self.alpha = alpha
        self.cv_threshold = cv_threshold

        # Internal state
        self._win_rates: List[float] = []
        self._battle_counts: List[int] = []
        self._streak: int = 0
        self._plateau_regions: List[Tuple[int, int]] = []
        self._in_plateau_since: Optional[int] = None  # index where current plateau started

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, win_rate: float, battle_count: int) -> PlateauInfo:
        """Record a new win-rate observation and return detection state."""
        self._win_rates.append(win_rate)
        self._battle_counts.append(battle_count)

        idx = len(self._win_rates) - 1

        if len(self._win_rates) < self.window:
            # Not enough data yet.
            return self._make_info(is_flat=False)

        # Grab the most recent `window` observations.
        y = self._win_rates[-self.window :]

        slope, p_value, std, cv = self._regression_stats(y)

        # Condition 1: slope not significantly != 0
        slope_flat = p_value > self.alpha
        # Condition 2: low noise
        stable = cv <= self.cv_threshold

        is_flat = slope_flat and stable

        if is_flat:
            self._streak += 1
            if self._streak >= self.patience and self._in_plateau_since is None:
                self._in_plateau_since = idx - self.patience + 1
        else:
            # Streak broken — close any open plateau region.
            if self._in_plateau_since is not None:
                self._plateau_regions.append((self._in_plateau_since, idx - 1))
                self._in_plateau_since = None
            self._streak = 0

        is_plateau = self._streak >= self.patience

        return self._make_info(
            is_flat=is_plateau,
            slope=slope,
            p_value=p_value,
            std=std,
            cv=cv,
        )

    def reset(self):
        """Clear all accumulated history."""
        self._win_rates.clear()
        self._battle_counts.clear()
        self._streak = 0
        self._plateau_regions.clear()
        self._in_plateau_since = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _regression_stats(y: List[float]) -> Tuple[float, float, float, float]:
        """OLS regression of *y* against integer indices 0..n-1.

        Returns (slope, p_value, std_y, coeff_of_variation).
        """
        n = len(y)
        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / n

        # Sums for OLS
        ss_xy = 0.0
        ss_xx = 0.0
        ss_yy = 0.0
        for i, yi in enumerate(y):
            dx = i - x_mean
            dy = yi - y_mean
            ss_xy += dx * dy
            ss_xx += dx * dx
            ss_yy += dy * dy

        slope = ss_xy / ss_xx if ss_xx > 0 else 0.0

        # Residual standard error
        y_hat_ss = 0.0
        for i, yi in enumerate(y):
            residual = yi - (y_mean + slope * (i - x_mean))
            y_hat_ss += residual * residual

        if n > 2:
            se_residual = math.sqrt(y_hat_ss / (n - 2))
            se_slope = se_residual / math.sqrt(ss_xx) if ss_xx > 0 else float("inf")
        else:
            se_slope = float("inf")

        # t-statistic for H0: slope == 0
        if se_slope > 0 and se_slope != float("inf"):
            t_stat = slope / se_slope
        else:
            t_stat = 0.0

        # Approximate two-tailed p-value using the t-distribution.
        # For df >= 5, the normal approximation is reasonable; for a
        # lightweight, dependency-free implementation we use the
        # regularised incomplete beta function approach.
        df = n - 2
        p_value = _two_tailed_t_pvalue(t_stat, df)

        std_y = math.sqrt(ss_yy / n) if n > 0 else 0.0
        cv = std_y / y_mean if y_mean > 0 else 0.0

        return slope, p_value, std_y, cv

    def _make_info(
        self,
        is_flat: bool,
        slope: float = 0.0,
        p_value: float = 1.0,
        std: float = 0.0,
        cv: float = 0.0,
    ) -> PlateauInfo:
        # Collect closed + any currently-open plateau region.
        regions = list(self._plateau_regions)
        if self._in_plateau_since is not None:
            regions.append((self._in_plateau_since, len(self._win_rates) - 1))

        return PlateauInfo(
            is_plateau=is_flat,
            streak=self._streak,
            slope=slope,
            p_value=p_value,
            std=std,
            cv=cv,
            win_rate_history=list(self._win_rates),
            battle_history=list(self._battle_counts),
            plateau_regions=regions,
        )


# ------------------------------------------------------------------
# Lightweight t-distribution p-value (no scipy dependency)
# ------------------------------------------------------------------

def _two_tailed_t_pvalue(t: float, df: int) -> float:
    """Approximate two-tailed p-value for Student's t distribution.

    Uses the regularised incomplete beta function:
        p = I_{df/(df+t^2)}(df/2, 1/2)
    which is exact.  We evaluate it via a continued-fraction expansion
    that converges quickly for the parameter ranges we encounter.
    """
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    return _regularised_beta(x, a, b)


def _regularised_beta(x: float, a: float, b: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """Regularised incomplete beta function I_x(a, b) via Lentz's algorithm."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use the continued fraction that converges when x < (a+1)/(a+b+2).
    # If not, use the symmetry relation I_x(a,b) = 1 - I_{1-x}(b,a).
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_beta(1.0 - x, b, a, max_iter, tol)

    # Front factor: x^a * (1-x)^b / (a * B(a,b))
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    # Evaluate continued fraction with modified Lentz's method.
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d

        # Odd step
        numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < tol:
            break

    return front * f
