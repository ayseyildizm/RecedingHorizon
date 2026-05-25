"""
Kalman filter for tracking the 3D position of the target node across views.

Implements Section IV-E of Burusa et al. (ICRA 2024), simplified to the
single-target case used in this thesis (one bunny model vs Burusa's
multi-node tomato plants).

State (Burusa IV-E):
    p^j_k = {mu^j_k, Sigma^j_k}     # Gaussian mean + 3x3 covariance
    c^j_k                           # class label

Prediction step:
    Static scene assumption -- position and class label held constant.

Update step:
    Standard Kalman update minimising MSE between measured and estimated
    3D position. Class label updated by majority voting (in the multi-node
    case); here we have a single class so this reduces to identity.

References
----------
Burusa, van Henten, Kootstra (ICRA 2024), Section IV-E.
"""

import numpy as np


class NodeKalmanFilter:
    """Kalman filter for a single 3D-position target.

    Parameters
    ----------
    initial_position : np.ndarray shape (3,)
        Initial position estimate mu_0. For Burusa-style "predefined view 0"
        protocol this is the user-supplied target_params, since at iter 0
        the planner is given a rough indication of where the target is
        (Section IV-B, "We assume that the location of ROI within V^T is
        given").
    initial_uncertainty : float
        Initial standard deviation along each axis (m). Default 0.017 m
        matches Burusa's reported sigma at iter 0 (1.7e-2 m, Table II).
    """

    # Process noise (Q): static-scene assumption => very small.
    # Kept small but non-zero for numerical stability.
    _PROCESS_NOISE_STD = 1e-4  # m per step

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_uncertainty: float = 0.017,
    ) -> None:
        self.mu = np.asarray(initial_position, dtype=np.float64).reshape(3)
        self.Sigma = np.eye(3, dtype=np.float64) * (initial_uncertainty ** 2)
        self.Q = np.eye(3, dtype=np.float64) * (self._PROCESS_NOISE_STD ** 2)
        # Class label storage (majority voting, Burusa IV-E).
        # Single-target: class 0 = "target", set on first measurement.
        self.class_votes = {}

    def predict(self) -> None:
        """Static-scene prediction step (Burusa IV-E.1).

        Position and class label held constant. Covariance grows by Q to
        reflect mild process uncertainty between steps.
        """
        # mu unchanged
        self.Sigma = self.Sigma + self.Q

    def update(
        self,
        measurement: np.ndarray,
        measurement_uncertainty: float,
        class_label: int = 0,
    ) -> None:
        """Kalman update with a new 3D measurement (Burusa IV-E.2).

        Parameters
        ----------
        measurement : np.ndarray shape (3,)
            Observed 3D centroid of the target this view.
        measurement_uncertainty : float
            Per-axis standard deviation of the measurement (m). Burusa
            (Section IV-A) uses the variance of the detected point set
            along the viewing direction as this estimate.
        class_label : int
            Observed class label for the target (0 = target / bunny here).
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(3)
        R = np.eye(3, dtype=np.float64) * (measurement_uncertainty ** 2)

        # Standard Kalman update (H = I since we measure position directly).
        S = self.Sigma + R                          # innovation covariance
        K = self.Sigma @ np.linalg.inv(S)           # Kalman gain
        innovation = z - self.mu
        self.mu = self.mu + K @ innovation
        self.Sigma = (np.eye(3) - K) @ self.Sigma

        # Majority voting on class label
        self.class_votes[class_label] = self.class_votes.get(class_label, 0) + 1

    @property
    def estimated_class(self) -> int:
        """Most-voted class label so far (Burusa IV-E.2)."""
        if not self.class_votes:
            return -1
        return max(self.class_votes, key=self.class_votes.get)

    @property
    def sigma_scalar(self) -> float:
        """Scalar position uncertainty for the metrics table.

        Burusa Table II reports sigma in m * 10^-2. We follow the convention
        of using the average of the per-axis standard deviations (matches
        Burusa's reported scalar values like 1.7, 1.6, 1.4 which are clearly
        per-axis, not trace-based).
        """
        return float(np.mean(np.sqrt(np.diag(self.Sigma))))

    @property
    def position(self) -> np.ndarray:
        """Current best position estimate."""
        return self.mu.copy()
