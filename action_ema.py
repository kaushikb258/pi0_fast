"""Action-EMA on final simulator motion commands, with no gripper filtering."""
import numpy as np


def validate_lambda(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError("action-ema lambda must be finite and satisfy 0 <= lambda < 1")
    return value


class ActionEMA:
    """Constant motion EMA; lambda=0 gives the unfiltered baseline."""
    def __init__(self, smoothing_lambda=0.):
        self.smoothing_lambda = validate_lambda(smoothing_lambda)
        self.reset()

    def reset(self):
        self.previous = None
        self.last_lambdas = np.zeros(6)

    def apply(self, action):
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite seven-component simulator action")
        output = action.copy()
        weights = np.zeros(6)
        if self.previous is not None:
            weights.fill(self.smoothing_lambda)
            if np.any(weights):
                output = output.astype(np.result_type(action.dtype, np.float32), copy=False)
                output[:6] = (self.smoothing_lambda*self.previous[:6]
                              + (1-self.smoothing_lambda)*action[:6])
        self.last_lambdas = weights.copy()
        self.previous = output.copy()
        return output


def action_change_metrics(actions):
    """RMS L2 change per control step, separately for XYZ and rotation commands.

    These are command-space differences, not measured physical acceleration.
    No difference is taken across episodes or from settling dummy actions.
    """
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 7)
    delta = np.diff(actions[:, :6], axis=0)
    if len(delta) == 0:
        return {"transitions": 0, "translation_delta_rms": None, "rotation_delta_rms": None}
    return {"transitions": len(delta),
            "translation_delta_rms": float(np.sqrt(np.mean(np.sum(delta[:, :3] ** 2, axis=1)))),
            "rotation_delta_rms": float(np.sqrt(np.mean(np.sum(delta[:, 3:] ** 2, axis=1))))}
