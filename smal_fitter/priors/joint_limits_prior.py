import numpy as np
import config


def _ranges_from_joint_limits(dd, default_range=np.pi):
    """Build a ``{joint_name: [[min, max] x 3]}`` ranges dict for an arbitrary rig.

    Issue #56: prefer user-defined per-joint rotation limits stored in the model
    ``.pkl`` under the ``joint_limits`` key. Expected shape ``(J, 3, 2)`` in the same
    order as ``dd['J_names']`` (radians, interpreted per axis in the axis-angle space
    of ``joint_rotations``), where ``[:, axis, 0]`` is the min and ``[:, axis, 1]`` the
    max for each axis.

    When ``joint_limits`` is absent, fall back to a wide-open
    ``[-default_range, +default_range]`` per axis so that new models are NOT
    pose-frozen. (The previous placeholder clamped every non-root joint to
    +/-0.01 rad, which effectively locked the pose whenever the limit loss was on.)

    The root joint (``J_names[0]``) is always fixed at ``[0, 0]`` on every axis; the
    fitter ignores it via the ``[3:]`` slice on ``min_values``/``max_values``.
    """
    joint_names = dd["J_names"]
    root_joint = joint_names[0]
    joint_limits = dd.get("joint_limits", None)
    ranges = {}

    if joint_limits is not None:
        jl = np.asarray(joint_limits, dtype=np.float64)
        expected = (len(joint_names), 3, 2)
        if jl.shape != expected:
            raise ValueError(
                f"'joint_limits' has shape {jl.shape}, expected {expected} (one [min,max] per axis per joint)."
            )
        if not np.all(jl[..., 0] <= jl[..., 1]):
            raise ValueError("'joint_limits' has min > max for at least one joint/axis.")
        if not np.isfinite(jl).all():
            raise ValueError("'joint_limits' contains non-finite values.")
        for j, joint in enumerate(joint_names):
            ranges[joint] = jl[j].tolist()  # [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    else:
        w = float(default_range)
        for joint in joint_names:
            ranges[joint] = [[-w, w], [-w, w], [-w, w]]

    # Root joint is fixed and ignored downstream.
    ranges[root_joint] = [[0, 0], [0, 0], [0, 0]]
    return ranges


if config.ignore_hardcoded_body:
    Ranges = _ranges_from_joint_limits(config.dd)
else:
    Ranges = {
        "pelvis": [[0, 0], [0, 0], [0, 0]],
        "pelvis0": [[-0.3, 0.3], [-1.2, 0.5], [-0.1, 0.1]],
        "spine": [[-0.4, 0.4], [-1.0, 0.9], [-0.8, 0.8]],
        "spine0": [[-0.4, 0.4], [-1.0, 0.9], [-0.8, 0.8]],
        "spine1": [[-0.4, 0.4], [-0.5, 1.2], [-0.4, 0.4]],
        "spine3": [[-0.5, 0.5], [-0.6, 1.4], [-0.8, 0.8]],
        "spine2": [[-0.5, 0.5], [-0.4, 1.4], [-0.5, 0.5]],
        "RFootBack": [[-0.2, 0.3], [-0.3, 1.1], [-0.3, 0.5]],
        "LFootBack": [[-0.3, 0.2], [-0.3, 1.1], [-0.5, 0.3]],
        "LLegBack1": [[-0.2, 0.3], [-0.5, 0.8], [-0.5, 0.4]],
        "RLegBack1": [[-0.3, 0.2], [-0.5, 0.8], [-0.4, 0.5]],
        "Head": [[-0.5, 0.5], [-1.0, 0.9], [-0.9, 0.9]],
        "RLegBack2": [[-0.3, 0.2], [-0.6, 0.8], [-0.5, 0.6]],
        "LLegBack2": [[-0.2, 0.3], [-0.6, 0.8], [-0.6, 0.5]],
        "RLegBack3": [[-0.2, 0.3], [-0.8, 0.2], [-0.4, 0.5]],
        "LLegBack3": [[-0.3, 0.2], [-0.8, 0.2], [-0.5, 0.4]],
        "Mouth": [[-0.1, 0.1], [-1.1, 0.5], [-0.1, 0.1]],
        "Neck": [[-0.8, 0.8], [-1.0, 1.0], [-1.1, 1.1]],
        "LLeg1": [[-0.05, 0.05], [-1.3, 0.8], [-0.6, 0.6]],  # Extreme
        "RLeg1": [[-0.05, 0.05], [-1.3, 0.8], [-0.6, 0.6]],
        "RLeg2": [[-0.05, 0.05], [-1.0, 0.9], [-0.6, 0.6]],  # Extreme
        "LLeg2": [[-0.05, 0.05], [-1.0, 1.1], [-0.6, 0.6]],
        "RLeg3": [[-0.1, 0.4], [-0.3, 1.4], [-0.4, 0.7]],  # Extreme
        "LLeg3": [[-0.4, 0.1], [-0.3, 1.4], [-0.7, 0.4]],
        "LFoot": [[-0.3, 0.1], [-0.4, 1.5], [-0.7, 0.3]],  # Extreme
        "RFoot": [[-0.1, 0.3], [-0.4, 1.5], [-0.3, 0.7]],
        "Tail7": [[-0.1, 0.1], [-0.7, 1.1], [-0.9, 0.8]],
        "Tail6": [[-0.1, 0.1], [-1.4, 1.4], [-1.0, 1.0]],
        "Tail5": [[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]],
        "Tail4": [[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]],
        "Tail3": [[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]],
        "Tail2": [[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]],
        "Tail1": [[-0.1, 0.1], [-1.5, 1.4], [-1.2, 1.2]],
    }


class LimitPrior(object):
    def __init__(self):
        if config.ignore_hardcoded_body:
            # Build ranges at init time from the current config.dd, not the
            # module-level Ranges which may have been built with a different
            # SMAL model at import time (e.g. before apply_smal_file_override).
            # Issue #56: reads user-defined 'joint_limits' from the model .pkl when
            # present, otherwise falls back to a wide-open range (see helper).
            ranges = _ranges_from_joint_limits(config.dd)

            self.parts = {}
            for j, joint in enumerate(config.dd["J_names"]):
                self.parts[joint] = j
        else:
            ranges = Ranges
            self.parts = {
                "pelvis0": 0,
                "spine": 1,
                "spine0": 2,
                "spine1": 3,
                "spine2": 4,
                "spine3": 5,
                "LLeg1": 6,
                "LLeg2": 7,
                "LLeg3": 8,
                "LFoot": 9,
                "RLeg1": 10,
                "RLeg2": 11,
                "RLeg3": 12,
                "RFoot": 13,
                "Neck": 14,
                "Head": 15,
                "LLegBack1": 16,
                "LLegBack2": 17,
                "LLegBack3": 18,
                "LFootBack": 19,
                "RLegBack1": 20,
                "RLegBack2": 21,
                "RLegBack3": 22,
                "RFootBack": 23,
                "Tail1": 24,
                "Tail2": 25,
                "Tail3": 26,
                "Tail4": 27,
                "Tail5": 28,
                "Tail6": 29,
                "Tail7": 30,
                "Mouth": 31,
            }
        self.id2name = {v: k for k, v in self.parts.items()}
        # Ignore the first joint.
        self.prefix = 3
        self.part_ids = np.array(sorted(self.parts.values()))
        self.min_values = np.hstack(
            [np.array(np.array(ranges[self.id2name[part_id]])[:, 0]) for part_id in self.part_ids]
        )
        self.max_values = np.hstack(
            [np.array(np.array(ranges[self.id2name[part_id]])[:, 1]) for part_id in self.part_ids]
        )
        self.ranges = ranges

    def __call__(self, x, xp):
        """
        Given x, rel rotation of 31 joints, for each parts compute the limit value.
        k is steepness of the curve, max_val + margin is the midpoint of the curve (val 0.5)
        Using Logistic:
        max limit: 1/(1 + exp(k * ((max_val + margin) - x)))
        min limit: 1/(1 + exp(k * (x - (min_val - margin))))
        With max/min:
        minlimit: max( min_vals - x , 0 )
        maxlimit: max( x - max_vals , 0 )
        With exponential:
        min: exp(k * (minval - x) )
        max: exp(k * (x - maxval) )
        """
        ## Max/min discontinous but fast. (flat + L2 past the limit)
        zeros = xp.zeros_like(x)
        return np.maximum(x - self.max_values, zeros) + np.maximum(self.min_values - x, zeros)

    def report(self, x):
        res = self(x).r.reshape(-1, 3)
        values = x[self.prefix :].r.reshape(-1, 3)
        bad = np.any(res > 0, axis=1)
        bad_ids = np.array(self.part_ids)[bad]
        np.set_printoptions(precision=3)
        for bad_id in bad_ids:
            name = self.id2name[bad_id]
            limits = self.ranges[name]
            (print("%s over! Overby:" % name),)
            (print(res[bad_id - 1, :]),)
            (print(" Limits:"),)
            (print(limits),)
            (print(" Values:"),)
            print(values[bad_id - 1, :])
