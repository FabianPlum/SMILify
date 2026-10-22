"""Per-joint rotation-limit prior (issue #56).

Bounds are interpreted per axis in the AXIS-ANGLE space of ``joint_rotations``.
Note the representation caveat (issue #56 review): axis-angle rotations have a
canonical magnitude of at most pi, so independent per-axis bounds are
mathematically non-unique for |theta| > pi — a "270 degree" rotation about an
axis is the same physical rotation as -90 degrees about it, and the hinge loss
will accept whichever representation the pose happens to use. Author limits
within [-pi, +pi] per axis (Blender's Limit Rotation UI maps onto this
directly); exactly +/-pi means "free". Nothing is clamped incorrectly at the
+/-pi boundary itself — the hinge is exactly 0 there.
See docs/joint_limits_user_guide.md (section 2 caveat) for the author-facing
explanation.
"""

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


class LimitPrior(object):
    def __init__(self):
        # Ranges are built here from the current config.dd, never at import time: a
        # malformed 'joint_limits' must not crash every importer (issue #56 review),
        # and apply_smal_file_override may swap config.dd after import. Reads the
        # user-defined 'joint_limits' from the model .pkl when present, otherwise
        # falls back to a wide-open range (see helper).
        ranges = _ranges_from_joint_limits(config.dd)

        self.parts = {}
        for j, joint in enumerate(config.dd["J_names"]):
            self.parts[joint] = j
        self.id2name = {v: k for k, v in self.parts.items()}
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
        Given x, the flattened rel rotations of all J joints, for each part compute the limit value.
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
