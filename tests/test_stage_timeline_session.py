"""Timeline staging: mapping a shared timeline with inserted frames back to each camera's video frames."""

import numpy as np

from smal_fitter.sleap_data.stage_timeline_session import align_timeline_to_video


def test_align_timeline_to_video_with_inserted_and_skipped_frames():
    """Inserted (interpolated) timeline frames map to -1, frames after skipped video frames take the new offset."""
    rng = np.random.default_rng(0)
    video_xy = rng.uniform(0, 1000, size=(40, 5, 2))
    video_xy[10] = np.nan  # no detection: resolved from its neighbours, which share the offset
    video_xy[20:23] = np.nan  # no detections around the skip: video 22's neighbours disagree, so it stays unresolved
    # Timeline: video 0-11, two interpolated frames, video 12-19, (video 20-21 skipped), video 22-39
    truth = np.array(list(range(12)) + [-1, -1] + list(range(12, 20)) + list(range(22, 40)))
    timeline_xy = video_xy[np.maximum(truth, 0)]
    timeline_xy[12] = (2 * video_xy[11] + video_xy[12]) / 3
    timeline_xy[13] = (video_xy[11] + 2 * video_xy[12]) / 3

    fmap = align_timeline_to_video(timeline_xy, video_xy)

    expected = truth.copy()
    expected[truth == 22] = -1
    np.testing.assert_array_equal(fmap, expected)
