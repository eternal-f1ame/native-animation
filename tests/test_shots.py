"""Shot windowing: scene list -> extractable [start, end) windows."""
from native_animation.data.shots import plan_shot_windows


def test_short_scenes_are_dropped():
    assert plan_shot_windows([(0.0, 1.5)]) == []          # < 2.2 s
    assert plan_shot_windows([(0.0, 2.4)]) == [(0.0, 2.4)]


def test_long_scene_is_tiled_without_overlap_and_remainder_kept_if_long_enough():
    windows = plan_shot_windows([(0.0, 23.0)], min_s=2.2, max_s=10.0)
    assert windows == [(0.0, 10.0), (10.0, 20.0), (20.0, 23.0)]  # 3.0 s remainder kept


def test_short_remainder_is_dropped():
    windows = plan_shot_windows([(0.0, 21.0)], min_s=2.2, max_s=10.0)
    assert windows == [(0.0, 10.0), (10.0, 20.0)]          # 1.0 s remainder dropped


def test_multiple_scenes_concatenate_in_order():
    windows = plan_shot_windows([(0.0, 3.0), (3.0, 4.0), (4.0, 9.5)])
    assert windows == [(0.0, 3.0), (4.0, 9.5)]             # middle scene too short
