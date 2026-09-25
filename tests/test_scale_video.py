import pytest
from shelf_vision.scale_video import tile_boxes, remap, suppress, Tracker


def test_tiles_cover_far_edges_without_out_of_bounds_or_duplicates():
    boxes = tile_boxes(1300, 1920, 960, 0.25)
    assert len(boxes) == len({tuple(b) for b in boxes}) == 6
    assert [340, 960, 1300, 1920] in boxes
    for y in range(1920):
        assert any(b[1] <= y < b[3] for b in boxes)
    assert tile_boxes(100, 200, 960, 0.25) == [[0, 0, 100, 200]]
    with pytest.raises(ValueError):
        tile_boxes(100, 100, 10, 1)


def test_tile_predictions_are_offset_clipped_and_degenerate_removed():
    rows = [{'bbox_xyxy': [-5, 10, 80, 120], 'confidence': .8},
            {'bbox_xyxy': [200, 200, 210, 210], 'confidence': .7}]
    assert remap(rows, [100, 200, 160, 300]) == [
        {'bbox_xyxy': [100, 210, 160, 300], 'confidence': .8}]


def test_overlap_merge_keeps_best_duplicate_but_not_adjacent_tray():
    rows = [{'bbox_xyxy': [0, 0, 10, 10], 'confidence': .8},
            {'bbox_xyxy': [1, 0, 11, 10], 'confidence': .9},
            {'bbox_xyxy': [12, 0, 22, 10], 'confidence': .7}]
    assert suppress(rows, .5) == [rows[1], rows[2]]


def detection(box):
    return {'bbox_xyxy': box, 'confidence': .8}


def test_camera_translation_preserves_id_and_carried_boxes_do_not_add_hits():
    tracker = Tracker(min_hits=2, max_missing=1)
    assert tracker.update([detection([0, 0, 10, 10])])[0]['confirmed'] is False
    row = tracker.update([detection([20, 0, 30, 10])], [[1, 0, 20], [0, 1, 0]])[0]
    assert row['id'] == 1 and row['confirmed'] and row['hits'] == 2
    carried = tracker.update([])[0]
    assert not carried['observed'] and carried['hits'] == 2
    assert tracker.update([]) == []
    assert tracker.update([detection([20, 0, 30, 10])])[0]['id'] == 2


def test_association_is_one_to_one_and_tentative_tracks_need_real_hits():
    tracker = Tracker(min_hits=3)
    tracker.update([detection([0, 0, 10, 10]), detection([12, 0, 22, 10])])
    rows = tracker.update([detection([1, 0, 11, 10])])
    assert sum(r['observed'] for r in rows) == 1
    assert not any(r['confirmed'] for r in rows)
