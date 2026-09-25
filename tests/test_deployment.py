import pytest
from shelf_vision.deployment_probe import yolo_box


def test_roi_training_boxes_are_clipped_and_relative_to_crop():
    assert yolo_box([80,30,160,70],[100,20,200,120])==pytest.approx([.3,.3,.6,.4])
    with pytest.raises(ValueError): yolo_box([0,0,5,5],[100,100,200,200])


def test_each_dataset_build_excludes_stale_train_and_validation_files(tmp_path):
    from shelf_vision.deployment_probe import fresh_dataset_directory
    first=fresh_dataset_directory(tmp_path)
    (first/'images/train').mkdir(parents=True)
    (first/'images/train/old.jpg').write_bytes(b'old source assigned to train')
    second=fresh_dataset_directory(tmp_path)
    assert second!=first
    assert not list(second.rglob('old.jpg'))
    assert (first/'images/train/old.jpg').exists()


def test_native_and_export_inference_use_identical_square_preprocessing():
    from shelf_vision.deployment_probe import parity_predict
    class FakeModel:
        def __call__(self,path,**kwargs):
            # Enforces the integration boundary, not framework internals.
            assert kwargs['rect'] is False
            assert kwargs['imgsz']==640
            return ['result']
    assert parity_predict(FakeModel(),'sample.jpg')=='result'
