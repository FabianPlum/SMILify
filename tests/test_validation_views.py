"""Multi-view training samples num_views_to_use views per sample; validation sees every view."""

from torch.utils.data import DataLoader

from smal_fitter.neuralSMIL.train_multiview_regressor import training_sampling
from smal_fitter.sleap_data.sleap_multiview_dataset import SLEAPMultiViewDataset, multiview_collate_fn
from tests.multiview_h5 import write_multiview_h5


def test_training_samples_num_views_to_use_and_validation_every_view(tmp_path):
    path = tmp_path / "multiview.h5"
    write_multiview_h5(path, views_per_sample=[5, 5, 3, 5])
    dataset = SLEAPMultiViewDataset(str(path), num_views_to_use=None)  # as the trainer builds it
    # One worker each: the workers copy the dataset when iteration starts, as in training
    train_loader, val_loader = (
        DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=multiview_collate_fn) for _ in range(2)
    )

    def camera_indices(loader):
        return [x["camera_indices"].tolist() for x_batch, _ in loader for x in x_batch]

    with training_sampling(dataset, augment=False, num_views_to_use=3):
        train_views = camera_indices(train_loader)
    val_views = camera_indices(val_loader)

    assert [len(v) for v in train_views] == [3, 3, 3, 3]
    assert val_views == [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2], [0, 1, 2, 3, 4]]
