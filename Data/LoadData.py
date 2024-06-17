import numpy as np
from torchgeo.datamodules import QuakeSetDataModule


def read_partition(split: str = 'train'):
    data_module = QuakeSetDataModule(download=False)
    if split == 'train':
        data_module.setup(stage="fit")
        loader = data_module.train_dataloader()
    elif split == 'validation':
        data_module.setup(stage="validate")
        loader = data_module.val_dataloader()
    else:
        data_module.setup(stage="test")
        loader = data_module.test_dataloader()
    images_list, labels_list, magnitudes_list = [], [], []
    for batch in loader:
        images, labels, mag = batch["image"], batch["label"], batch["magnitude"]
        images_list.append(images.numpy())  # Convert torch tensor to numpy array
        labels_list.append(labels.numpy())
        magnitudes_list.append(mag.numpy())
    images = np.concatenate(images_list)
    labels = np.concatenate(labels_list)
    magnitudes = np.concatenate(magnitudes_list)

    return images, labels, magnitudes
