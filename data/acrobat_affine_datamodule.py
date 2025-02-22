from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.acrobat_affine_dataset import AcrobatAffineDataset


class BaseAcrobatAffineDataModule(pl.LightningDataModule):
    def __init__(self, root_path, resize_shape: Tuple[int, int], warp_offset: int, batch_size: int, num_workers: int):
        super(BaseAcrobatAffineDataModule, self).__init__()
        self.root_path = root_path
        self.train_list_path = train_list_path
        self.val_list_path = val_list_path
        self.test_list_path = test_list_path

        self.balanced_train = balanced_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.resize_shape = resize_shape
        self.warp_offset = warp_offset

        self.train_batch_collate_fn = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = AcrobatAffineDataset(
            root_path=self.root_path,
            resize_shape=self.resize_shape,
            offset=self.warp_offset
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_batch_collate_fn
        )


if __name__ == '__main__':
    import pathlib

    dm = AcrobatAffineDataModule(
        pathlib.Path('/datasets/extra_space2/ostap/temp_dataset'), resize_shape=(960 + 128 * 2, 720 + 128 * 2),
        warp_offset=128, batch_size=4, num_workers=4
    )
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    print(batch['image0'].shape)
    print(batch['image1'].shape)
    print(batch['transformation']['H'].shape)
