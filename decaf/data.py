from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import decaf.logger as log
from opacus.data_loader import DPDataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: list) -> None:
        data = np.array(data, dtype="float32")
        self.x = torch.from_numpy(data)
        self.n_samples = self.x.shape[0]
        log.info("***** DATA ****")
        log.info(f"n_samples = {self.n_samples}")

    def __getitem__(self, index: int) -> Any:
        return self.x[index]

    def __len__(self) -> int:
        return self.n_samples


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: list,
        val_data: list,
        data_dir: Path = Path.cwd(),
        batch_size: int = 64,
        num_workers: int = 0,
        enable_dp: bool = False
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = Dataset(train_data)
        self.val_dataset = Dataset(val_data)
        self.dims = self.train_dataset.x.shape[1:]
        self.enable_dp = enable_dp

    def train_dataloader(self) -> DataLoader:
        if not self.enable_dp:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
        
        return DPDataLoader(self.train_dataset, sample_rate=self.batch_size/len(self.train_dataset), num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=32
        )

# class DataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         data: list,
#         data_dir: Path = Path.cwd(),
#         batch_size: int = 64,
#         num_workers: int = 0,
#         valid_data: list = [],
#         enable_dp: bool = False
#     ) -> None:
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.dataset = Dataset(data)
#         self.data_val = Dataset(valid_data)
#         self.dims = self.dataset.x.shape[1:]

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.data_val, batch_size=self.batch_size, num_workers=self.num_workers
#         )

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.data_test, batch_size=self.batch_size, num_workers=self.num_workers
#         )