import os.path

import pytorch_lightning as pl
import numpy as np
from .datasets import LULC, LULCSubset, get_global_idxs, LULCOptim
import torch
import torch.utils.data as data
from typing import Optional
from sklearn.model_selection import train_test_split

from msmtu.transforms import StandardizeTS

mean_modis = np.array([1778.0370114114635, 2885.055655885381, 1256.2631572253467, 1586.7335998607268,
                       2817.6737163027287, 2420.865232616906, 1763.7376743953382])

std_modis = np.array([2067.9758198306, 1811.1213005051807, 2146.4020578479854, 2059.230258563475,
                      1454.778090999886, 1595.473540093584, 1487.7149351931803])

mean_long = -4.5726
std_long = 1.42003659

mean_lat = 37.4675
std_lat = 0.54427359

mean_altitude = 526.941630062964
std_altitude = 434.20557148483016

mean_slope = 9.098918613850193
std_slope = 6.635985730650851

mean_precipitation = 3398.9591808260107
std_precipitation = 1157.0350863758044

mean_evapotranspiration = 5147.744891820799
std_evapotranspiration = 466.2150145975434

mean_temperature_ave = 15.593676475326593
std_temperature_ave = 1.9789136521273902

mean_temperature_max = 21.606108
std_temperature_max = 2.076332

mean_temperature_min = 10.047357
std_temperature_min = 2.012799


def collate_fn(recs):
    forward = np.array(list(map(lambda x: x['forward'], recs)))
    backward = np.array(list(map(lambda x: x['backward'], recs)))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(np.array(list(map(lambda r: r['values'], recs))))
        masks = torch.FloatTensor(np.array(list(map(lambda r: r['masks'], recs))))
        deltas = torch.FloatTensor(np.array(list(map(lambda r: r['deltas'], recs))))

        return {'values': values, 'masks': masks, 'deltas': deltas}

    ret_dict = {
        'forward': to_tensor_dict(forward),
        'backward': to_tensor_dict(backward),
        'labels': torch.FloatTensor(np.array(list(map(lambda x: x['label'], recs)))),
        'probs': torch.FloatTensor(np.array(list(map(lambda x: x['probs'], recs)))),
        'ancillary': torch.FloatTensor(np.array(list(map(lambda x: [
            x['longitude'],
            x['latitude'],
            x['altitude'],
            x['slope'],
            # x['precipitation'],
            # x['evapotranspiration'],
            # x['temp_ave'],
            # x['temp_max'],
            # x['temp_min'],
        ], recs)))),
        'square_ids': torch.FloatTensor(np.array(list(map(lambda x: x['square_id'], recs)))),
    }

    return ret_dict


class LULCDataModuleOptim(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            test_imp: bool = False,
            data_dim: int = 6,
            train_labels_path: str = None,
            val_labels_path: str = None,
            test_labels_path: str = None,
            ancillary_path: str = None,
    ):
        super(LULCDataModuleOptim, self).__init__()

        dict_ancillary = {
            'longitude': [mean_long, std_long],
            'latitude': [mean_lat, std_lat],
            'altitude': [mean_altitude, std_altitude],
            'slope': [mean_slope, std_slope],
            # 'precipitation': [mean_precipitation, std_precipitation],
            # 'evapotranspiration': [mean_evapotranspiration, std_evapotranspiration],
            # 'temp_ave': [mean_temperature_ave, std_temperature_ave],
            # 'temp_max': [mean_temperature_max, std_temperature_max],
            # 'temp_min': [mean_temperature_min, std_temperature_min],
        }
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = StandardizeTS(
            mean=mean_modis,
            std=std_modis,
            dict_ancillary=dict_ancillary,
        )

        print(f'Transform: {self.transform}')

        self.test_imp = test_imp
        self.data_dim = data_dim

        if not os.path.exists(train_labels_path):
            raise ValueError(f'Training labels path does not exists: {train_labels_path}')
        else:
            self.train_labels_path = train_labels_path

        if not os.path.exists(val_labels_path):
            raise ValueError(f'Val labels path does not exists: {val_labels_path}')
        else:
            self.val_labels_path = val_labels_path

        if not os.path.exists(test_labels_path):
            raise ValueError(f'Test label path does not exists: {test_labels_path}')
        else:
            self.test_labels_path = test_labels_path

        if not os.path.exists(ancillary_path):
            raise ValueError(f'ancillary path does not exists: {ancillary_path}')
        else:
            self.ancillary_path = ancillary_path

        self.train_set = LULCOptim(
            self.data_dir,
            self.train_labels_path,
            self.transform,
            self.data_dim,
            ancillary_path=ancillary_path
        )

    def _get_val_size(self):
        return self.val_size / (1 - self.test_size)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.val_set = LULCOptim(self.data_dir, self.val_labels_path, self.transform, self.data_dim,
                                     ancillary_path=self.ancillary_path)

        if stage == 'test' or stage is None:
            self.test_set = LULCOptim(self.data_dir, self.test_labels_path, self.transform, self.data_dim,
                                      ancillary_path=self.ancillary_path)

    def train_dataloader(self):
        train_loader = data.DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return train_loader

    def val_dataloader(self):

        val_loader = data.DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return val_loader

    def test_dataloader(self):
        test_loader = data.DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return test_loader

    def predict_dataloader(self):
        predict_loader = data.DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return predict_loader

    def get_dicts(self):
        return self.train_set.get_dicts()


class LULCDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            train_val_test_split: tuple = (0.7, 0.1, 0.2),
            test_imp: bool = False,
            data_dim: int = 6,
            val_dir: str = None,
            test_dir: str = None
    ):
        super(LULCDataModule, self).__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = StandardizeTS(mean=mean_modis, std=std_modis)
        # self.transform = NormalizeTS(x_min=min_modis, x_max=max_modis, feature_range=(-1, 1))
        """self.transform = Compose([
            StandardizeTS(mean=mean_modis, std=std_modis),
            TasseledCap(),
        ])"""
        print(f'Transform: {self.transform}')
        self.train_size = train_val_test_split[0]
        self.val_size = train_val_test_split[1]
        self.test_size = train_val_test_split[2]

        assert self.train_size + self.val_size + self.test_size == 1

        self.test_imp = test_imp
        self.data_dim = data_dim

        self.full_set = LULC(
            directory=self.data_dir,
            transform=self.transform,
            data_dim=self.data_dim,
        )

        self.val_dir = None
        if val_dir is not None:
            assert self.val_size == 0, f'"val_dir" ({val_dir}) provided but validation split == {self.val_size}. ' \
                                       f'Validation split should be 0 in this case. '
            self.val_dir = val_dir

        self.test_dir = None
        if test_dir is not None:
            assert self.test_size == 0, f'"test_dir" ({test_dir}) provided but test split == {self.val_size}. Test ' \
                                        f'split should be 0 in this case. '
            self.test_dir = test_dir

    def _get_val_size(self):
        return self.val_size / (1 - self.test_size)

    def prepare_data(self) -> None:

        partitions_dir = os.path.join(self.data_dir, 'partitions')

        print(f'Length full set: {len(self.full_set)}')
        if os.path.exists(partitions_dir):
            print('Partitions found!!!')
            # self.train_idxs, self.test_idxs = get_global_idxs(self.data_dir, self.excluded_classes)
            print(f'Train idxs path: {os.path.join(partitions_dir, "train_idxs.npy")} ')
            self.train_idxs = np.load(os.path.join(partitions_dir, 'train_idxs.npy'))
            self.test_idxs = np.load(os.path.join(partitions_dir, 'test_idxs.npy'))
        else:
            indices = list(range(len(self.full_set)))

            if self.test_size:
                if self.train_size:
                    # Dataset for train and test
                    train_indices, test_indices, _, _ = train_test_split(
                        indices,
                        self.full_set.targets,
                        stratify=self.full_set.targets,
                        test_size=self.test_size,
                        random_state=1234
                    )

                    self.train_idxs, self.test_idxs = np.array(train_indices), np.array(test_indices)
                else:
                    # Dataset only for test
                    self.train_idxs, self.test_idxs = np.array([]), np.array(indices)
            else:
                self.train_idxs, self.test_idxs = np.array(indices), np.array([])

        print('Train_idxs:')
        print(self.train_idxs)
        print('----------------------')
        print('Test_idxs:')
        print(self.test_idxs)
        print('----------------------')

        print(f'Len full set: {len(self.full_set)}')
        print(
            f'Len train_idxs + self.test_idxs: {len(self.train_idxs)} + {len(self.test_idxs)} = {len(self.train_idxs) + len(self.test_idxs)}')
        unique, counts = np.unique(self.full_set.targets, return_counts=True)

        assert len(self.full_set) == len(self.train_idxs) + len(self.test_idxs)
        print(self.train_idxs[np.isin(self.train_idxs, self.test_idxs)])
        assert not (np.isin(self.train_idxs, self.test_idxs).any())  # Check that they do not have common elements

    def setup(self, stage: Optional[str] = None) -> None:
        # Create train set
        if self.train_size:
            self.train_set = LULCSubset(self.full_set, self.train_idxs)
            if self.val_size != 0:
                train_indices, val_indices, _, _ = train_test_split(
                    self.train_idxs,
                    self.train_set.targets,
                    stratify=self.train_set.targets,
                    test_size=self._get_val_size(),
                    random_state=1234
                )

                self.train_idxs = np.array(train_indices)
                self.val_idxs = np.array(val_indices)

                # Check that they do not have common elements
                assert not (np.isin(self.train_idxs, self.val_idxs).any())

                # Create training and validation datasets
                self.train_set = LULCSubset(self.full_set, self.train_idxs)
                self.val_set = LULCSubset(self.full_set, self.val_idxs, self.test_imp)
            elif self.val_dir is not None:
                self.val_set = LULC(
                    directory=self.val_dir,
                    transform=self.transform,
                    data_dim=self.data_dim,
                )

        # Create test set
        if self.test_size != 0:
            self.test_set = LULCSubset(self.full_set, self.test_idxs, self.test_imp)
        elif self.test_dir is not None:
            self.test_set = LULC(
                directory=self.test_dir,
                transform=self.transform,
                data_dim=self.data_dim,
            )

    def train_dataloader(self):
        train_loader = data.DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return train_loader

    def val_dataloader(self):

        val_loader = data.DataLoader(
            dataset=self.val_set,
            batch_size=len(self.val_set),
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return val_loader

    def test_dataloader(self):
        test_loader = data.DataLoader(
            dataset=self.test_set,
            batch_size=len(self.test_set),
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return test_loader

    def get_dicts(self):
        return self.full_set.get_dicts()
