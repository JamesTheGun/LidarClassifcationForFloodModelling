from typing import List, Tuple
from dataclasses import dataclass
import subprocess

import pandas as pd
import torch
import torch.nn.functional as F

from structured_data_utils.config.constants import ESPSG, GEOTIFF_LOCATIONS_TO_CORRESPONDING_STANDARDISED_LOCATION, RES, EMPTY_VAL
from structured_data_utils.structuring import get_negative_geotiff_tensor_and_offset, get_positive_geotiff_tensor_and_offset, get_combined_geotiff_tensor_and_offset
from structured_data_utils.structured_data_interfacing import get_segments_with_sliding_window, remove_empty_segments

def standardise_geotiff(target_dir: str, write_dir: str):
    subprocess.run([
        "gdalwarp",
        "-t_srs", ESPSG,
        "-tr", RES, RES,
        "-r", "bilinear",
        "-overwrite",
        target_dir,
        write_dir,
    ], check=False,
    capture_output=True,
    text=True)

def standardise_core_geotiffs():
    for target, write_dir in GEOTIFF_LOCATIONS_TO_CORRESPONDING_STANDARDISED_LOCATION.items():
        standardise_geotiff(target, write_dir)

def offset_meters_to_offset_pixels(offset):
    print(offset[0]/float(RES))
    offset_x_corrected = int(offset[0]/float(RES))
    offset_y_corrected = int(offset[1]/float(RES))
    return (offset_x_corrected, offset_y_corrected)

def pad_pos_mask_to_match(pos_tensor: torch.Tensor, other_tensor: torch.Tensor, offset: Tuple[int, int]):
    pad_x = other_tensor.shape[-1] - pos_tensor.shape[-1]
    pad_y = other_tensor.shape[-2] - pos_tensor.shape[-2]

    padded = F.pad(pos_tensor, (0, pad_x, 0, pad_y), mode="constant", value=EMPTY_VAL)
    pixel_offset = offset_meters_to_offset_pixels(offset)
    #padded = torch.roll(padded, shifts = pixel_offset, dims=(-2,-1))
    padded = torch.roll(padded, shifts = pixel_offset[0], dims=1)
    padded = torch.roll(padded, shifts = pixel_offset[1], dims=0)
    return padded

def load_combined_pos_neg_df_structured() -> DataWithLabels:
    positive, offset_positive = get_positive_geotiff_tensor_and_offset()
    combined, offset_combined = get_combined_geotiff_tensor_and_offset()

    offset = (
        offset_positive[0] - offset_combined[0],
        offset_combined[1] - offset_positive[1],
    )

    positive = pad_pos_mask_to_match(positive, combined, offset)
    positive = positive.unsqueeze(0)
    combined = combined.unsqueeze(0)

    labels = torch.ones_like(combined[:1])

    pos_mask = positive[0] != EMPTY_VAL

    labels[0, pos_mask] = 0

    data = combined.clone()
    data[0, pos_mask] = positive[0, pos_mask]

    return DataWithLabels(data[0], labels[0])

class DataWithLabels:
    data: torch.Tensor
    labels: torch.Tensor
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        assert data.shape == labels.shape, "labels' shape do not match the given dataset's shape"
        self.data = data
        self.test = labels

@dataclass
class ModelData:
    """prepare and store data with methods for retreiving train/test split"""
    data: torch.Tensor = None
    labels: torch.Tensor = None
    train_set: torch.Tensor = None
    test_set: torch.Tensor = None
    hyper_params: dict = None
    folds_train: List[torch.Tensor] = None
    folds_test: List[torch.Tensor] = None
    folds: list[torch.Tensor] = None
    fold_index: int = 0

    def prepare_data(self):
        self.data, self.labels = load_combined_pos_neg_df_structured()

    def set_hyper_params(self, HYPER_PARAMETERS):
        self.hyper_params = HYPER_PARAMETERS

    def prepare_train_test(self, train_per_holdout: float = None):
        self.train_set, self.test_set = self._get_train_test_tensors(
            self.data, train_per_holdout
        )

    def prepare_folds(train_test_ratio_fold):
        pass

    def new_split(self, hyper_params: dict, train_to_hold_out_ratio: int = None, train_test_ratio_fold: int = None):
        if (self.data == None):
            self.prepare_data()
        self.set_hyper_params(hyper_params)
        self.prepare_train_test(train_to_hold_out_ratio)
        self.prepare_folds(train_test_ratio_fold)

    @classmethod
    def _get_train_test_tensors(cls, data_with_labels: DataWithLabels, train_percent: float) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return get_segments_with_sliding_window(data_with_labels)

    @staticmethod
    def _get_folds(data: torch.Tensor, train_per_holdout: int) -> List[torch.Tensor]:
        pass

    def next_fold(self) -> torch.Tensor | None:
        pass

    def get_folds(self) -> List[torch.Tensor]:
        pass

    def get_data_from_fold(self, fold_index: int):
        pass
