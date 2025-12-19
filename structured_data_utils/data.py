from typing import List, Tuple
from dataclasses import dataclass
import pandas as pd
import torch
import subprocess
from structured_data_utils.config.constants import ESPSG, GEOTIFF_LOCATIONS_TO_CORRESPONDING_STANDARDISED_LOCATION, RES
from structured_data_utils.structuring import get_negative_geotiff_tensor_and_offset, get_positive_geotiff_tensor_and_offset, get_combined_geotiff_tensor_and_offset

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

def load_combined_pos_neg_df_structured() -> pd.DataFrame:
    positive = get_positive_geotiff_tensor_and_offset()
    combined = get_combined_geotiff_tensor_and_offset()
    label = torch.full_like(combined[:1], 0)
    combined = torch.cat([label,combined], dim=0)

    base_tensor = torch.tensor((0,*combined.shape[1:]), dtype=combined.dtype, device=combined.device)
    all_negative: torch.Tensor = base_tensor.new_zeros((0,*combined.shape[1:]))
    all_positive: torch.Tensor = base_tensor.new_ones((0,*combined.shape[1:]))
    combined = torch.cat([combined, all_negative], dim=0)
    positive = torch.cat([positive, all_positive], dim=0)
    mask = positive != 0
    combined[mask] = positive[mask]
    
@dataclass
class modelData:
    """prepare and store data with methods for retreiving train/test split"""
    data: pd.DataFrame = None
    train_set: torch.Tensor = None
    test_set: torch.Tensor = None
    hyper_params: dict = None
    folds_train: List[torch.Tensor] = None
    folds_test: List[torch.Tensor] = None
    folds: list[torch.Tensor] = None
    fold_index: int = 0

    def prepare_data(self):
        self.data = load_combined_pos_neg_df_structured()

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
    def _get_train_test_tensors(cls, data: pd.DataFrame, train_per_holdout: int):
        las_tensor = build_lidar_tensor(data)
        return cls._get_train_test_sets(las_tensor, train_per_holdout)

    @staticmethod
    def _get_folds(data: torch.Tensor, train_per_holdout: int) -> List[torch.Tensor]:
        pass
    
    @staticmethod
    def _get_train_test_sets(data: torch.Tensor, test_per_train: int = 7) -> Tuple[torch.Tensor]:
        pass

    def next_fold(self) -> torch.Tensor | None:
        if self.fold_index > len(self.folds):
            return False
        fold = self.folds[self.fold_index]
        self.fold_index+=1
        return fold

    def get_folds(self) -> List[torch.Tensor]:
        return self.folds
    
    def get_data_from_fold(self, fold_index: int):
        return self.test_set[self.folds[fold_index]]
