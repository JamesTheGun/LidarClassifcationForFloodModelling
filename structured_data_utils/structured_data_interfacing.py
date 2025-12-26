
import torch
from typing import List, Tuple
from structured_data_utils.config.constants import EMPTY_VAL
from structured_data_utils.data import DataWithLabels

def get_segments_with_sliding_window(data_with_labels: DataWithLabels, window_size=400, stride=200) -> DataWithLabels:
    patches = (
        data_with_labels.data.unfold(0, window_size, stride)
         .unfold(1, window_size, stride)
    )
    patch_labels = (
        data_with_labels.labels.unfold(0, window_size, stride)
         .unfold(1, window_size, stride)
    )
    patches = patches.contiguous().view(-1, window_size, window_size)
    patch_labels = patch_labels.contiguous().view(-1, window_size, window_size)
    data_with_labels_out = DataWithLabels(patches, patch_labels)
    return data_with_labels_out

def remove_empty_segments(data_with_labels: DataWithLabels) -> DataWithLabels:
    not_empty = data_with_labels.data != EMPTY_VAL
    mean_occupied = not_empty.float().mean(dim=(1,2))
    mask = mean_occupied > 0.5
    data_with_labels = DataWithLabels(data_with_labels.data[mask], data_with_labels.labels[mask])
    return data_with_labels

def generate_train_test_sets(labeled_tensor: torch.Tensor):
    pass

def generate_folds(labeled_tensor: torch.Tensor) -> List[torch.Tensor]:
    pass