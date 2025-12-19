
import torch
from typing import List

def get_segments_with_sliding_window(x, window_size=200, stride=100):
    patches = x.unfold(0, window_size, window_size).unfold(1, window_size, window_size)
    # shape: (nH, nW, tile_h, tile_w)

    # flatten tiles to a batch: (nH*nW, tile_h, tile_w)
    patches = patches.contiguous().view(-1, window_size, window_size)
    return patches

def generate_train_test_sets(labeled_tensor: torch.Tensor):
    pass

def generate_folds(labeled_tensor: torch.Tensor) -> List[torch.Tensor]:
    pass