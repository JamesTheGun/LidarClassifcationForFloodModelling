import pandas as pd
import json
import pdal
import re
import torch
import json
from ..common.constants import POSITIVE_CLASS_DIR, NEGATIVE_CLASS_DIR, POSITIVE_GEOTIFF_DIR, NEGATIVE_GEOTIFF_DIR

def two_dimensionify(las_data: pd.DataFrame) -> torch.Tensor:
    pass

def generate_geotiffs():
    with open("pdal_pipeline.json") as txt:
        pipeline_dict = json.load(txt)
        pipeline_dict = json.dumps(pipeline_dict)

    pipeline_dict_pos = re.sub(r"PLACEHOLDER_LAS", POSITIVE_CLASS_DIR, pipeline_dict)
    pipeline_dict_neg = re.sub(r"PLACEHOLDER_LAS", NEGATIVE_CLASS_DIR, pipeline_dict)
    pipeline_dict_pos = re.sub(r"PLACEHOLDER_OUT", POSITIVE_GEOTIFF_DIR, pipeline_dict)
    pipeline_dict_neg = re.sub(r"PLACEHOLDER_OUT", NEGATIVE_GEOTIFF_DIR, pipeline_dict)

    pos_cls = pdal.Pipeline(json.dumps(pipeline_dict_pos))
    neg_cls = pdal.Pipeline(json.dumps(pipeline_dict_neg))
    pos_cls.execute()
    neg_cls.execute()

def get_structured_tensor() -> torch.Tensor:
    pass