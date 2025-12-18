import pandas as pd
import json
import pdal
import re
import torch
import json
import os
import glob
from common.constants import POSITIVE_CLASS_DIR, NEGATIVE_CLASS_DIR, POSITIVE_GEOTIFF_DIR, NEGATIVE_GEOTIFF_DIR
import rasterio

class GeotiffGeneration():
    @staticmethod
    def _load_pipeline_template(template_path: str) -> str:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _list_las_files(input_dir: str) -> list[str]:
        return sorted(
            glob.glob(os.path.join(input_dir, "*.las")) +
            glob.glob(os.path.join(input_dir, "*.laz"))
        )

    @staticmethod
    def _build_pipeline_spec(template_text: str, las_path: str, out_path: str) -> str:
        return (
            template_text
            .replace("PLACEHOLDER_LAS", las_path)
            .replace("PLACEHOLDER_OUT", out_path)
        )

    @staticmethod
    def _run_pdal_pipeline(pipeline_spec_text: str) -> None:
        pdal.Pipeline(pipeline_spec_text).execute()

    @classmethod
    def _generate_geotiffs_for_dir(
        cls, 
        template_text: str,
        input_dir: str,
        output_dir: str
    ) -> None:
        spec = cls._build_pipeline_spec(template_text, input_dir, output_dir)
        cls._run_pdal_pipeline(spec)

    @classmethod
    def generate_geotiffs(cls) -> None:
        template_text = cls._load_pipeline_template(r"misc\pdal_pipeline.json")

        cls._generate_geotiffs_for_dir(
            template_text,
            POSITIVE_CLASS_DIR,
            POSITIVE_GEOTIFF_DIR,
        )

        cls._generate_geotiffs_for_dir(
            template_text,
            NEGATIVE_CLASS_DIR,
            NEGATIVE_GEOTIFF_DIR,
        )

def two_dimensionify(las_data: pd.DataFrame) -> torch.Tensor:
    pass

def tensor_from_geotiff(geotiff_dir: str) -> torch.Tensor:
    with rasterio.open(geotiff_dir) as target:
        image_data = target.read()
    as_tensor = torch.from_numpy(image_data)
    return as_tensor

def get_negative_geotiff_tensor():
    return tensor_from_geotiff(NEGATIVE_GEOTIFF_DIR)

def get_positive_geotiff_tensor():
    return tensor_from_geotiff(POSITIVE_GEOTIFF_DIR)