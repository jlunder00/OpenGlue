import glob
import logging
import math
import pathlib
from pathlib import Path
from typing import Tuple, List, Union, Optional

import cv2
import deepdish as dd
import numpy as np
import os
import torch
import yaml
from torch import multiprocessing

from models.features import get_feature_extractor

import extract_features

#from extract_features.py for jupyter notebook feature extraction
def process_chunk(process_id: int, images_list: List[Tuple[str, Union[str, None]]], feature_extractor_config: dict,
                  output_path: Union[str, pathlib.Path], device: str, recompute: bool, target_size: List[int]):
    """Function to execute on each worker"""
    '''
        torch.multiprocessing wont start the function process_chunk in the interactive environment unless the function is defined
        in another file and imported. Since I needed to make changes to process_chunk to use it without argparse in the interactive
        environment, I had to redefine it here
    '''
    if device == 'cuda':
        device = f'cuda:{process_id}'
    else:
        device = 'cpu'

    cv2.setNumThreads(1)
    logger = logging.getLogger(__name__)

    features_name = feature_extractor_config['name']
    feature_extractor = get_feature_extractor(features_name)(**feature_extractor_config['parameters'])
    feature_extractor.eval().to(device)

    with torch.inference_mode():
        images_list = images_list[process_id]
        for i, (image_path, scene) in enumerate(images_list, start=1):
            output_path_scene = output_path
            if scene is not None:
                output_path_scene = output_path_scene / scene

            os.makedirs(output_path_scene, exist_ok=True)

            base_name = image_path.rpartition(os.path.sep)[2].rpartition('.')[0]
            # skip image if output already exists and recompute=False
            if not recompute and extract_features.check_if_features_exist(output_path_scene, base_name):
                continue

            image = extract_features.read_image(image_path, target_size)
            image = (torch.FloatTensor(image) / 255.).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
            _, _, resize_height, resize_width = image.size()
            lafs, scores, descriptors = map(lambda x: x[0].cpu().numpy(), feature_extractor(image))

            # save results
            extract_features.save_outputs(
                output_path_scene,
                base_name,
                (lafs, scores, descriptors, np.array([resize_width, resize_height]))
            )

            if i % 100 == 0:
                logger.info(f'PID #{process_id}: Processed {i}/{len(images_list)} images.')