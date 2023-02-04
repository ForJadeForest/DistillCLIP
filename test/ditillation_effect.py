import pathlib
import torch

from clip_score import get_clip_score
from typing import List, Dict, Tuple
import os


def load_model(model_path):
    return torch.load(model_path)


def load_text(file_path: str) -> Dict:
    """
    return the dict, the key is image id, the value is text candidates
    """
    pass


def load_image(image_dir: str) -> Tuple:
    """
    image_dir: the dir of images

    return the image path of List
    """
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    image_ids = [pathlib.Path(path).stem for path in image_paths]
    return image_paths, image_ids


model_path = ''
text_path = ''
image_dir = ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(model_path)
texts, images = load_text(text_path), load_image(image_dir)
get_clip_score(model, images, texts, device=device)