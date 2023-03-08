import torch

from test.utils import get_model, load_original_clip, load_compression_freeze_text_cos_diff_clip_model, teacher_load, get_args
from clip import load


args = get_args()
args = load_original_clip(args)
args = load_compression_freeze_text_cos_diff_clip_model(args)
device = args.device
image_path = args.image_path
text_path = args.text_path
clip_path = args.clip_path
load_teacher = args.load_teacher
model = get_model(device, load_teacher, clip_path, image_path, text_path)

# args = get_model(device=args.device, load_teacher=True)
# model = load('ViT-B/32', device='cuda')
print('')
