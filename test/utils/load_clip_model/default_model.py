import dataclasses
from typing import Optional


@dataclasses.dataclass
class CLIPModelLoadConfig:
    image_path: Optional[str] = None
    text_path: Optional[str] = None
    clip_path: Optional[str] = None
    load_teacher: bool = None


def load_md_single_clip_model():
    return CLIPModelLoadConfig(
        image_path='/data/share/pyz/Dis_CLIP/final/image/wp_single/188-val_acc0.237-loss0.13170.ckpt',
        text_path='/data/share/pyz/Dis_CLIP/final/text/wp_md_single/135-val_acc0.299-loss0.04125.ckpt',
        clip_path=None,
        load_teacher=False
    )


def load_ce_loss_clip_model():
    return CLIPModelLoadConfig(
        image_path=None,
        text_path=None,
        clip_path='/data/share/pyz/Dis_CLIP/final/clip/ce_clip/checkpoints/290-val_acc0.242-loss0.28316.ckpt',
        load_teacher=False
    )


def load_sr_clip_model():
    return CLIPModelLoadConfig(
        image_path=None,
        text_path=None,
        clip_path='/data/share/pyz/Dis_CLIP/final/clip/wp_sr/117-val_acc0.249-loss0.05607.ckpt',
        load_teacher=False
    )


def load_l_clip_model():
    return CLIPModelLoadConfig(
        image_path=None,
        text_path=None,
        clip_path='/data/share/pyz/Dis_CLIP/final/clip/l_clip/253-val_acc0.245-loss0.05582-v1.ckpt',
        load_teacher=False
    )


def load_smd_clip_model():
    return CLIPModelLoadConfig(
        image_path='/data/share/pyz/Dis_CLIP/final/image/ws_best/234-val_acc0.262-loss0.11146.ckpt',
        text_path='/data/share/pyz/Dis_CLIP/final/text/ws_best/225-val_acc0.301-loss0.03477.ckpt',
        clip_path=None,
        load_teacher=False
    )


def load_single_clip_model():
    return CLIPModelLoadConfig(
        image_path='/data/share/pyz/Dis_CLIP/final/image/wp_single/188-val_acc0.237-loss0.13170.ckpt',
        text_path='/data/share/pyz/Dis_CLIP/final/text/wp_single/163-val_acc0.296-loss0.03877.ckpt',
        clip_path=None,
        load_teacher=False
    )


def load_original_clip():
    return CLIPModelLoadConfig(
        load_teacher=True
    )


model_name_map = {
    'WP+MD&Single': load_md_single_clip_model(),
    'Teacher': load_original_clip(),
    'WP&Single': load_single_clip_model(),
    'smd': load_smd_clip_model(),
    'L-CLIP': load_l_clip_model(),
    'WP&SR': load_sr_clip_model(),
    'CE-CLIP': load_ce_loss_clip_model(),
}
