from model.component.weight_share_model import RepeatVisionTransformer, RepeatTextTransformer


def mini_vision_encoder():
    visual_encoder = RepeatVisionTransformer(
        img_size=224, patch_size=32, in_chans=3, out_dim=512, embed_dim=768, depth=6, num_heads=24, mlp_ratio=4.0,
        qkv_bias=True, repeated_times=2, use_transform=True
    )
    return visual_encoder



def mini_text_encoder():
    text_encoder = RepeatTextTransformer(
        depth=4, repeated_times=2,
        use_transform=True
    )
    return text_encoder



def mini_compression_text_encoder():
    text_encoder = RepeatTextTransformer(
        depth=4, repeated_times=2,
        use_transform=True,
        compression_embedding=True,
        embedding_compression_dim=256
    )
    return text_encoder
