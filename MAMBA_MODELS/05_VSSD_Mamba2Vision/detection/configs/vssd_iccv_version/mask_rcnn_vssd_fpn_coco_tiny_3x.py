_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
]

model = dict(
    backbone=dict(
        type='MM_VSSD',
        out_indices=(0, 1, 2, 3),
        pretrained="/public/home/ac5a74vrda/vssd/exclude/vssd_tiny_mesa.pth",
        embed_dim=96,
        depths=(2, 2, 8, 2),
        num_heads=(4, 4, 8, 16),
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=1,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],
        bidirection=False,
        drop_path_rate=0.2,
        async_state=[12, 24, 48, 64],
        mlp_ratio=3.0,
        rmt_downsample=True,
        rmt_patch_embed=True,
        use_cpe=True,
        ssd_linear_norm=True,
        exp_da=True,
        rope=True,
        ssd_positve_dA=True,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
)

default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4)
)
