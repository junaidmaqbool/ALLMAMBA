_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MM_VSSD',
        out_indices=(0, 1, 2, 3),
        pretrained="/public/home/ac5a74vrda/vssd/exclude/vssd_small_mesa.pth",
        embed_dim=96,
        depths=(2, 4, 15, 4),
        num_heads=(4, 4, 8, 16),
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=1,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        attn_types = ['mamba2', 'mamba2', 'mamba2', 'standard'],
        bidirection = False,
        drop_path_rate = 0.4,
        async_state= [12, 24, 48, 64],
        mlp_ratio = 3.0,
        rmt_downsample = True,
        rmt_patch_embed = True,
        use_cpe = True,
        ssd_linear_norm = True,
        exp_da = True,
        rope = True,
        ssd_positve_dA = True,
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150, act_cfg=dict(type='ReLU', inplace=False)),
    auxiliary_head=dict(in_channels=384, num_classes=150, act_cfg=dict(type='ReLU', inplace=False), )
    # neck=dict(in_channels=[96, 192, 384, 768]),
)

default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', max_keep_ckpts=2)
)

train_dataloader = dict(
    batch_size=4,
)
# Flops: 1.021T
# Params: 81.147M