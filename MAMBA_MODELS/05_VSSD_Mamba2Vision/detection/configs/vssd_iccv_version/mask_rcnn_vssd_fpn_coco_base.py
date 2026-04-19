_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MM_VSSD',
        out_indices=(0, 1, 2, 3),
        #pretrained="/public/home/ac5a74vrda/vssd/exclude/vssd_base_e300_with_mesa.pth",
        pretrained="/scratch/pg44/md0612/yh/vssd/exclude/vssd_base_e300_with_mesa.pth",
        embed_dim=96,
        depths=(3, 4, 21, 5),
        num_heads=(3, 6, 12, 24),
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=2,
        ssd_chunk_size=256,
        linear_attn_duality=True,
        attn_types = ['mamba2', 'mamba2', 'mamba2', 'standard'],
        bidirection = False,
        drop_path_rate = 0.6,
        use_cpe = True,
        ssd_positve_dA = True,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
)

checkpoint_config = dict(max_keep_ckpts=2)


default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', max_keep_ckpts=2)
)

# train_dataloader = dict(
#     batch_size=1,
# )