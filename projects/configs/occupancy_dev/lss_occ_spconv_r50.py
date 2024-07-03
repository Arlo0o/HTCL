_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
img_norm_cfg = None

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 0.25]

occ_size = [(point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0], 
            (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1],
            (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]]
occ_size = [int(x) for x in occ_size]

# voxel_size for LSS: [256, 256, 16]
lss_downsample = [2, 2, 2]
voxel_size_lss = [voxel_size[0] * lss_downsample[0], voxel_size[1] * lss_downsample[1], voxel_size[2] * lss_downsample[2]]
sparse_shape = [occ_size_i // down_factor for occ_size_i, down_factor in zip(occ_size, lss_downsample)]

# downsample ratio in [x, y, z] when generating 3D volumes in LSS
data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    # image-view augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# [256, 256, 16]
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_size_lss[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_size_lss[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_size_lss[2]],
    'dbound': [2.0, 58.0, 0.5],
}
numC_Trans = 80

model = dict(
    type='BEVDepthOccupancy',
    img_backbone=dict(
        pretrained='/mnt/cfs/algorithm/yunpeng.zhang/pretrained/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootSpconv',
        loss_depth_weight=1.,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False,
        # Simple Voxel Layers
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size_lss,
            max_voxels=200000),
        voxel_encoder=dict(type='HardSimpleVFE', num_features=(numC_Trans + 3)),
        middle_encoder=dict(
            type='CustomSparseUNet',
            in_channels=(numC_Trans + 3),
            sparse_shape=sparse_shape[::-1],
            norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
            base_channels=64,
            output_channels=numC_Trans,
            # 3个 stage 会做 4x 下采样
            encoder_channels=((64, 64, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1)),
            decoder_channels=((128, 128, 128), (128, 128, 64), (64, 64, 64)),
            decoder_paddings=((1, 0), (1, 0), (1, 1)),
        ),
    ),
    # after spconv, empty voxels still have no features. therefore, conv3d is needed to propagate features globally
    img_bev_encoder_backbone=None,
    # img_bev_encoder_backbone=dict(
    #     type='CustomResNet3D',
    #     depth=10,
    #     n_input_channels=numC_Trans,
    #     block_inplanes=voxel_channels,
    #     norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
    # ),
    img_bev_encoder_neck=dict(
        type='SpconvNeck3D',
        in_channels=[128, 128, 64],
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
        out_channels=256,
    ),
    pts_bbox_head=dict(
        type='OccHead',
        in_channels=[256, 256, 256],
        out_channel=17,
        out_point_channel=17,
        loss_voxel_prototype='cylinder3d',
        supervise_points=True,
        point_cloud_range=point_cloud_range,
        num_level=3,
        loss_weight_cfg=dict(
            loss_voxel_weight=1.0,
            loss_point_weight=10.0,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(pts=None),
    test_cfg=dict(pts=None),
)

dataset_type = 'CustomNuScenesOccLSSDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=data_config,
                sequential=False, aligned=True, trans_only=False, 
                mmlabnorm=True, load_depth=True, img_norm_cfg=img_norm_cfg),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        speed_mode='abs_speed'),
    # how to handle bda
    dict(type='LoadSemanticPoint'),
    dict(type='CreateVoxelLabels', point_cloud_range=point_cloud_range, grid_size=occ_size, unoccupied=0),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points_occ']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config,
         sequential=False, aligned=True, trans_only=False, mmlabnorm=True, img_norm_cfg=img_norm_cfg),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        speed_mode='abs_speed',
        is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True),
    dict(type='LoadMesh', to_float32=True, load_semantic=True),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points_occ', 'gt_semantic'],
            meta_keys=['pc_range', 'occ_size']),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='/mnt/cfs/algorithm/yunpeng.zhang/public_data/occupancy_infos/nuscenes_infos_temporal_val.pkl',
    bevdepth_ann_file='/mnt/cfs/algorithm/junjie.huang/data/bevdet-nuscenes_infos_val.pkl',
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    queue_length=1,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        bevdepth_ann_file='/mnt/cfs/algorithm/junjie.huang/data/bevdet-nuscenes_infos_train.pkl',
        ann_file='/mnt/cfs/algorithm/yunpeng.zhang/public_data/occupancy_infos/nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        queue_length=1,
        use_semantic=True,
        box_type_3d='LiDAR'),
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=1, pipeline=test_pipeline)