model = dict(
    type='EncoderDecoder',  # Encoder-Decoder模型架构
    backbone=dict(
        type='ResNet',  # 使用ResNet作为特征提取骨干网络
        depth=50,  # ResNet-50
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'
    ),
    decode_head=dict(
        type='DeepLabV3Head',  # 使用DeepLabV3的解码头
        in_channels=2048,  # 输入通道数
        channels=512,  # 输出通道数
        num_classes=21,  # 类别数，根据你的数据集修改
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False
    ),
    auxiliary_head=dict(
        type='FCNHead',  # 辅助头用于辅助训练
        in_channels=1024,
        channels=256,
        num_classes=21,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
