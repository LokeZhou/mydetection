from mmdet.apis import init_detector,inference_detector,show_result


config_file = 'configs/mask_rcnn_x101_32x4d_fpn_1x.py'
checkpoint_file = 'checkpoints/mask_rcnn_x101_32x4d_fpn_2x_20181218-f023dffa.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file)

# 测试一张图片
img = '/home/loke/samples/train/left/1_57.png'


result = inference_detector(model, img)
show_result(img, result, model.CLASSES)

