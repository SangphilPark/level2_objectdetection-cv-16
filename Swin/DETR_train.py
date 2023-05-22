# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import torch

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./configs/detr/detr_r50_8x2_150e_coco.py')

root = '../../dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + \
    'random_split/train_0.2_2020_1.json'  # train json 정보

cfg.data.val.classes = classes
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = root + 'random_split/val_0.2_2020_1.json'  # train json 정보


cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json'  # test json 정보

# epoch 조정
cfg.runner.max_epochs = 40

cfg.data.samples_per_gpu = 4

cfg.seed = 2022
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/detr_r50_8x2_150e_trash'

cfg.load_from = './detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

cfg.model.bbox_head.num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()

cfg.evaluation = dict(metric='bbox', save_best='bbox_mAP')

# wandb hyperparameter(마지막에 위치해야함)
for h in cfg.log_config.hooks:
    if h['type'] == 'WandbLoggerHook':
        h['init_kwargs']['config'] = cfg._cfg_dict.to_dict()

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)

train_detector(model, datasets[0], cfg, distributed=False, validate=True)
