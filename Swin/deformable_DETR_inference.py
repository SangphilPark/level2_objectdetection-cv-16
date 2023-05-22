
import cv2
import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import torch

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile(
    './configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py')

root = '../../dataset/'

epoch = 'latest'

# dataset config 수정
cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json'  # test json 정보
# cfg.data.test.pipeline[1]['img_scale']= (512,512)
cfg.data.test.test_mode = True

cfg.data.samples_per_gpu = 4

cfg.seed = 2021
cfg.gpu_ids = [1]
cfg.work_dir = './work_dirs/deformable_detr_trash'

cfg.model.bbox_head.num_classes = 10

cfg.model.train_cfg = None
# build dataset & dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)

# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

model = build_detector(cfg.model, test_cfg=cfg.get(
    'test_cfg'))  # build detector
checkpoint = load_checkpoint(
    model, checkpoint_path, map_location='cpu')  # ckpt load

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])

# submission 양식에 맞게 output 후처리
prediction_strings = []
file_names = []
coco = COCO(cfg.data.test.ann_file)
img_ids = coco.getImgIds()
len(img_ids)


def output_visualization(output, test_dir, save_dir):
    cmap = [
        (000, 000, 255),
        (255, 000, 000),
        (000, 255, 000),
        (255, 153, 153),
        (000, 255, 255),
        (255, 255, 000),
        (102, 000, 153),
        (255, 000, 255),
        (188, 189, 34),
        (23, 190, 207)
    ]  # 클래스 별 색 지정

    num = 0
    class_num = 10
    # out -> 4772, 10, prediction num, 5 (xmin, x1, ymin, y1, score)
    for i, out in enumerate(output):
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        image_id = str(num).zfill(4)
        img = cv2.imread(test_dir + image_id + ".jpg")

        for j in range(class_num):
            for o in out[j]:
                xmin, ymin = o[0], o[2]
                xmax, ymax = o[1], o[3]
                w, h = xmax - xmin, ymax - ymin

                # text 구성요소
                text_content = classes[j]
                text_position = (int(xmin), int(ymin) + 20)
                text_font = cv2.FONT_HERSHEY_SIMPLEX
                text_scale = 0.8
                text_thickness = 2
                text_size, _ = cv2.getTextSize(
                    text_content, text_font, text_scale, text_thickness)

                # background 지정
                background_position = (int(xmin), int(ymin))
                background_size = (text_size[0], text_size[1]+10)
                background_color = cmap[j]

                img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(
                    xmax), int(ymax)), cmap[j], thickness=text_thickness)
                cv2.rectangle(img, background_position, (
                    background_position[0] + background_size[0], background_position[1] + background_size[1]), background_color, -1)
                cv2.putText(img=img, text=text_content, org=text_position, fontFace=text_font,
                            fontScale=text_scale, color=(255, 255, 255), thickness=text_thickness)

        file_path = save_dir + image_id + ".jpg"  # save 주소 지정
        cv2.imwrite(file_path, img)
        num += 1


test_dir = "../../dataset/test/"
save_dir = "../../dataset/test_visual_deformable_DETR/"
output_visualization(output, test_dir, save_dir)
class_num = 10
# out -> 4772, 10, prediction num, 5 (x, x1, y, y1, score)
for i, out in enumerate(output):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    for j in range(class_num):
        for o in out[j]:
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                o[2]) + ' ' + str(o[3]) + ' '

    prediction_strings.append(prediction_string)
    file_names.append(image_info['file_name'])


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(
    cfg.work_dir, f'submission_{epoch}.csv'), index=None)
submission.head()
