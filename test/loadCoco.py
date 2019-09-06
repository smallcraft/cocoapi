#!/usr/bin/env python
# coding=utf-8
from pycocotools.coco import COCO
import numpy as np
annfile='/home/dataset/coco/coco2017/annotations/instances_train2017.json'
catNames=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

catSuperNames=['appliance', 'food', 'indoor', 'accessory', 'electronic', 'furniture', 'vehicle', 'sports', 'animal', 'kitchen', 'person', 'outdoor']

coco = COCO(annfile)

### go through all category id and names 
catIds = coco.getCatIds(catNms=catNames)
for index in range(len(catIds)):
    print('%d %s'%(catIds[index],catNames[index]))

### get one image info 
catIds = coco.getCatIds(catNms='dog')
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
print(img)

### load Ann info
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)


