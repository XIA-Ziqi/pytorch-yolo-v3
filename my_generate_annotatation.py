import torch
from skimage.transform import resize
import os
import os.path
import copy
import sys
import glob
from tqdm import tqdm
import time

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def target_transform(target, classes):
    res = []

    for obj in target.iter('object'):
        tmp = []
        # difficult = int(obj.find('difficult').text) == 1
        # if not self.keep_difficult and difficult:
        #    continue
        name = obj.find('name').text
        name = classes.index(name)
        # name = obj[0].text.lower().strip()
        # print(type(name))
        bb = obj.find('bndbox')
        # bbox = obj[1]
        # print(bbox)
        bndbox = [bb.find('xmin').text, bb.find('ymin').text, bb.find('xmax').text, bb.find('ymax').text]
        # supposes the order is xmin, ymin, xmax, ymax

        tmp.append(copy.deepcopy(int(name)))
        tmp.append(copy.deepcopy(float(bndbox[0])))
        tmp.append(copy.deepcopy(float(bndbox[1])))
        tmp.append(copy.deepcopy(float(bndbox[2])))
        tmp.append(copy.deepcopy(float(bndbox[3])))
        res.append(copy.deepcopy(tmp))
    return res


def initdata(root, ids, classes, image_nums, target_transform=None, img_size=416):
    root = root
    dataset_name = 'VOC2007'
    annopath = os.path.join(root, dataset_name, 'Annotations',
                            '%s.xml')  # /home/ziqixia/Dataset/Pascal_VOC/2007/VOCdevkit/VOC2007/Annotations/%s.xml
    imgpath = os.path.join(root, dataset_name, 'JPEGImages', '%s.jpg')


    max_objects = 100
    self_img_shape = (img_size, img_size)  ##but not 416

    for index in tqdm(range(image_nums)):
        img_id = ids[index]
        target = ET.parse(annopath % img_id).getroot()  # 获取根节点
        img = np.array(Image.open(imgpath % img_id).convert('RGB'))
        # print(self._imgpath % img_id)

        h, w, _ = img.shape

        if target_transform is not None:
            target = target_transform(target, classes)
        labels = None

        if len(target) > 0:
            target = np.array(target)
            x1 = target[:, 1]
            y1 = target[:, 2]
            x2 = target[:, 3]
            y2 = target[:, 4]
            # transform the pascal form label to coco form
            labels = np.zeros((target.shape))
            labels[:, 0] = target[:, 0]
            labels[:, 1] = (x1 + x2) / (2 * w)
            labels[:, 2] = (y1 + y2) / (2 * h)
            labels[:, 3] = (x2 - x1) / w
            labels[:, 4] = (y2 - y1) / h

        filled_labels = np.zeros((len(labels), 5))
        if labels is not None:
            filled_labels[range(len(labels))[:max_objects]] = labels[:max_objects]
        else:
            print('no object')

        np.savetxt("./data/custom/labels/%s.txt" % img_id, filled_labels, fmt='%d %f %f %f %f')


if __name__ == '__main__':
    # root:directory to dataset
    root = '/home/ziqixia/Dataset/Pascal_VOC/2007/VOCdevkit'
    classes = ['background','person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', \
               'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', \
               'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    image_nums = len(glob.glob('./data/custom/images/*.jpg'))
    ids = list(range(1, image_nums + 1))
    ids = list(map(lambda i: str(i).zfill(6), ids))
    image_set = 'train'  # train,val,test
    initdata(root, ids, classes,image_nums, target_transform=target_transform, img_size=416)
    time.sleep(0.5)
