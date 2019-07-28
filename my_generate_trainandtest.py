import os
import numpy as np


def id2address(x, savepath):
    return savepath % x + '/n'


def trainandtestdata(root, image_set):

    dataset_name = 'VOC2007'
    imgsetpath = os.path.join(root, dataset_name, 'ImageSets', 'Main', '%s.txt')

    saveimagepath = './data/custom/images/%s.jpg'
    if image_set=='val':
        savepath = './data/custom/%s.txt' % 'valid'
    else:
        savepath = './data/custom/%s.txt' % image_set


    image_address = []

    with open(imgsetpath % image_set) as f:  # read the id  ???one time one line????
        image_id = f.readlines()
        image_id = [x.strip('\n') for x in image_id]  # remove \n

    for i in range(0, len(image_id)):
        image_address.append(saveimagepath % image_id[i] + '\n')

    with open(savepath, 'w') as f2:
        f2.writelines(image_address)


if __name__ == '__main__':
    root = '/home/ziqixia/Dataset/Pascal_VOC/2007/VOCdevkit'
    for image_set in ['train', 'val', 'test']:
        trainandtestdata(root, image_set)
