#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-07-19 10:29:34
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode

# NEW SILIC
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"
# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)
reso = 0.4 # arcsec/pix
# END NEW SILIC

INPUT_SIZE   = 512
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)

# predicted_dir_path = 'mAP/predicted'
# ground_truth_dir_path = 'mAP/ground-truth'
predicted_dir_path = 'mAP/predicted_fake'
ground_truth_dir_path = 'mAP/ground-truth_fake'
if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

os.mkdir(predicted_dir_path)
os.mkdir(ground_truth_dir_path)
os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

# Build Model
input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights("./yolov3")

with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
    for num, line in enumerate(annotation_file):
        annotation = line.strip().split()
        image_path = annotation[0]
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

        if len(bbox_data_gt) == 0:
            bboxes_gt=[]
            classes_gt=[]
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

        print('=> ground truth of %s:' % image_name)
        num_bbox_gt = len(bboxes_gt)
        with open(ground_truth_path, 'w') as f:
            for i in range(num_bbox_gt):
                class_name = CLASSES[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
        print('=> predict result of %s:' % image_name)
        predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
        # Predict Process
        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')


        if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
            '''
            # NEW SILIC
            # Draw truth
            tmp_bboxes = [[1,1,1,1,1,1]]
            tmp_bboxes[0][:4] = bboxes_gt[0]
            image = utils.draw_bbox(image, tmp_bboxes, classes=['cluster','truth'])
            image = utils.draw_bbox(image, bboxes)
            # Draw mem gal
            clus_id = int(image_name.split('.')[0])
            g1 = np.where(clu_full_data['ID'] == clus_id)[0]
            g2 = np.where(mem_full_data['ID'] == clus_id)[0]
            c = SkyCoord(
                ra=clu_full_data['RA'][g1]*u.degree,
                dec=clu_full_data['DEC'][g1]*u.degree,
                frame='icrs',
            )
            c2 = SkyCoord(
                ra=mem_full_data['RA'][g2]*u.degree,
                dec=mem_full_data['DEC'][g2]*u.degree,
                frame='icrs',
            )
            off = c.spherical_offsets_to(c2)
            xs = np.round(1024 - off[0].arcsec / reso * 1.01).astype('int')
            ys = np.round(1024 - off[1].arcsec / reso * 1.01).astype('int')
            for x, y in zip(xs, ys):
                image = cv2.circle(image, (x,y), 5, (0,0,255),3)
            # END NEW SILIC
            '''
            cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH+image_name, image)

        with open(predict_result_path, 'w') as f:
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = CLASSES[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())

