#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C


### YOLO options
__C.YOLO                      = edict()


### NEW SILIC SETUP
# __C.YOLO.ROOT = "2048x2048_ds2_0p396_pad50"
# __C.YOLO.ROOT = "2048x2048_ds4_0p396_pad50"
__C.YOLO.ROOT = "2048x2048_ds4_0p396_pad50_zcut0p3"


### Set the class name
__C.YOLO.CLASSES              = "./runs/clusters.names"
__C.YOLO.ANCHORS              = "./runs/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]  #image sampling factor (1/8, 1/16, 1/32) at different scales
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5 #Loss threshold - 1 perfect, 0 not the good box - IOU = intersection over union


### Training options
__C.TRAIN                     = edict()
__C.TRAIN.ANNOT_PATH          = "./runs/%s/train.txt" % __C.YOLO.ROOT  #file with path to training images, one line each image and bounding box on the same line
__C.TRAIN.BATCH_SIZE          = 8 #how many images per batch
# __C.TRAIN.BATCH_SIZE          = 4
# __C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE          = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
# __C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.INPUT_SIZE          = [512] # automatic rescaling to this size in pixels
__C.TRAIN.DATA_AUG            = True # data augmentation shift + rotation + flip ?
# __C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_INIT             = 1e-4  # intial learning rate 
__C.TRAIN.LR_END              = 1e-6 # final learning rate
# __C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.WARMUP_EPOCHS       = 4 #initial learning rate different for these epochs and then final learning rate
__C.TRAIN.EPOCHS              = 30


### TEST options
__C.TEST                      = edict()
__C.TEST.ANNOT_PATH           = "./runs/%s/valid.txt" % __C.YOLO.ROOT
# __C.TEST.BATCH_SIZE           = 2
__C.TEST.BATCH_SIZE           = 1 #1 batch per image, so that he can do the plot with loss per image
# __C.TEST.INPUT_SIZE           = 544
__C.TEST.INPUT_SIZE           = 512
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./runs/%s/detection/" % __C.YOLO.ROOT #plot bounding box + galaxies in clusters (Stephane)
__C.TEST.SCORE_THRESHOLD      = 0.3 #prob associated to bounding box threshold
__C.TEST.IOU_THRESHOLD        = 0.45 # to be understood


### RESUME options - Stephane to continue or stop training
__C.RESUME.DO_RESUME          = False
__C.RESUME.FROM_EPOCH         = 10
