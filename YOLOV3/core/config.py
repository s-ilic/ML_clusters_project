from easydict import EasyDict as edict


__C                           = edict()
cfg                           = __C


####################
### Main options ###
####################

__C.YOLO                      = edict()

### Choose name of output folder (code will create it in "./runs")
# __C.YOLO.ROOT = "2048x2048_ds2_0p396_pad50"
# __C.YOLO.ROOT = "2048x2048_ds4_0p396_pad50"
__C.YOLO.ROOT = "2048x2048_ds4_0p396_pad50_zcut0p3" # Root of output files

### File containing the YOLO classes names
__C.YOLO.CLASSES              = "./runs/clusters.names"

### File containing the YOLO baseline anchors
### NOTES: After doing some clustering studies on ground truth labels, it turns out
### that most bounding boxes have certain height-width ratios. So instead of directly
### predicting a bounding box, YOLOv2 (and v3) predict off-sets from a predetermined
### set of boxes with particular height-width ratios - those predetermined set of
### boxes are the anchor boxes.
__C.YOLO.ANCHORS              = "./runs/baseline_anchors.txt"

### List of strides == integer factors by which the input images are reduced
### within the YOLO network when performing the multiscale detection
### NB: Factors are applied to TRAIN.INPUT_SIZE, not the "real" image size
__C.YOLO.STRIDES              = [8, 16, 32]

### Self explanatory
__C.YOLO.ANCHOR_PER_SCALE     = 3


### "Intersection Over Union" loss threshold :- 1 perfect, 0 not the good box - IOU = intersection over union
__C.YOLO.IOU_LOSS_THRESH      = 0.5


########################
### Training options ###
########################

__C.TRAIN                     = edict()

### Verbose mode (print losses during training)
__C.TRAIN.VERBOSE             = True

### Training options
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

####################
### Test options ###
####################

__C.TEST                      = edict()

### TEST options
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
__C.RESUME                    = edict()
__C.RESUME.DO_RESUME          = False
__C.RESUME.FROM_EPOCH         = 10
