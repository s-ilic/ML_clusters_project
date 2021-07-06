import sys
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode

#################################
# Image settings
ix_start = 0           # which test image to start from (0 = from beginning)
ntl = int(sys.argv[1]) # which save file to load
reso = 0.396127        # arcsec/pix
pix_size = 1024        # image side size in pixels
pad_size = 50          # padding size in pixels
#################################

# Set up some variables
INPUT_SIZE   = cfg.TRAIN.INPUT_SIZE[0]
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)

# Set up some paths
mAP_dir_path = './runs/%s/mAP_empty_%s' % (cfg.YOLO.ROOT, ntl)
predicted_dir_path = './runs/%s/mAP_empty_%s/predicted' % (cfg.YOLO.ROOT, ntl)
ground_truth_dir_path = './runs/%s/mAP_empty_%s/ground-truth' % (cfg.YOLO.ROOT, ntl)
detected_image_path = './runs/%s/detect_empty_%s' % (cfg.YOLO.ROOT, ntl)

# Clean output folders if needed
if ix_start == 0:
    if os.path.exists(mAP_dir_path):
        shutil.rmtree(mAP_dir_path)
    if os.path.exists(predicted_dir_path):
        shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path):
        shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(detected_image_path):
        shutil.rmtree(detected_image_path)
    os.mkdir(mAP_dir_path)
    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(detected_image_path)

# Build model
input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
feature_maps = YOLOv3(input_layer)
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)
model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights("./runs/%s/yolov3_epoch%s" % (cfg.YOLO.ROOT, ntl))

# Main loop
with open("./runs/%s/valid_empty.txt" % cfg.YOLO.ROOT, 'r') as annotation_file:
    for num, line in enumerate(annotation_file):
        if num >= ix_start:
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

            # Draw boxes in image
            if detected_image_path is not None:
                # Draw truth
                tmp_bboxes = []
                for bb in bboxes_gt:
                    tmp_bb = [1,1,1,1,1,1]
                    tmp_bb[:4] = bb
                    tmp_bboxes.append(tmp_bb)
                image = utils.draw_bbox(image, tmp_bboxes, classes=['cluster','truth'])
                image = utils.draw_bbox(image, bboxes)
                # Save image
                cv2.imwrite(detected_image_path+'/'+image_name, image)

            # Output prediction results
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

