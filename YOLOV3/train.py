import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg


# Read training and validation sets
trainset = Dataset('train')
validset = Dataset('valid')

# Set up some variables
logdir = "./runs/%s/log" % cfg.YOLO.ROOT
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

# Define input/output tensors for model creation
input_tensor = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE[0],cfg.TRAIN.INPUT_SIZE[0],3])
conv_tensors = YOLOv3(input_tensor)
output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

# Create model, logs, and output directories (if needed)
model = tf.keras.Model(input_tensor, output_tensors)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

# Load previous weights if resuming some training
if cfg.RESUME.DO_RESUME:
    model.load_weights("./runs/%s/yolov3_epoch%s" % (cfg.YOLO.ROOT, cfg.RESUME.FROM_EPOCH))
    global_steps.assign_add((cfg.RESUME.FROM_EPOCH + 1) * steps_per_epoch)

# Main training function
def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
        #          "prob_loss: %4.2f   total_loss: %4.2f" %(global_steps, optimizer.lr.numpy(),
        #                                                   giou_loss, conf_loss,
        #                                                   prob_loss, total_loss))
        tf.print(
            "%d  %e  %e  %e  %e  %e" % (
                global_steps,
                optimizer.lr.numpy(),
                giou_loss, conf_loss,
                prob_loss, total_loss,
            ),
            output_stream='file:///home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/%s/log_train.txt' % cfg.YOLO.ROOT,
        )
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        elif global_steps < total_steps:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        else:
            lr = cfg.TRAIN.LR_END
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()

# Create validation log
if os.path.exists(logdir + '_valid'):
    shutil.rmtree(logdir + '_valid')
validate_writer = tf.summary.create_file_writer(logdir + '_valid')

# Main validation function
def validate_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=False)
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        tf.print(
            "%d  %e  %e  %e  %e  %e" % (
                global_steps,
                optimizer.lr.numpy(),
                giou_loss, conf_loss,
                prob_loss, total_loss,
            ),
            output_stream='file:///home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/%s/log_valid.txt' % cfg.YOLO.ROOT,
        )
        # writing summary data
        with validate_writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("validate_loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("validate_loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("validate_loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("validate_loss/prob_loss", prob_loss, step=global_steps)
        validate_writer.flush()
        return total_loss


# Main loop
for epoch in range(cfg.TRAIN.EPOCHS):
    print("Epoch %s out of %s" % (epoch + 1, cfg.TRAIN.EPOCHS))
    for image_data, target in tqdm(trainset, smoothing=1):
        train_step(image_data, target)
    for image_data, target in tqdm(validset, smoothing=1):
        valid_loss = validate_step(image_data, target)
    model.save_weights("./runs/%s/yolov3_epoch%s" % (cfg.YOLO.ROOT, epoch))
