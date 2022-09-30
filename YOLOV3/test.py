import os, sys
import cv2
import shutil
import numpy as np
import h5py
import tensorflow as tf
import core.utils as utils
from core.config import cfg
if cfg.YOLO.MODE == "class":
    from core.yolov3 import YOLOv3, decode
elif cfg.YOLO.MODE == "no_class":
    from core.nocl_yolov3 import YOLOv3, decode
elif cfg.YOLO.MODE == "regression":
    from core.reg_yolov3 import YOLOv3, decode
else:
    raise ValueError("Wrong mode specified: %s (must be either class, no_class, or regression)" % cfg.YOLO.MODE)

# Optional overriding of settings
if len(sys.argv) > 2:
    raise ValueError("Wrong number of arguments")
elif len(sys.argv) == 2:
    cmds = sys.argv[1].split(',')
    for cmd in cmds:
        exec(cmd)

# Settings
reso = cfg.TEST.RESO
pix_size = cfg.TEST.PIX_SIZE
input_size   = cfg.YOLO.SIZE

# Prep for real images/galaxies
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

# Set up some paths, filenames, folders, and files
output_root = cfg.TEST.OUTPUT_ROOT
database_fname = f'./runs/{cfg.YOLO.ROOT}/database_{output_root}.hdf5'
if os.path.isfile(database_fname):
    raise OSError(f"Output database already exists: {database_fname}")
f = h5py.File(database_fname, "w")
f.close()
detected_image_path = f'./runs/{cfg.YOLO.ROOT}/detect_{output_root}'
if cfg.TEST.OUTPUT_IMG:
    if os.path.exists(detected_image_path):
        raise OSError(f"Output folder for images already exists: {detected_image_path}")
    os.mkdir(detected_image_path)

# Build model
input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)
model = tf.keras.Model(input_layer, bbox_tensors)
weights_fname = cfg.TEST.WEIGHTS_FNAME
model.load_weights(weights_fname) #loads the weights from the training

# Main loop: feed each image to the trained network and record outputs
with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:

    for num, line in enumerate(annotation_file):

        # Grab truth bboxes (if any)
        annotation = line.strip().split()
        image_path = annotation[0]
        image_name = image_path.split('/')[-1]
        print('Image %s' % image_path)
        num_bbox_gt = len(annotation[1:])
        bboxes_gt = []
        classes_gt = []
        weights_gt = []
        for box in annotation[1:]:
            splbox = box.split(',')
            if cfg.YOLO.MODE == "class":
                bboxes_gt.append(list(map(int, splbox[:4])))
                classes_gt.append(int(splbox[4]))
                weights_gt.append(float(splbox[-1]))
            elif cfg.YOLO.MODE == "no_class":
                bboxes_gt.append(list(map(int, splbox[:4])))
                classes_gt.append(1)
                weights_gt.append(float(splbox[-1]))

        # Create corresponding datasets in main database
        f = h5py.File(database_fname, "a")
        grp_name = image_name.split('.')[0]
        d0 = f.create_dataset(
            f"{grp_name}/num_bbox_gt",
            (1,),
            dtype='i',
        )
        d0[0] = num_bbox_gt
        d1 = f.create_dataset(
            f"{grp_name}/bboxes_gt",
            (num_bbox_gt, 4),
            dtype='f',
        )
        d1[:, :] = np.array(bboxes_gt)
        d2 = f.create_dataset(
            f"{grp_name}/classes_gt",
            (num_bbox_gt,),
            dtype='i',
        )
        d2[:] = np.array(classes_gt)
        d3 = f.create_dataset(
            f"{grp_name}/weights_gt",
            (num_bbox_gt,),
            dtype='f',
        )
        d3[:] = np.array(weights_gt)
        f.close()

        # Print the truth bboxes
        print('  * ground truth boxes:')
        for i in range(num_bbox_gt):
            xmin, ymin, xmax, ymax = bboxes_gt[i]
            cl = classes_gt[i]
            wgt = weights_gt[i]
            print("    - [%d, %d, %d, %d], class = %d, weight = %.3f" % (xmin, ymin, xmax, ymax, cl, wgt))

        # Prediction process
        image_orig = cv2.imread(image_path) # read image
        image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB) # convert col space
        image_size = image.shape[:2]
        image_data = utils.image_preprocess(
            np.copy(image),
            [input_size, input_size],
        )
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        pred_bbox = model(image_data, training=False)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        outputs_pr = utils.postprocess_boxes(
            pred_bbox,
            image_size,
            input_size,
            cfg.TEST.SCORE_THRESHOLD,
            clip=cfg.TEST.CLIP_BBOX,
            no_class=(cfg.YOLO.MODE == "no_class"),
        )
        outputs_pr = utils.nms(
            outputs_pr,
            cfg.TEST.IOU_THRESHOLD,
            method=cfg.TEST.IOU_METHOD,
            sigma=cfg.TEST.SIGMA,
        )

        # Parse the predicted boxes
        num_bbox_pr = len(outputs_pr)
        bboxes_pr = []
        scores_pr = []
        classes_pr = []
        for output in outputs_pr:
            bboxes_pr.append(output[:4])
            scores_pr.append(output[4])
            classes_pr.append(output[5])

        # Create corresponding datasets in main database
        f = h5py.File(database_fname, "a")
        d0 = f.create_dataset(
            f"{grp_name}/num_bbox_pr",
            (1,),
            dtype='i',
        )
        d0[0] = num_bbox_pr
        d1 = f.create_dataset(
            f"{grp_name}/bboxes_pr",
            (num_bbox_pr, 4),
            dtype='f',
        )
        d1[:, :] = np.array(bboxes_pr)
        d2 = f.create_dataset(
            f"{grp_name}/scores_pr",
            (num_bbox_pr,),
            dtype='f',
        )
        d2[:] = np.array(scores_pr)
        d3 = f.create_dataset(
            f"{grp_name}/classes_pr",
            (num_bbox_pr,),
            dtype='i',
        )
        d3[:] = np.array(classes_pr)
        f.close()

        # Print the predicted bboxes
        print('  * predicted boxes:')
        for i in range(num_bbox_pr):
            xmin, ymin, xmax, ymax, score, cl = outputs_pr[i]
            print("    - [%d, %d, %d, %d], score = %.3f, class = %d" % (xmin, ymin, xmax, ymax, score, cl))

        # Optional: make copy of image, draw bboxes, and circle galaxies
        if cfg.TEST.OUTPUT_IMG:
            image = image_orig.copy()
            # Draw truth bboxes
            for bb in bboxes_gt:
                image = utils.draw_bbox(
                    image,
                    [bb + [1, 1]],
                    classes=['detected','truth'],
                    fontScale=2,
                )
            # Draw detected bboxes
            for bb, sc in zip(bboxes_pr, scores_pr):
                image = utils.draw_bbox(
                    image,
                    [list(bb) + [sc, 0]],
                    classes=['detected','truth'],
                    fontScale=2,
                )
            # Optional: draw member galaxies
            if cfg.TEST.DRAW_GALS and (num_bbox_gt > 0):
                clus_id = int(grp_name)
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
                xs = np.round(pix_size - off[0].arcsec / reso * 1.01).astype('int')
                ys = np.round(pix_size - off[1].arcsec / reso * 1.01).astype('int')
                for x, y in zip(xs, ys):
                    image = cv2.circle(image, (x,y), 5, (0,0,255),3)
            # Save image
            cv2.imwrite(f"{detected_image_path}/{image_name}", image)

