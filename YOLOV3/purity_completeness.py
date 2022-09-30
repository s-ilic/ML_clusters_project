import os, sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.io import fits
from multiprocessing import Pool
import numpy.lib.recfunctions as rfn
from itertools import permutations
from scipy.optimize import minimize
import h5py

# Settings:
#   - area threshold above which cluster bboxes are included in the counting
a_thres = 0.9
#   - counting mode: either pure countin
do_plots = False

# Path to database with network testing results
db_path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x0p396_ds4_mb8_nopad_nocl/database_test_valid.hdf5'

# Path to annotation file used for network testing
ann_path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x0p396_ds4_mb8_nopad_nocl/test_valid.txt'

# Read cluster data in fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = np.array(hdul1[1].data)
mem_full_data = np.array(hdul2[1].data)

# Add fields to fits data if needed
clu_full_data = rfn.append_fields(
    clu_full_data,
    'M',
    10.**14.344*(clu_full_data['LAMBDA'] / 40.)**1.33,
)
clu_full_data = rfn.append_fields(
    clu_full_data,
    'log10M',
    14.344  + 1.33 * np.log10(clu_full_data['LAMBDA']/ 40.),
)

# Function for computing IoU and GIoU
def IoUs(coords_pr, coords_gt):
    # Predicted bbox coordinates
    x1p, y1p, x2p, y2p = coords_pr
    # Ground truth bbox coordinates
    x1t, y1t, x2t, y2t = coords_gt
    # For the predicted box, ensure x1<x2 and y1<y2
    X1p, X2p, Y1p, Y2p = min(x1p,x2p), max(x1p,x2p), min(y1p,y2p), max(y1p,y2p)
    # Calculate area of true box
    At = (x2t-x1t)*(y2t-y1t)
    # Calculate area of predicted box
    Ap = (X2p-X1p)*(Y2p-Y1p)
    # Calculate intersection of true and predicted boxes
    x1i, x2i, y1i, y2i = max(X1p,x1t), min(X2p,x2t), max(Y1p,y1t), min(Y2p,y2t)
    if (x2i>x1i) & (y2i>y1i):
        intrsc = (x2i-x1i)*(y2i-y1i)
    else:
        intrsc = 0.
    # Find coordinates of smallest enclosing box
    x1c, x2c, y1c, y2c = min(X1p,x1t), max(X2p,x2t), min(Y1p,y1t), max(Y2p,y2t)
    # Calculate its area
    Ac = (x2c-x1c)*(y2c-y1c)
    # Calculate IoU metrics
    union = Ap + At - intrsc
    IoU = intrsc / union
    GIoU = IoU - 1. + union / Ac
    return IoU, GIoU

# Open database
db = h5py.File(db_path, "r")

# Main function: for a given image and thresholds, returns all relevant metrics
def compute_metrics(p):
    # Read input parameters: image name, weight and score threshold
    k, w_thres, s_thres = p
    # Grab ground truth bboxes and apply weight threshold
    g = db[k]['weights_gt'][:] >= w_thres
    coords_gt = db[k]['bboxes_gt'][:, :][g]
    n_gt = len(coords_gt)
    # Grab predicted bboxes and apply score threshold
    g = db[k]['scores_pr'][:] >= s_thres
    coords_pr = db[k]['bboxes_pr'][:, :][g]
    n_pr = len(coords_pr)
    ### PURE COUNTING METRICS
    # No predicted bbox and no truth bbox: add one TN
    if (n_pr == 0) & (n_gt == 0):
        TN = 1
        TP, FP, FN = 0, 0, 0
    # If no predicted bbox: add one FN per truth bbox
    elif n_pr == 0:
        FN = n_gt
        TP, TN, FP = 0, 0, 0
    # If correct number of predicted bboxes: add one TP per truth bbox
    elif n_pr == n_gt:
        TP = n_gt
        TN, FP, FN = 0, 0, 0
    # If too many predicted bboxes: add one TP per truth bbox and one FP per excess predicted bbox
    elif n_pr > n_gt:
        TP = n_gt
        FP = n_pr - n_gt
        TN, FN = 0, 0
    # If not enough predicted bboxes: add one TP per predicted bbox and one FN per missed truth bbox
    elif n_pr < n_gt:
        TP = n_pr
        FN = n_gt - n_pr
        TN, FP = 0, 0
    ### PARING TRUTH AND PREDICTED BOXES TO GET GIoUs
    # Special case of no predicted bboxes or no truth boxes
    if (n_pr == 0) or (n_gt == 0):
        n_pairs = 0
        avg_GIoU = np.nan
        return TP, TN, FP, FN, avg_GIoU, n_pairs
    # Compute matrix of GIoUs between all {predicted,truth} pairs
    mat_GIoU = np.zeros((n_pr, n_gt))
    for ix in range(n_pr):
        for jx in range(n_gt):
            mat_GIoU[ix, jx] = IoUs(coords_pr[ix], coords_gt[jx])[1]
    # Special case of one truth or one predicted bbox (take max of previous array)
    if (n_pr == 1) or (n_gt == 1):
        n_pairs = 1
        avg_GIoU = mat_GIoU.max()
        return TP, TN, FP, FN, avg_GIoU, n_pairs
    # General case (find set of pairs with max avg GIoU)
    n_pairs = min(n_pr, n_gt)
    all_avg_GIoU = []
    for comb in list(permutations(range(max(n_pr,n_gt)), n_gt)):
        tmp_sum_GIoU = 0.
        for ix in range(n_gt):
            if comb[ix] < n_pr:
                tmp_sum_GIoU += mat_GIoU[comb[ix], ix]
        all_avg_GIoU.append(np.mean(tmp_sum_GIoU))
    avg_GIoU = np.max(all_avg_GIoU)
    return TP, TN, FP, FN, avg_GIoU, n_pairs

# Parallelization of calculation
n_threads = 32
if __name__ == '__main__':
    # Function for looping over all images, for given thresholds
    def get_tot(w_thres, s_thres):
        pool = Pool(n_threads)
        # Create list of inputs
        inputs = []
        for img_name in db.keys():
            inputs.append([img_name, w_thres, s_thres])
        # Compute metrics for all images
        res = pool.map(compute_metrics, inputs)
        pool.close()
        pool.join()
        # "Numpify" results
        all_stats = np.array(res)
        # Compute total Metrics
        all_TP = np.sum(all_stats[:, 0])
        all_TN = np.sum(all_stats[:, 1])
        all_FP = np.sum(all_stats[:, 2])
        all_FN = np.sum(all_stats[:, 3])
        filter = (all_stats[:, 5] != 0)
        tot_avg_GIoU = np.average(
            all_stats[filter, 4],
            weights=all_stats[filter, 5],
        )
        # Recall (== Completness)
        comp = all_TP / (all_TP + all_FN)
        # Precision (== Purity)
        pure = all_TP / (all_TP + all_FP)
        # Accuracy
        acc = (all_TP + all_TN) / (all_TP + all_TN + all_FP + all_FN)
        # (Euclidean) distance to {pure, comp} = {1, 1}
        dist_11 = np.sqrt((1. - comp)**2. + (1. - pure)**2.)
        # Return results
        output = (
            all_TP, all_TN, all_FP, all_FN,
            comp, pure, acc,
            dist_11, # distance to (pure, comp) = (1, 1)
            tot_avg_GIoU,
        )
        return all_TP, all_TN, all_FP, all_FN, comp, pure, acc, dist_11, tot_avg_GIoU
    # Example of use
    w_thres_test, s_thres_test = 1., 0.7
    res = []
    s_thres_vals = np.linspace(0.3, 0.98, 70)
    for s_thres_test in s_thres_vals:
        print(s_thres_test)
        res.append(list(get_tot(w_thres_test, s_thres_test)))
    res = np.array(res)

# Do a plot
x = np.linspace(0.3, 0.99, 70)
plt.plot(x, res[:, 6], label='Completness', color='C0')
plt.plot(x, res[:, 7], label='Purity', color='C1')
plt.plot(x, res[:, 8], label='Accuracy', color='C2')
plt.axvline(x[res[:, 8].argmax()], ls='--', color='C2')
plt.plot(x, res[:, 9], label='D(1,1)', color='C3')
plt.axvline(x[res[:, 9].argmin()], ls='--', color='C3')
plt.plot(x, res[:, 10], label='Avg GIoU', color='C4')
plt.legend()
plt.show()
