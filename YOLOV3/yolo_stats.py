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

do_plots = False

############################################
## Settings for which trained YOLO to use ##
############################################
# path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x0p396_ds4_mb8_nopad_nocl'
# root = "epoch49"
path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x0p396_%s_nopad_nocl' % sys.argv[1]
root = "epoch%s" % sys.argv[2]


#####################################
## Read cluster data in fits files ##
#####################################
pathData="/home/users/ilic/ML/SDSS_fits_data/"
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = np.array(hdul1[1].data)
mem_full_data = np.array(hdul2[1].data)
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

#########################################################
## Prepare some paths to YOLO detections/ground truths ##
#########################################################
gt_path = path + '/mAP_%s/ground-truth' % root
pr_path = path + '/mAP_%s/predicted' % root
gtf_path = path + '/mAP_%s_empty/ground-truth' % root
prf_path = path + '/mAP_%s_empty/predicted' % root

################################################################
## Read clusters validation file with true bbox for reference ##
################################################################
all_clu = {}
with open(path + '/valid.txt','r') as f:
    lines = f.readlines()
ix_clu = 0
for line in tqdm(lines):
    clu = []
    spl = line.split()
    # ix_clu = int(spl[0].split('/')[-1][:-5])
    for s in spl[1:]:
        clu.append(list(map(float, s.split(','))))
    # all_clu.append(clu)
    all_clu[ix_clu] = np.array(clu).astype('int')
    ix_clu += 1
    # all_clu.append(int(line.split()[0].split('/')[-1][:-5]))

###################################
## Read results from predictions ##
###################################
gt_lc = np.loadtxt(
    gt_path + '/all_txts_lc',
    dtype=[('lc','int'),('fn','<U20')],
)
gt_txt = np.loadtxt(
    gt_path + '/all_txts',
    usecols=(1,2,3,4,5),
)
pr_lc = np.loadtxt(
    pr_path + '/all_txts_lc',
    dtype=[('lc','int'),('fn','<U20')],
)
pr_txt = np.loadtxt(
    pr_path + '/all_txts',
    usecols=(1,2,3,4,5),
)
gtf_lc = np.loadtxt(
    gtf_path + '/all_txts_lc',
    dtype=[('lc','int'),('fn','<U20')],
)
gtf_txt = np.loadtxt(
    gtf_path + '/all_txts', # should be empty
    usecols=(1,2,3,4),
)
prf_lc = np.loadtxt(
    prf_path + '/all_txts_lc',
    dtype=[('lc','int'),('fn','<U20')],
)
prf_txt = np.loadtxt(
    prf_path + '/all_txts',
    usecols=(1,2,3,4,5),
)
clcgt = np.append(0, np.cumsum(gt_lc['lc'])[:-1])
clcpr = np.append(0, np.cumsum(pr_lc['lc'])[:-1])
clcprf = np.append(0, np.cumsum(prf_lc['lc'])[:-1])




####################################################################################
####################################################################################
####################################################################################
####################################################################################




#########################################################
## For empty images, do some pure counting performance ##
#########################################################
# Loop over all images without clusters
ct = 0
TN, FP = 0, []
for i in tqdm(range(len(gtf_lc)-1)):
    # Read predicted bbox(es)
    coords_pr = prf_txt[clcprf[i]:clcprf[i+1], :]
    n_pr = len(coords_pr)
    # If no predicted bbox: add one TN
    if n_pr == 0:
        TN += 1
    # If any predicted bbox: add one FP per predicted bbox
    # NB: those FP are associated with their image 'index' via 'ct'
    #     and with their detection score
    else:
        for i in range(n_pr):
            FP.append([coords_pr[i][0], ct])
        ct += 1
all_TN_empty = TN
all_FP_empty = np.array(FP)

# Compute metrics as a function of detection threshold
'''
thres_all_TN_empty, thres_all_FP_empty = [], []
thres = np.linspace(0., 0.9999, 10000)
for t in tqdm(thres):
    g = all_FP_empty[:, 0] >= t                     # keep only FP above detection threshold
    new_ct = len(np.unique(all_FP_empty[g, 1]))     # number of images remaining with n_FP > 0
    diff = ct - new_ct                              # number of images with n_FP = 0
    thres_all_FP_empty.append(g.sum())              # recount the number of FP after threshold
    thres_all_TN_empty.append(all_TN_empty + diff)  # If all FP for given image removed, add one TN
thres_all_TN_empty = np.array(thres_all_TN_empty)
thres_all_FP_empty = np.array(thres_all_FP_empty)
'''


##############################################################
## For clusters images, option 1: pure counting performance ##
##############################################################
'''
# Area threshold on which bbox to include in the counting
# all_a = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
all_a = np.linspace(0., 1., 101)

for a_thres in all_a:

    print(f"Area threshold = {a_thres}")

    # Loop over all images with clusters
    TP, FP, FN = [], [], []
    for i in tqdm(range(len(gt_lc)-1)):
        # Read ground truth bbox(es)
        coords_gt = gt_txt[clcgt[i]:clcgt[i+1], :]
        g = coords_gt[:, -1] >= a_thres
        coords_gt = coords_gt[g, :]
        n_gt = len(coords_gt)
        # Read predicted bbox(es)
        coords_pr = pr_txt[clcpr[i]:clcpr[i+1], :]
        n_pr = len(coords_pr)
        # If any predicted box(es), order them by decreasing probability score
        if n_pr > 0:
            ix = coords_pr[:, 0].argsort()[::-1]
        # If no predicted bbox: add one FN per true bbox
        if n_pr == 0:
            for i in range(n_gt):
                FN.append(1.)
        # If correct number of predicted bboxes: add one TP per true bbox
        elif n_pr == n_gt:
            for c in coords_pr:
                TP.append(c[0])
        # If too many predicted bboxes:
        #   add one TP per true bbox and one FP per excess predicted bbox
        elif n_pr > n_gt:
            for i in ix[:n_gt]:
                TP.append(coords_pr[i][0])
            for i in ix[n_gt:]:
                FP.append(coords_pr[i][0])
        # If not enough predicted bboxes:
        #   add one TP per predicted bbox and one FN per missing predicted bbox
        elif n_pr < n_gt:
            for c in coords_pr:
                TP.append(c[0])
            for i in range(n_gt-n_pr):
                FN.append(1.)
    TP = np.array(TP)
    FP = np.array(FP)
    FN = np.array(FN)

    # Compute metrics as a function of threshold
    nTP, nFP, nFN = [], [], []
    thres = np.linspace(0., 0.9999, 10000)
    for t in tqdm(thres):
        g1 = TP >= t
        g2 = FP >= t
        g3 = FN >= t
        nTP.append(g1.sum()) # TPs below threshold are removed
        nFP.append(g2.sum()) # FPs below threshold are removed
        nFN.append(g3.sum() + (len(g1) - g1.sum())) # TPs removed become FNs
    nTP = np.array(nTP)
    nFP = np.array(nFP)
    nFN = np.array(nFN)

    # Recall/Completness
    comp = nTP / (nTP + nFN)
    # Precision/Purity
    pure = nTP / (nTP + nFP + nFPb)
    # Accuracy
    acc = (nTP + nTN) / (nTP + nTN + nFN + nFP + nFPb)


    # Plots
    g = np.isfinite(pure) & np.isfinite(comp) & np.isfinite(acc)
    pure = pure[g]
    comp = comp[g]
    acc = acc[g]
    thres = thres[g]

    dist = np.sqrt((pure-1)**2. + (comp-1)**2.)
    diff = (pure - comp)**2.
    i_dist_min = np.argmin(dist)
    i_diff_min = np.argmin(diff)
    green_pure = pure[i_dist_min]
    green_comp = comp[i_dist_min]
    green_thres = thres[i_dist_min]
    purple_pure = purple_comp = pure[i_diff_min]
    purple_thres = thres[i_diff_min]
    print("green thres = %s" % green_thres)
    print("purple thres = %s" % purple_thres)

    with open(path + "/%s_stat.txt" % root, "a") as f:
        f.write("%s %s %s %s %s %s\n" % (
            a_thres,
            green_pure,
            green_comp,
            green_thres,
            purple_pure,
            purple_thres,
        ))

    if do_plots:
        plt.clf()
        plt.title(root)
        plt.plot(pure, comp)
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.xlim(0.8, 1.)
        plt.ylim(0.8, 1.)
        th = np.linspace(0., 2.*np.pi, 1001)
        plt.plot(
            dist[i_dist_min]*np.cos(th)+1.,
            dist[i_dist_min]*np.sin(th)+1.,
            ls='--',
            color='green',
            lw=0.5,
        )
        plt.axvline(green_pure, ls='--', color='green', lw=0.5)
        plt.axhline(green_comp, ls='--', color='green', lw=0.5)
        plt.plot([0.8, 1.], [0.8, 1.], ls='--', color='purple', lw=0.5)
        plt.axvline(purple_pure, ls='--', color='purple', lw=0.5)
        plt.axhline(purple_comp, ls='--', color='purple', lw=0.5)
        suf = "thr%s" % str(a_thres).replace(".", "p")
        plt.savefig(path + "/rec_vs_prec_%s_%s.pdf" % (root, suf))
        plt.savefig(path + "/rec_vs_prec_%s_%s.png" % (root, suf))

        plt.clf()
        plt.title(root)
        plt.plot(thres, comp, label='comp = TP / (TP + FN)')
        plt.plot(thres, pure, label='pure = TP / (TP + FP)')
        plt.plot(thres, acc, label='acc = (TP + TN) / TOT')
        plt.axvline(green_thres, ls='--', color='green', lw=0.5)
        plt.axhline(green_pure, ls='--', color='green', lw=0.5)
        plt.axhline(green_comp, ls='--', color='green', lw=0.5)
        plt.axvline(purple_thres, ls='--', color='purple', lw=0.5)
        plt.axhline(purple_pure, ls='--', color='purple', lw=0.5)
        plt.legend()
        plt.xlim(0.3, 1.)
        plt.ylim(0., 1.)
        plt.savefig(path + "/rec_prec_acc_%s_%s.pdf" % (root, suf))
        plt.savefig(path + "/rec_prec_acc_%s_%s.png" % (root, suf))
'''

####################################################
## For clusters images, option 2: counting & IoUs ##
####################################################

def IoUs(pred_r, true_r):
    # Predicted box coordinates
    x1p,y1p,x2p,y2p = pred_r
    # True box coordinates
    x1t,y1t,x2t,y2t = true_r
    # For the predicted box, ensure x1<x2 and y1<y2
    X1p = min(x1p,x2p)
    X2p = max(x1p,x2p)
    Y1p = min(y1p,y2p)
    Y2p = max(y1p,y2p)
    # Calculate area of true box
    At = (x2t-x1t)*(y2t-y1t)
    # Calculate area of predicted box
    Ap = (X2p-X1p)*(Y2p-Y1p)
    # Calculate intersection of true and predicted boxes
    x1i = max(X1p,x1t)
    x2i = min(X2p,x2t)
    y1i = max(Y1p,y1t)
    y2i = min(Y2p,y2t)
    if (x2i>x1i) & (y2i>y1i):
        intrsc = (x2i-x1i)*(y2i-y1i)
    else:
        intrsc = 0.
    # Find coordinates of smallest enclosing box
    x1c = min(X1p,x1t)
    x2c = max(X2p,x2t)
    y1c = min(Y1p,y1t)
    y2c = max(Y2p,y2t)
    # Calculate its area
    Ac = (x2c-x1c)*(y2c-y1c)
    # Calculate metrics
    union = Ap + At - intrsc
    IoU = intrsc / union
    GIoU = IoU - 1. + union / Ac
    return IoU, GIoU

def do_stat(p):
    i, a_thres, p_thres = p
    # Read ground truth bbox(es)
    coords_gt = gt_txt[clcgt[i]:clcgt[i+1], :]
    g = coords_gt[:, -1] >= a_thres # Apply area threshold
    coords_gt = coords_gt[g, :]
    n_gt = len(coords_gt)
    # Read predicted bbox(es)
    coords_pr = pr_txt[clcpr[i]:clcpr[i+1], :]
    g = coords_pr[:, 0] >= p_thres # Apply score threshold
    coords_pr = coords_pr[g, :]
    n_pr = len(coords_pr)
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
    # If too many predicted bboxes:
    # add one TP per truth bbox and one FP per excess predicted bbox
    elif n_pr > n_gt:
        TP = n_gt
        FP = n_pr - n_gt
        TN, FN = 0, 0
    # If not enough predicted bboxes:
    # add one TP per predicted bbox and one FN per missing predicted bbox
    elif n_pr < n_gt:
        TP = n_pr
        FN = n_gt - n_pr
        TN, FP = 0, 0
    ### Pair truth and predicted boxes to get best average GIoU
    # Special case of no predicted bboxes
    if n_pr == 0:
        avg_GIoU = -2.
        return TP, TN, FP, FN, avg_GIoU
    # Special case of no truth bboxes
    if n_gt == 0:
        avg_GIoU = -2.
        return TP, TN, FP, FN, avg_GIoU
    # Compute matrix of GIoU between (predicted,truth) pairs
    mat_GIoU = np.zeros((n_pr, n_gt))
    for ix in range(n_pr):
        for jx in range(n_gt):
            mat_GIoU[ix, jx] = IoUs(coords_pr[ix, 1:], coords_gt[jx, :-1])[1]
    # Special case of one truth or one predicted bbox (take max of previous array)
    if (n_pr == 1) or (n_gt == 1):
        avg_GIoU = mat_GIoU.max()
        return TP, TN, FP, FN, avg_GIoU
    # General case (find set of pairs with max avg GIoU)
    all_avg_GIoU = []
    for comb in list(permutations(range(max(n_pr,n_gt)), n_gt)):
        tmp_sum_GIoU = 0.
        for ix in range(n_gt):
            if comb[ix] < n_pr:
                tmp_sum_GIoU += mat_GIoU[comb[ix], ix]
        all_avg_GIoU.append(np.mean(tmp_sum_GIoU))
    avg_GIoU = np.max(all_avg_GIoU)
    return TP, TN, FP, FN, avg_GIoU

if __name__ == '__main__':
    # def get_tot(pthr, athr, min_mode=True):
    def get_tot(pthr, athr):
        pool = Pool(32)
        inputs = []
        for i in range(len(gt_lc)-1):
            inputs.append([i, athr, pthr])
        # res = list(
        #     tqdm(
        #         pool.imap(do_stat, inputs),
        #         total=len(inputs),
        #         smoothing=0.,
        #     )
        # )
        res = pool.map(do_stat, inputs)
        pool.close()
        pool.join()
        all_stats = np.array(res)
        # g = all_stats[:, -1] > -1.9
        # if not min_mode:
        #     tot_avg_GIoU = (
        #         np.sum(all_stats[g, 0] * all_stats[g, -1])
        #         / np.sum(all_stats[g, 0])
        #     )
        #     return tot_avg_GIoU
        # Metrics
        all_TP = np.sum(all_stats[:, 0])
        all_TN = np.sum(all_stats[:, 1])
        all_FP = np.sum(all_stats[:, 2])
        all_FN = np.sum(all_stats[:, 3])
        tot_avg_GIoU = (
            np.sum(all_stats[:, 0] * all_stats[:, 4])
            / np.sum(all_stats[:, 0])
        )
        # Empty images
        g = all_FP_empty[:, 0] >= pthr
        new_ct = len(np.unique(all_FP_empty[g, 1]))
        diff = ct - new_ct
        thres_all_FP_empty = g.sum()
        thres_all_TN_empty = all_TN_empty + diff
        # Recall/Completness
        comp = all_TP / (all_TP + all_FN)
        # Precision/Purity
        pure = all_TP / (all_TP + all_FP + thres_all_FP_empty)
        # Accuracy
        acc = (
            (all_TP + all_TN + thres_all_TN_empty) /
            (
                all_TP + all_TN + all_FP + all_FN
                + thres_all_FP_empty + thres_all_TN_empty
            )
        )
        output = (
            all_TP, all_TN, all_FP, all_FN,
            thres_all_FP_empty, thres_all_TN_empty,
            comp, pure, acc,
            np.sqrt((1-comp)**2+(1-pure)**2), tot_avg_GIoU,
        )
        return output
    # mini = minimize(get_tot, [0.5], args=(1., True), method='Nelder-Mead')
    plt.figure(figsize=(16,9))
    all_outs = []
    ix = 1
    for athr_tmp in tqdm(np.linspace(0., 1., 21)):
        plt.subplot(3,7,ix)
        ix += 1
        outs = []
        for pthr_tmp in np.linspace(0.3, 0.99, 70):
            outs.append(get_tot(pthr_tmp, athr_tmp))
        all_outs.append(outs)
        outs = np.array(outs)
        x = np.linspace(0.3, 0.99, 70)
        g = np.all(np.isfinite(outs), axis=1)
        plt.plot(x, outs[:, 6], label='Completness', color='C0')
        plt.plot(x, outs[:, 7], label='Purity', color='C1')
        plt.plot(x, outs[:, 8], label='Accuracy', color='C2')
        plt.axvline(x[g][outs[g, 8].argmax()], ls='--', color='C2')
        plt.plot(x, outs[:, 9], label='D(1,1)', color='C3')
        plt.axvline(x[g][outs[g, 9].argmin()], ls='--', color='C3')
        plt.plot(x, outs[:, 10], label='Avg GIoU', color='C4')
    plt.legend()
    plt.suptitle(f"Epoch {sys.argv[2]}")
    plt.tight_layout()
    plt.savefig(f"{path}/GIoU_{sys.argv[2]}.pdf")
    np.savez(
        f"{path}/stats_{sys.argv[2]}",
        all_outs=np.array(all_outs),
    )

sys.exit()

plt.clf()
x, y, d, c, io = [], [], [], [], []
for i in range(1,100):
    t = np.load(f"/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x0p396_ds4_mb8_nopad_nocl/stats_{i}.npz")['all_outs'][20, :, :]
    g = np.all(np.isfinite(t), axis=1)
    ix = t[g, 9].argmin()
    x.append(t[g, 6][ix])
    y.append(t[g, 7][ix])
    d.append(t[g, 9].min())
    c.append(i)
    io.append(t[g, 10][ix])
plt.subplot(1,2,1)
plt.scatter(x, y, c=c)
plt.colorbar()
plt.subplot(1,2,2)
plt.scatter(x, y, c=io)
plt.colorbar()
plt.savefig("opti.pdf")
dist = np.array(d)**2. + (1.-np.array(io))**2.
r = np.sqrt(dist.min())
theta = np.linspace(0., 2.*np.pi, 1001)
plt.figure(figsize=(6,6))
plt.scatter(io, d, c=c)
plt.plot(r*np.cos(theta)+1, r*np.sin(theta), color='red')
plt.xlim(0.8,1)
plt.ylim(0.,0.2)
cbar  =plt.colorbar()
cbar.set_label("Epoch")
plt.savefig("opti_2.pdf")
dist = np.array(d)**2. + (1.-np.array(io))**2.
r = np.sqrt(dist.min())
pl


###################################################################
###################################################################


########################
## Binned performance ##
########################

# Binned performance
# metric = "Z_LAMBDA"
# metric = "LAMBDA"
# metric = "M"
metric = "log10M"
n_bins = 10
# thres = 0.7709 # meth 1
thres = 0.7781 # meth 2

q = clu_full_data[metric]
bins = np.linspace(
    q.min(),
    q.max() + (q.max() - q.min()) / n_bins * 1e-3,
    n_bins + 1,
)
dico = {ID:i for ID, i in zip(clu_full_data['ID'], np.digitize(q, bins) - 1)}


# Cluster images
TP, FN = np.zeros(n_bins + 1), np.zeros(n_bins + 1)
for f in tqdm(os.listdir(gt_path)):
    i = int(f[:-4])
    ix = dico[all_clu[i]]
    with open(f"{pr_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_pr = []
        for l in lines:
            sl = l.split()
            if float(sl[1]) >= thres:
                coords_pr.append([float(x) for x in sl[1:]])
    n_pr = len(coords_pr)
    # No box cases:
    if n_pr == 0:
        FN[ix] += 1.
    # Any number of boxes
    else:
        TP[ix] += 1.

# Empty images
TN, FP = 0., 0.
for f in tqdm(os.listdir(gt_path)):
    with open(f"{prf_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_pr = []
        for l in lines:
            sl = l.split()
            if float(sl[1]) >= thres:
                coords_pr.append([float(x) for x in sl[1:]])
    n_pr = len(coords_pr)
    # No box cases:
    if n_pr == 0:
        TN += 1.
    else:
        FP += 1.

comp = TP / (TP + FN)
pure1 = TP / (TP + TP / TP.sum() * FP)
pure2 = TP / (TP + FP / n_bins)
# acc = (TP + TN) / (TP + TN / n_bins + FN + FP / n_bins)


plt.clf()
plt.figure(figsize=(12,4))
plt.subplot(1, 3, 1)
plt.plot(bins, comp, ds='steps-post')
plt.ylabel('Completness')
plt.subplot(1, 3, 2)
plt.plot(bins, pure1, ds='steps-post', label='FP follows clusters distribution')
plt.plot(bins, pure2, ds='steps-post', label='FP uniform')
plt.legend()
plt.ylabel('Purity')
plt.subplot(1, 3, 3)
plt.plot(bins, TP + FN, ds='steps-post')
plt.hist(q, bins=bins, histtype='step', ls='--', color='black')
plt.xlabel(metric)
plt.ylabel('Number of clusters')
plt.tight_layout()
plt.savefig("purity_completness_%s_bis.pdf" % metric)


###################################################################
###################################################################


####################################
## Fancy performance - bbox pairs ##
####################################

def get_IOU(c_pred, c_true):
    x1, y1, x2, y2 = c_pred
    X1, Y1, X2, Y2 = c_true
    A1 = (x2 - x1) * (y2 - y1)
    A2 = (X2 - X1) * (Y2 - Y1)
    x_dist = (min(x2, X2) -
              max(x1, X1))
    y_dist = (min(y2, Y2) -
              max(y1, Y1))
    I = 0
    if x_dist > 0 and y_dist > 0:
        I = x_dist * y_dist
    U = A1 + A2 - I
    return I / U

def get_R(c_pred, c_true):
    x1, y1, x2, y2 = c_pred
    X1, Y1, X2, Y2 = c_true
    A1 = (x2 - x1) * (y2 - y1)
    A2 = (X2 - X1) * (Y2 - Y1)
    return 4. / np.pi * np.arctan(A1 / A2)

# thres = 0.77
thres = 0.8869

# Loop over all images with clusters
# for f in tqdm(os.listdir(gt_path)):
def get_ct(f):
    # Read ground truth bbox(es)
    with open(f"{gt_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_gt = []
        for l in lines:
            ls = [int(x) for x in l.split()[1:]]
            test = (ls.count(0) + ls.count(2048)) > 0
            # coords_gt.append(ls + [1 * test])
            coords_gt.append(ls)
    n_gt = len(coords_gt)
    # Read predicted bbox(es)
    with open(f"{pr_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_pr = []
        for l in lines:
            ls = l.split()
            if float(ls[1]) >= thres:
                coords_pr.append([int(x) for x in ls[2:]])
    n_pr = len(coords_pr)
    # return [n_gt, n_pr]
    if n_gt == n_pr == 1:
        # return 1
    # else:
        # return 0
        return [
            get_IOU(coords_pr[0], coords_gt[0]),
            get_R(coords_pr[0], coords_gt[0]),
        ]


if __name__ == '__main__':
    pool = Pool(32)
    res = list(
        tqdm(
            pool.imap(get_ct, os.listdir(gt_path)),
            total=len(os.listdir(gt_path)),
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()

res = [r for r in res if r is not None]
res = np.array(res)

import matplotlib
from labellines import labelLines
plt.clf()
x = np.linspace(0., 1., 1001)
V = list(np.linspace(0.1, 1., 10)) + list(1./np.linspace(0.1, 1., 10)[:-1])
# V = list(np.linspace(0.2, 1., 5)) + list(1./np.linspace(0.2, 1., 5)[:-1])
SV = ['%3.2f' % v for v in V]
# V = [1, 1.5, 1/1.5, 2, 1/2, 1.25, 1./1.25]
# SV = ['1', '1.5', '1/1.5', '2', '1/2', '1.25','1/1.25']
for v, sv in zip(V, SV):
    if v <= 1.:
        X = [0., v]
    else:
        X = [0., 1./v]
    Y = [4. / np.pi * np.arctan(v), 4. / np.pi * np.arctan(v)]
    if v == 1.:
        lab = r'$A_{pred}/A_{true}=1$'
    else:
        lab = sv
    plt.plot(
        X, Y,
        ls='--',
        color='black',
        alpha=0.3,
        lw=0.1,
        label=lab,
    )
labelLines(plt.gca().get_lines(), fontsize=4, xvals=(0.01, 0.01), ha="left")
plt.plot(x, 4. / np.pi * np.arctan(x), ls='--', color='black', alpha=0.3, lw=0.1)
plt.plot(x, 4. / np.pi * np.arctan(1./x), ls='--', color='black', alpha=0.3, lw=0.1)
plt.hist2d(
    res[:, 0],
    res[:, 1],
    bins=128,
    range=[[0, 1], [0, 2]],
    norm=matplotlib.colors.LogNorm(),
)
plt.xlabel("IoU")
plt.ylabel(r"$\frac{4}{\pi}\tan^{-1}(\frac{A_{pred}}{A_{true}})$")
plt.colorbar()
plt.savefig("perf_graph.pdf")

import corner
plt.clf()
corner.corner(
    res,
    plot_contours=False,
    plot_datapoints=False,
    # range=[[0,1],[0,2]],
    bins=64,
    labels=[
        "IoU",
        r"$\frac{4}{\pi}\tan^{-1}(\frac{A_{pred}}{A_{true}})$",
    ],
)
plt.savefig("perf_graph_corner.pdf")



###################################################################
###################################################################
###################################################################
###################################################################

'''
thres = 0.5536
cl = [
    'cluster_0p10_0p20',
    'cluster_0p15_0p25',
    'cluster_0p20_0p30',
    'cluster_0p25_0p35',
    'cluster_0p30_0p40',
    'cluster_0p35_0p45',
    'cluster_0p40_0p50',
    'cluster_0p45_0p55',
    'cluster_0p50_0p60',
    'cluster_0p55_0p65',
]
midz = [0.15+i*0.05 for i in range(len(cl))]

# Pure counting performance
nf = len(os.listdir(gt_path))
pr = []
prf = []
for ix, fn in tqdm(enumerate(os.listdir(gt_path)), total=nf):
    with open(gt_path + '/' + fn, 'r') as f:
        lines = f.readlines()
    n_clus = len(lines)
    with open(pr_path + '/' + fn, 'r') as f:
        lines = f.readlines()
    n_detect = len(lines)
    ct = 0
    for line in lines:
        if ct < n_clus:
            ct += 1
            pr.append(float(line.split()[1]))
        else:
            prf.append(float(line.split()[1]))
    if ct < n_clus:
        pr += [0.]*(n_clus - ct)
# nf = len(os.listdir(gtf_path))
for ix, fn in tqdm(enumerate(os.listdir(gtf_path)[:nf]), total=nf):
    with open(prf_path + '/' + fn, 'r') as f:
        lines = f.readlines()
    for line in lines:
        prf.append(float(line.split()[1]))


###################################################################
###################################################################




nf = len(os.listdir(gt_path))
gt = np.zeros((nf, 4))
pr = np.zeros((nf, 5))
zs = np.zeros(nf)
ls = np.zeros(nf)
for ix, fn in tqdm(enumerate(os.listdir(gt_path)),total=nf):
    g = np.where(clu_full_data['ID'] == all_clu[ix])[0][0]
    zs[ix] = clu_full_data['Z_LAMBDA'][g]
    ls[ix] = clu_full_data['LAMBDA'][g]
    with open(gt_path + '/' + fn, 'r') as f:
        lines = f.readlines()
    if len(lines) > 1:
        print("WARNING, MORE THAN ONE TRUE BOX")
    gt[ix, :] = [int(x) for x in lines[0].split()[1:]]
    with open(pr_path + '/' + fn, 'r') as f:
        lines = f.readlines()
    pr[ix, :4] = [int(x) for x in lines[0].split()[2:]]
    pr[ix, 4] = lines[0].split()[1]
    if len(lines) > 1:
        print("WARNING, MORE THAN ONE PREDICTED BOX")

# gtf = np.zeros((len(os.listdir(gtf_path)), 4))
prf = np.zeros((len(os.listdir(gtf_path)), 5))
cts = np.zeros(len(os.listdir(gtf_path)))
for ix, fn in enumerate(os.listdir(gtf_path)):
    # with open(gtf_path + '/' + fn, 'r') as f:
    #     lines = f.readlines()
    # gtf[ix, :] = [int(x) for x in lines[0].split()[1:]]
    with open(prf_path + '/' + fn, 'r') as f:
        lines = f.readlines()
    if len(lines) >= 1:
        prf[ix, :4] = [int(x) for x in lines[0].split()[2:]]
        prf[ix, 4] = lines[0].split()[1]
        cts[ix] = len(lines)
    # elif len(lines) > 1:
        # tmp = np.argmax([float(line.split()[1]) for line in lines])
        # prf[ix, 0] = lines[0].split()[1]
        # prf[ix, 1:] = [int(x) for x in lines[0].split()[2:]]
        # print("WARNING, MORE THAN ONE BOX")

# HISTOGRAM OF PROBABILITIES
plt.figure()
plt.hist(pr[:, 4], bins=64, alpha=0.5, range=[0,1], label='Cluster set')
plt.hist(prf[:, 4], bins=64, alpha=0.5, range=[0,1], label='Empty set')
plt.xlabel("Confidence score")
plt.legend()
plt.savefig('hist_perf.pdf')

# PRECISION AND RECALL
x = np.linspace(0., 1., 1001)
TP, FP, TN, FN = [], [], [], []
for xx in x:
    g = pr[:, 4] < xx
    FN.append(g.sum())
    g = pr[:, 4] >= xx
    TP.append(g.sum())
    g = prf[:, 4] >= xx
    FP.append(g.sum())
    g = prf[:, 4] < xx
    TN.append(g.sum())
TP = np.array(TP)
FP = np.array(FP)
FN = np.array(FN)
TN = np.array(TN)
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(x, TP/(TP+FP), label='Precision')
axs[0].plot(x, TP/(TP+FN), label='Recall')
axs[0].set_ylim(0.94, 1.02)
axs[0].axhline(1., ls='--', color='black', alpha=0.2)
axs[0].axhline(0.99, ls='--', color='black', alpha=0.2)
axs[0].axhline(0.98, ls='--', color='black', alpha=0.2)
axs[0].axhline(0.97, ls='--', color='black', alpha=0.2)
axs[0].axhline(0.96, ls='--', color='black', alpha=0.2)
axs[0].axhline(0.95, ls='--', color='black', alpha=0.2)
axs[1].plot(x, TP/(TP+FP), label='Precision')
axs[1].plot(x, TP/(TP+FN), label='Recall')
axs[1].set_ylim(-0.05, 1.05)
axs[1].legend()
fig.subplots_adjust(hspace=0)
fig.tight_layout()
fig.savefig('rec_prec.pdf')

# CORNER PLOT
to_corn = np.zeros((pr.shape[0], 17))
to_corn[:, 0] = gt[:, 0] #x1
to_corn[:, 1] = gt[:, 1] #y1
to_corn[:, 2] = gt[:, 2] #x2
to_corn[:, 3] = gt[:, 3] #y2
to_corn[:, 4] = 0.5 * (gt[:, 0] + gt[:, 2]) #xc
to_corn[:, 5] = 0.5 * (gt[:, 1] + gt[:, 3]) #yc
to_corn[:, 6] = (gt[:, 2] - gt[:, 0]) #dx
to_corn[:, 7] = (gt[:, 3] - gt[:, 1]) #dy
to_corn[:, 8] = pr[:, 0] - gt[:, 0]
to_corn[:, 9] = pr[:, 1] - gt[:, 1]
to_corn[:, 10] = pr[:, 2] - gt[:, 2]
to_corn[:, 11] = pr[:, 3] - gt[:, 3]
to_corn[:, 12] = 0.5 * (pr[:, 0] + pr[:, 2]) - to_corn[:, 4]
to_corn[:, 13] = 0.5 * (pr[:, 1] + pr[:, 3]) - to_corn[:, 5]
to_corn[:, 14] = (pr[:, 2] - pr[:, 0]) - to_corn[:, 6]
to_corn[:, 15] = (pr[:, 3] - pr[:, 1]) - to_corn[:, 7]
to_corn[:, 16] = pr[:, 4]
g = pr[:, 4] != 0.
corner.corner(
    to_corn[g, :][:, [0,1,2,3,8,9,10,11,16]],
    labels=[
        'xmin','ymin','xmax','ymax',
        # 'xc','yc','dx','dy',
        'delta_xmin','delta_ymin','delta_xmax','delta_ymax',
        # 'delta_xc','delta_yc','delta_dx','delta_dy',
        'prob',
    ],
)
plt.savefig('corner_plot_test1.pdf')
plt.clf()
corner.corner(
    to_corn[g, :][:, [4,5,6,7,12,13,14,15,16]],
    labels=[
        # 'xmin','ymin','xmax','ymax',
        'xc','yc','dx','dy',
        # 'delta_xmin','delta_ymin','delta_xmax','delta_ymax',
        'delta_xc','delta_yc','delta_dx','delta_dy',
        'prob',
    ],
)
plt.savefig('corner_plot_test2.pdf')

# COMPLETENESS/PURITY AS FUNCTION OF Z/LAMBDA
from scipy.optimize import minimize
def prec_min_rec(x):
    g = pr[:, 4] < x
    FN = g.sum()
    g = pr[:, 4] >= x
    TP = g.sum()
    g = prf[:, 4] >= x
    FP = g.sum()
    g = prf[:, 4] < x
    TN = g.sum()
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)
    return (prec-rec)**2.
res = minimize(prec_min_rec, [0.5], method='Nelder-Mead')
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
bo = np.percentile(zs, np.linspace(0., 100., 11))
h1, b1 = np.histogram(zs, range=[zs.min(), zs.max()], bins=bo)
h2, b2 = np.histogram(zs[pr[:, 4] >=res['x'][0]], range=[zs.min(), zs.max()], bins=bo)
y = h2/h1
y = np.append(y, y[-1])
plt.step(b1, y, where='post')
plt.xlabel('Redshift')
plt.ylabel('Completeness')
plt.subplot(1,2,2)
bo = np.percentile(ls, np.linspace(0., 100., 11))
h1, b1 = np.histogram(ls, range=[ls.min(), ls.max()], bins=bo)
h2, b2 = np.histogram(ls[pr[:, 4] >=res['x'][0]], range=[ls.min(), ls.max()], bins=bo)
y = h2/h1
y = np.append(y, y[-1])
plt.step(b1, y, where='post')
plt.xlabel('Richness')
plt.ylabel('Completeness')
plt.tight_layout()
plt.savefig('comp_z_l.pdf')


##################################

fname = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/log_2048x2048_0p396_pad50.txt"

STEP, lr, giou_loss, conf_loss, prob_loss, total_loss = np.loadtxt(fname, unpack=True)

plt.plot(STEP, giou_loss)
plt.plot(STEP, conf_loss)
plt.plot(STEP, prob_loss)
plt.plot(STEP, total_loss)
plt.show()

##################################
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.io import fits
import healpy as hp

path = '/home/users/ilic/ML/SDSS_image_data/empty_2048x2048_0p396127'

ra, dec = [], []
nf = len(os.listdir(path))
for ix, fn in tqdm(enumerate(os.listdir(path)),total=nf):
    if '.txt' in fn:
        tmp_ra, tmp_dec = np.loadtxt(path + '/' + fn, unpack=True)
        ra.append(tmp_ra)
        dec.append(tmp_dec)

m = np.zeros(hp.nside2npix(2048))
for r, d in tqdm(zip(ra, dec)):
    v = hp.ang2vec(r, d, lonlat=True)
    disk = hp.query_disc(2048, v, 1024*0.4/3600.*np.pi/180.)
    m[disk] += 1
hp.mollview(m, xsize=2048)
plt.savefig('empty_map.png', dpi=600)
'''

#################################################################

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# path = "2048x2048_ds4_0p396_pad50"
# path = "2048x2048_ds4_0p396_pad50_zcut0p3"
# path = "2048x2048_ds4_0p396_pad50_zbins_ovl"
# path = "2048x0p396_ds1_mb32_nopad_A"
# path = "2048x0p396_ds4_mb8_nopad_reg_z"
path = "2048x0p396_ds4_mb8_nopad_nocl"
# path = "2048x0p396_ds2_mb2_nopad_nocl"
# path = "test2"
t1 = np.loadtxt("runs/" + path + '/log_train.txt')
t2 = np.loadtxt("runs/" + path + '/log_valid.txt')
#plt.subplot(2,1,2)
#plt.plot(t1[:, 0], t1[:, 5], label='training sample')
#plt.plot(t2[:, 0], t2[:, 5], '+', label='validation sample')
uniqs, invixs = np.unique(t2[:, 0], return_inverse=True)
x, y1, y2, y2m, y2med = [], [], [], [], []
y2errls, y2errus = [], []
for i, u in enumerate(uniqs[:-1]):
    y2errl, y2erru = [], []
    if i==0:
        lb = -np.inf
    else:
        lb = uniqs[i-1]
    x.append(i+1)
    g = (t2[:, 0] == u) & (np.isfinite(t2[:, 5]))
    y2m.append(t2[:, 5][g].mean())
    y2med.append(np.median(t2[:, 5][g]))
    
    # y2errl.append(y2[-1]-np.percentile(t2[:, 5][g], 16))
    # y2erru.append(np.percentile(t2[:, 5][g], 84)-y2[-1])
    # y2errl2.append(y2[-1]-np.percentile(t2[:, 5][g], 2.5))
    # y2erru2.append(np.percentile(t2[:, 5][g], 97.5)-y2[-1])
    y2.append(np.percentile(t2[:, 5][g], 50))
    for n in [5*i for i in range(1,10)]:
        y2errl.append(np.percentile(t2[:, 5][g], n))
        y2erru.append(np.percentile(t2[:, 5][g], 100-n))
    y2errls.append(y2errl)
    y2errus.append(y2erru)
    g = (t1[:, 0] <= u) & (t1[:, 0] > lb)
    y1.append(t1[:, 5][g].mean())
y2errls = np.array(y2errls)
y2errus = np.array(y2errus)
plt.clf()
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.plot(x, y1, '+-', label='Training')
plt.plot(x, y1, label='Training')
# plt.errorbar(x, y2, yerr=[y2errl, y2erru], label='Validation', capsize=5)
# plt.errorbar(x, y2, yerr=[y2errl2, y2erru2], color='C1', capsize=5)
# plt.plot(x, y2, label='Validation')
plt.plot(x, y2, label='Validation', color="C1", alpha=0.2)
plt.axhline(y2[-1], ls='--', color='black', lw=0.5)
for i in range(y2errls.shape[1]):
    plt.fill_between(x, y2errls[:, i], y2errus[:, i], color="C1", alpha=0.1)
#plt.ylim(0,10)
plt.yscale('log')
plt.legend(loc="lower left")
plt.savefig("runs/" + path + '/loss_alt.pdf')

plt.clf()
plt.plot(x, y1, label='Training')
#plt.plot(x, y2m, label='Validation', color='red')
plt.plot(x, y2med, label='Validation', color='red')
g = np.isfinite(t2[:, 5])
plt.hist2d(
    invixs[g],
    t2[g, 5],
    bins=[
        np.arange(101)-0.5,
        10.**np.linspace(np.log10(t2[g, 5].min()), np.log10(t2[g, 5].max()),65),
    ],
    cmap=plt.cm.Oranges,
    vmax=1300,
    #norm=matplotlib.colors.LogNorm(),
)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("runs/" + path + '/loss_binned.pdf')

###############
#plt.subplot(2,1,2)
plt.clf()
plt.plot(t1[:, 0], t1[:, 5], label='training sample')
plt.plot(t2[:, 0], t2[:, 5], '+', label='validation sample')
for u in np.unique(t2[:, 0]):
    g = (t2[:, 0] == u) & (np.isfinite(t2[:, 5]))
    perc = np.percentile(t2[:, 5][g], [2.5, 16., 50., 84., 97.5])
    # plt.plot(u*np.ones(5), perc, '+',color='C2')
    plt.errorbar(u, perc[2], yerr=[[perc[2]-perc[1]], [perc[3]-perc[2]]], color='C2', capsize=5)
    plt.errorbar(u, perc[2], yerr=[[perc[2]-perc[0]], [perc[4]-perc[2]]], color='C2', capsize=5)
plt.xlabel("Number of batchs")
plt.ylabel("Loss")
# plt.ylim(0,25)
plt.yscale('log')
plt.legend()
plt.savefig("runs/" + path + '/loss.pdf')

