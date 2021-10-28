import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.io import fits
from multiprocessing import Pool
import numpy.lib.recfunctions as rfn

############################################
## Settings for which trained YOLO to use ##
############################################
# path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds2_0p396_pad50'
path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50'
# path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50_zcut0p3'
# path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50_zbins'
# path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50_zbins_ovl'
# num = 19
num = 29
meth = "nms"
# meth = "soft-nms"

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
gt_path = path + '/mAP_%s/ground-truth' % num
pr_path = path + '/mAP_%s/predicted' % num
# gt_path = path + '/mAP_%s_%s/ground-truth' % (num, meth)
# pr_path = path + '/mAP_%s_%s/predicted' % (num, meth)
gtf_path = path + '/mAP_empty_%s/ground-truth' % num
prf_path = path + '/mAP_empty_%s/predicted' % num
# gtf_path = path + '/mAP_empty_%s_%s/ground-truth' % (num, meth)
# prf_path = path + '/mAP_empty_%s_%s/predicted' % (num, meth)

# Clusters
all_clu = []
with open(path + '/valid.txt','r') as f:
    lines = f.readlines()
for line in lines:
    all_clu.append(int(line.split()[0].split('/')[-1][:-5]))


####################################################################################
####################################################################################
####################################################################################

#########################################################
## Pure counting performance - method 1: use all boxes ##
#########################################################

# Loop over all images with clusters
TP, FP, FN = [], [], []
for f in tqdm(os.listdir(gt_path)):
    # Read ground truth bbox(es)
    with open(f"{gt_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_gt = [[float(x) for x in l.split()[1:]] for l in lines]
    n_gt = len(coords_gt)
    # Read predicted bbox(es)
    with open(f"{pr_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_pr = [[float(x) for x in l.split()[1:]] for l in lines]
    n_pr = len(coords_pr)
    # If any predicted box(es), order them by decreasing probability score
    if n_pr > 0:
        ix = np.array(coords_pr)[:, 0].argsort()[::-1]
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

# Loop over all images without clusters
ct = 0
TN, FP = [], []
for f in tqdm(os.listdir(gt_path)):
    # Read predicted bbox(es)
    with open(f"{prf_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_pr = [[float(x) for x in l.split()[1:]] for l in lines]
    n_pr = len(coords_pr)
    # If no predicted bbox: add one TN
    if n_pr == 0:
        TN.append(1.)
    # If any predicted bbox: add one FP per predicted bbox
    # NB: those bboxes are associated with image 'index' via 'ct'
    else:
        for i in range(n_pr):
            FP.append([coords_pr[i][0], ct])
        ct += 1
TNb = np.array(TN)
FPb = np.array(FP)

# Compute metrics as a function of threshold
nTNb, nFPb = [], []
thres = np.linspace(0., 0.9999, 10000)
for t in tqdm(thres):
    g = FPb[:, 0] >= t
    new_ct = len(np.unique(FPb[g, 1])) # number of images remaining with nFP > 0
    diff = ct - new_ct
    nFPb.append(g.sum()) # FP below threshold are removed
    nTNb.append(len(TNb) + diff) # If all FP for given image removed, add one TN
nTN = np.array(nTNb)
nFPb = np.array(nFPb)

# Recall/Completness
comp = nTP / (nTP + nFN)
# Precision/Purity
pure = nTP / (nTP + nFP + nFPb)
# Accuracy
acc = (nTP + nTN) / (nTP + nTN + nFN + nFP + nFPb)

# Plots
plt.clf()
plt.plot(pure, comp)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.xlim(0.9, 1.)
plt.ylim(0.9, 1.)
dist = np.sqrt((pure-1)**2. + (comp-1)**2.)
r = np.nanmin(dist)
th = np.linspace(0., 2.*np.pi, 1001)
plt.plot(r*np.cos(th)+1., r*np.sin(th)+1., ls='--', color='green', lw=0.5)
xt = (pure)[dist == r]
yt = (comp)[dist == r]
plt.axvline(xt, ls='--', color='green', lw=0.5)
plt.axhline(yt, ls='--', color='green', lw=0.5)
green_thres = thres[dist == r]
print("green thres = %s" % green_thres)
diff = (pure - comp)**2.
r = np.nanmin(diff)
plt.plot([0.9, 1.], [0.9, 1.], ls='--', color='purple', lw=0.5)
xt = yt = (pure)[diff == r]
plt.axvline(xt, ls='--', color='purple', lw=0.5)
plt.axhline(yt, ls='--', color='purple', lw=0.5)
purple_thres = thres[diff == r]
print("purple thres = %s" % purple_thres)
plt.savefig("rec_vs_prec.pdf")
plt.clf()
plt.plot(thres, comp, label='comp = TP / (TP + FN)')
plt.plot(thres, pure, label='pure = TP / (TP + FP)')
plt.plot(thres, acc, label='acc = (TP + TN) / TOT')
plt.axvline(green_thres, ls='--', color='green', lw=0.5)
plt.axvline(purple_thres, ls='--', color='purple', lw=0.5)
plt.legend()
plt.xlim(0.3,1.)
plt.ylim(0.9, 1.)
plt.savefig("rec_prec_acc.pdf")

# ds2
# green thres = [0.7709]
# purple thres = [0.7861]

# ds4
# green thres = [0.8869]
# purple thres = [0.887]


###################################################################
###################################################################


###############################################################
## Pure counting performance - method 1: use only main boxes ##
###############################################################

# Loop over all images with clusters
TP, FN = [], []
for f in tqdm(os.listdir(gt_path)):
    # Read ground truth bbox(es)
    with open(f"{gt_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_gt = [[float(x) for x in l.split()[1:]] for l in lines]
    n_gt = len(coords_gt)
    # Read predicted bbox(es)
    with open(f"{pr_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_pr = [[float(x) for x in l.split()[1:]] for l in lines]
    n_pr = len(coords_pr)
    # If any predicted box(es), order them by decreasing probability score
    if n_pr > 0:
        ix = np.array(coords_pr)[:, 0].argsort()[::-1]
    # If no predicted bbox: add one FN
    if n_pr == 0:
        FN.append(1.)
    # If any predicted bbox: add one TP
    else:
        TP.append(coords_pr[ix[0]][0])
TP = np.array(TP)
FN = np.array(FN)

# Compute metrics as a function of threshold
nTP, nFN = [], []
thres = np.linspace(0., 0.9999, 10000)
for t in tqdm(thres):
    g1 = TP >= t
    g2 = FN >= t
    nTP.append(g1.sum()) # TPs below threshold are removed
    nFN.append(g2.sum() + (len(g1) - g1.sum())) # TPs removed become FNs
nTP = np.array(nTP)
nFN = np.array(nFN)

# Loop over all images without clusters
ct = 0
TN, FP = [], []
for f in tqdm(os.listdir(gt_path)):
    # Read predicted bbox(es)
    with open(f"{prf_path}/{f}", 'r') as file:
        lines = file.readlines()
        coords_pr = [[float(x) for x in l.split()[1:]] for l in lines]
    n_pr = len(coords_pr)
    # If no predicted bbox: add one TN
    if n_pr == 0:
        TN.append(1.)
    # If any predicted bbox: add one FP per predicted bbox
    # NB: those bboxes are associated with image 'index' via 'ct'
    else:
        for i in range(n_pr):
            FP.append([coords_pr[i][0], ct])
        ct += 1
TN = np.array(TN)
FP = np.array(FP)

# Compute metrics as a function of threshold
nTN, nFP = [], []
thres = np.linspace(0., 0.9999, 10000)
for t in tqdm(thres):
    g = FP[:, 0] >= t
    new_ct = len(np.unique(FP[g, 1])) # number of images remaining with nFP > 0
    diff = ct - new_ct
    nFP.append(new_ct) # == n_images with at least one FP
    nTN.append(len(TN) + diff) #  == n_images with no FP
nTN = np.array(nTN)
nFP = np.array(nFP)

# Recall/Completness
comp = nTP / (nTP + nFN)
# Precision/Purity
pure = nTP / (nTP + nFP)
# Accuracy
acc = (nTP + nTN) / (nTP + nTN + nFN + nFP)

# Plots
plt.clf()
plt.plot(pure, comp)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.xlim(0.9, 1.)
plt.ylim(0.9, 1.)
dist = np.sqrt((pure-1)**2. + (comp-1)**2.)
r = np.nanmin(dist)
th = np.linspace(0., 2.*np.pi, 1001)
plt.plot(r*np.cos(th)+1., r*np.sin(th)+1., ls='--', color='green', lw=0.5)
xt = (pure)[dist == r]
yt = (comp)[dist == r]
plt.axvline(xt[0], ls='--', color='green', lw=0.5)
plt.axhline(yt[0], ls='--', color='green', lw=0.5)
green_thres = thres[dist == r]
print("green thres = %s" % green_thres)
diff = (pure - comp)**2.
r = np.nanmin(diff)
plt.plot([0.9, 1.], [0.9, 1.], ls='--', color='purple', lw=0.5)
xt = yt = (pure)[diff == r]
plt.axvline(xt, ls='--', color='purple', lw=0.5)
plt.axhline(yt, ls='--', color='purple', lw=0.5)
purple_thres = thres[diff == r]
print("purple thres = %s" % purple_thres)
plt.savefig("rec_vs_prec2.pdf")
plt.clf()
plt.plot(thres, comp, label='comp = TP / (TP + FN)')
plt.plot(thres, pure, label='pure = TP / (TP + FP)')
plt.plot(thres, acc, label='acc = (TP + TN) / TOT')
plt.axvline(green_thres[0], ls='--', color='green', lw=0.5)
plt.axvline(purple_thres, ls='--', color='purple', lw=0.5)
plt.legend()
plt.xlim(0.3,1.)
plt.ylim(0.9, 1.)
plt.savefig("rec_prec_acc2.pdf")

#green thres = [0.7781 0.7782]
#purple thres = [0.7904]


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
# path = "2048x2048_ds4_0p396_pad50"
# path = "2048x2048_ds4_0p396_pad50_zcut0p3"
# path = "2048x2048_ds4_0p396_pad50_zbins_ovl"
path = "2048x0p396_ds1_mb32_nopad_A"
# path = "test2"
plt.clf()
t1 = np.loadtxt("runs/" + path + '/log_train.txt')
t2 = np.loadtxt("runs/" + path + '/log_valid.txt')
#plt.subplot(2,1,2)
#plt.plot(t1[:, 0], t1[:, 5], label='training sample')
#plt.plot(t2[:, 0], t2[:, 5], '+', label='validation sample')
uniqs = np.unique(t2[:, 0])
x, y1, y2, y2errl, y2erru, y2errl2, y2erru2 = [], [], [], [], [], [], []
for i, u in enumerate(uniqs):
    if i==0:
        lb = -np.inf
    else:
        lb = uniqs[i-1]
    x.append(i+1)
    g = (t2[:, 0] == u) & (np.isfinite(t2[:, 5]))
    y2.append(t2[:, 5][g].mean())
    y2errl.append(y2[-1]-np.percentile(t2[:, 5][g], 16))
    y2erru.append(np.percentile(t2[:, 5][g], 84)-y2[-1])
    y2errl2.append(y2[-1]-np.percentile(t2[:, 5][g], 2.5))
    y2erru2.append(np.percentile(t2[:, 5][g], 97.5)-y2[-1])
    g = (t1[:, 0] <= u) & (t1[:, 0] > lb)
    y1.append(t1[:, 5][g].mean())
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(x, y1, '+-', label='Training')
plt.errorbar(x, y2, yerr=[y2errl, y2erru], label='Validation', capsize=5)
plt.errorbar(x, y2, yerr=[y2errl, y2erru], color='C1', capsize=5)
#plt.ylim(0,10)
plt.legend()
plt.yscale('log')
plt.savefig("runs/" + path + '/loss_alt.png')
###############
#plt.subplot(2,1,2)
plt.clf()
plt.plot(t1[:, 0], t1[:, 5], label='training sample')
plt.plot(t2[:, 0], t2[:, 5], '+', label='validation sample')
for u in np.unique(t2[:, 0]):
    g = (t2[:, 0] == u) & (np.isfinite(t2[:, 5]))
    perc = np.percentile(t2[:, 5][g], [2.5, 16., 50., 84., 97.5])
    plt.plot(u*np.ones(5), perc, '+',color='C2')
plt.xlabel("Number of batchs")
plt.ylabel("Loss")
plt.ylim(0,25)
plt.legend()
plt.savefig("runs/" + path + '/loss.png')

