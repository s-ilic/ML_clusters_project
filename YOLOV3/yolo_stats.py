import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.io import fits

# Read cluster data in fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data


gt_path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/mAP/ground-truth'
pr_path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/mAP/predicted'

gtf_path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/mAP/ground-truth_fake'
prf_path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/mAP/predicted_fake'

all_clu = []
with open('/home/users/ilic/ML/ML_clusters_project/YOLOV3/data/dataset/valid.txt','r') as f:
    lines = f.readlines()
for line in lines:
    all_clu.append(int(line.split()[0].split('/')[-1][:-5]))

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
    gt[ix, :] = [int(x) for x in lines[0].split()[1:]]
    with open(pr_path + '/' + fn, 'r') as f:
        lines = f.readlines()
    if len(lines) == 1:
        pr[ix, :4] = [int(x) for x in lines[0].split()[2:]]
        pr[ix, 4] = lines[0].split()[1]
    elif len(lines) > 1:
        print("WARNING, MORE THAN ONE BOX")

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
