import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.io import fits
from multiprocessing import Pool

# Settings
# path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds2_0p396_pad50'
path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50'
# path = '/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50_zcut0p3'
num = 29

# Read cluster data in fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data

# Paths
gt_path = path + '/mAP_%s/ground-truth' % num
pr_path = path + '/mAP_%s/predicted' % num
gtf_path = path + '/mAP_empty_%s/ground-truth' % num
prf_path = path + '/mAP_empty_%s/predicted' % num

# Clusters
all_clu = []
with open(path + '/valid.txt','r') as f:
    lines = f.readlines()
for line in lines:
    all_clu.append(int(line.split()[0].split('/')[-1][:-5]))

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

# PRECISION AND RECALL
pr = np.array(pr)
prf = np.array(prf)
x = np.linspace(0.2, 1., 1001)
TP, FP, TN, FN = [], [], [], []
for xx in x:
    g = pr < xx
    FN.append(g.sum())
    g = pr >= xx
    TP.append(g.sum())
    g = prf >= xx
    FP.append(g.sum())
    g = prf < xx
    TN.append(g.sum())
TP = np.array(TP)
FP = np.array(FP)
FN = np.array(FN)
TN = np.array(TN)
# fig, axs = plt.subplots(2, 1, sharex=True)
plt.subplot2grid((4, 4), (0, 0), 2, 2)
plt.plot(x, TP/(TP+FP), label='Precision')
plt.plot(x, TP/(TP+FN), label='Recall')
plt.ylim(0.89, 1.01)
for v in np.linspace(0.9, 1., 6):
    plt.axhline(v, ls='--', color='black', alpha=0.2)
plt.xlabel("Detection threshold")
plt.subplot2grid((4, 4), (2, 0), 2, 2)
plt.plot(x, TP/(TP+FP), label='Precision')
plt.plot(x, TP/(TP+FN), label='Recall')
plt.ylim(-0.05, 1.05)
plt.xlabel("Detection threshold")
plt.legend()
# fig.subplots_adjust(hspace=0)
plt.subplot2grid((4, 4), (1, 2), 2, 2)
plt.plot(TP/(TP+FP), TP/(TP+FN))
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.xlim(0.9, 1.)
plt.ylim(0.9, 1.)
dist = np.sqrt((TP/(TP+FP)-1)**2. + (TP/(TP+FN)-1)**2.)
r = np.nanmin(dist)
th = np.linspace(0., 2.*np.pi, 1001)
plt.plot(r*np.cos(th)+1., r*np.sin(th)+1., ls='--', color='green', lw=0.5)
xt = (TP/(TP+FP))[dist == r]
yt = (TP/(TP+FN))[dist == r]
plt.axvline(xt, ls='--', color='green', lw=0.5)
plt.axhline(yt, ls='--', color='green', lw=0.5)
diff = (TP/(TP+FP) - TP/(TP+FN))**2.
r = np.nanmin(diff)
plt.plot([0.9, 1.], [0.9, 1.], ls='--', color='purple', lw=0.5)
xt = yt = (TP/(TP+FP))[diff == r]
plt.axvline(xt, ls='--', color='purple', lw=0.5)
plt.axhline(yt, ls='--', color='purple', lw=0.5)
plt.tight_layout()
plt.savefig(path + ('/rec_prec_%s.pdf' % num))
'''
plt.subplot2grid((4, 4), (0, 0), 2, 2)
plt.plot(x, TP/(TP+FP), label='Precision', color='C0', ls='--')
plt.plot(x, TP/(TP+FN), label='Recall', color='C1', ls='--')
plt.subplot2grid((4, 4), (2, 0), 2, 2)
plt.plot(x, TP/(TP+FP), label='Precision', color='C0', ls='--')
plt.plot(x, TP/(TP+FN), label='Recall', color='C1', ls='--')
plt.subplot2grid((4, 4), (1, 2), 2, 2)
plt.plot(TP/(TP+FP), TP/(TP+FN), color='C0', ls='--')
plt.savefig(path + ('/rec_prec_%s_compar.pdf' % num))
'''

# HISTOGRAM OF PROBABILITIES
plt.figure()
plt.hist(pr, bins=64, alpha=0.5, range=[0,1], label='Cluster set')
plt.hist(prf, bins=64, alpha=0.5, range=[0,1], label='Empty set')
plt.xlabel("Confidence score")
plt.legend()
plt.savefig(path + ('/hist_perf_%s.pdf'  % num))
'''
pr0 = pr
prf0 = prf
plt.figure()
plt.hist(pr0, bins=64, alpha=0.5, range=[0,1], label='Cluster set')
plt.hist(prf0, bins=64, alpha=0.5, range=[0,1], label='Empty set')
plt.hist(pr, bins=64, alpha=0.5, range=[0,1], label='Cluster set', histtype='step', color='C0')
plt.hist(prf, bins=64, alpha=0.5, range=[0,1], label='Empty set', histtype='step', color='C1')
plt.xlabel("Confidence score")
plt.legend()
plt.savefig(path + ('/hist_perf_%s_compar.pdf'  % num))
'''

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


#################################################################

import matplotlib.pyplot as plt
import numpy as np
path = "2048x2048_ds4_0p396_pad50"
# path = "2048x2048_ds4_0p396_pad50_zcut0p3"
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

