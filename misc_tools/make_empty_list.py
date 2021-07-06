import os
import numpy as np
import healpy as hp
from tqdm import tqdm

n_want = 12203
out_fname = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds2_0p396_pad50/valid_empty.txt"
nside = 2048
m = np.load('poly_map.npz')['arr_0']

empty_path = "/home/users/ilic/ML/SDSS_image_data/empty_2048x2048_0p396127"
ra, dec, good, fnames = [], [], [], []
all_fnames = os.listdir(empty_path)
nf = len(all_fnames)
for ix, fn in tqdm(enumerate(all_fnames),total=nf):
    if '.txt' in fn:
        tmp_ra, tmp_dec = np.loadtxt(empty_path + '/' + fn, unpack=True)
        ra.append(tmp_ra)
        dec.append(tmp_dec)
        pix = hp.ang2pix(nside, tmp_ra, tmp_dec, lonlat=True)
        good.append(m[pix])
        fnames.append(fn)

ct = 0
with open(out_fname, "w") as f:
    for g, fn in zip(good, fnames):
        if g == 1.:
            ct += 1
            f.write(empty_path + '/' + fn[:-3] + 'jpeg\n')
        if ct == n_want:
            break
