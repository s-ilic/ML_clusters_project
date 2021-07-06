import numpy as np
from astropy.io import fits
from tqdm import tqdm
from healpy.projector import GnomonicProj as GP
import os, sys
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.image as mpimg
from multiprocessing import Pool


# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)


#################################################################################
#################################################################################


reso = 0.396127
# reso = 0.792254
pix_size = 1024 # half full pic size/side
# pix_size = 2048
pix_pad = 50
thres = 0.9

def get_boxes(i):
    c = SkyCoord(
        ra=clu_full_data['RA'][i]*u.degree,
        dec=clu_full_data['DEC'][i]*u.degree,
        frame='icrs',
    )
    c2 = SkyCoord(
        ra=mem_full_data['RA']*u.degree,
        dec=mem_full_data['DEC']*u.degree,
        frame='icrs',
    )
    off = c.spherical_offsets_to(c2)
    test = (np.abs(off[0].arcsec) <= (pix_size * reso)) & (np.abs(off[1].arcsec) <= (pix_size * reso))
    ids_clus_inside = np.unique(mem_full_data['ID'][test])
    out = []
    for ids in ids_clus_inside:
        g = np.where(mem_full_data['ID'] == ids)[0]
        off = c.spherical_offsets_to(c2[g])
        xs = pix_size - off[0].arcsec / reso
        ys = pix_size - off[1].arcsec / reso
        x1 = int(np.floor(np.min(xs))) - pix_pad
        y1 = int(np.floor(np.min(ys))) - pix_pad
        x2 = int(np.ceil(np.max(xs))) + pix_pad
        y2 = int(np.ceil(np.max(ys))) + pix_pad
        L = min(pix_size*2, x2) - max(0, x1)
        H = min(pix_size*2, y2) - max(0, y1)
        main_clus = 0
        if ids == clu_full_data['ID'][i]:
            main_clus = 1
        if (L<0) | (H<0):
            raise ValueError
        else:
            A = (x2 - x1) * (y2 - y1)
        ix_clu = np.where(clu_full_data['ID'] == ids)[0][0]
        out.append([
            x1, y1, x2, y2,
            L * H / A,
            main_clus,
            clu_full_data['Z_LAMBDA'][ix_clu],
        ])
    return out
if __name__ == '__main__':
    pool = Pool(32)
    res = list(
        tqdm(
            pool.imap(get_boxes, range(n_clu)),
            total=n_clu,
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()

import json
with open("all_bb_2048x2048_0p396_pad50.txt", "w") as fp:
    json.dump(res, fp)
'''
import json
fname = 'all_bb_2048x2048_0p396_pad50.txt'
with open(fname, "r") as fp:
    res = json.load(fp)
'''
################################
########## FOR YOLOV3 ##########
################################

keep = []
for i, r in enumerate(res):
    tmp = np.array(r)
    g = tmp[:, -2] == 1
    conds = []
    conds.append(tmp[g, -3][0] == 1.) # main cluster has to be fully in image
    conds.append(tmp[g, -1][0] > 0.3) # main cluster has to be at z>0.3
    if all(conds):
        keep.append(i)
cut = len(keep) // 2

np.random.seed(42)
np.random.shuffle(keep)

fname_train = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50_zcut0p3/train.txt"
fname_valid = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50_zcut0p3/valid.txt"

for ix in tqdm(keep[:cut], smoothing=0):
    path = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p396127/%s.jpeg" % clu_full_data['ID'][ix]
    str_to_write = ''
    str_to_write += path
    for r in res[ix]:
        if r[-2] >= thres:
            str_to_write += ' %s,%s,%s,%s,0' % (
                max(0, r[0]), max(0, r[1]), min(2048, r[2]), min(2048, r[3]),
            )
    str_to_write += '\n'
    with open(fname_train, 'a') as f:
        f.write(str_to_write)

for ix in tqdm(keep[cut:], smoothing=0):
    path = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p396127/%s.jpeg" % clu_full_data['ID'][ix]
    str_to_write = ''
    str_to_write += path
    for r in res[ix]:
        if r[-2] >= thres:
            str_to_write += ' %s,%s,%s,%s,0' % (
                max(0, r[0]), max(0, r[1]), min(2048, r[2]), min(2048, r[3]),
            )
    str_to_write += '\n'
    with open(fname_valid, 'a') as f:
        f.write(str_to_write)


#####################################
########## FOR keras-yolo3 ##########
#####################################

keep = []
for i, r in enumerate(res):
    tmp = np.array(r)
    g = tmp[:, -1] == 1
    if tmp[g, -2][0] == 1.:
        keep.append(i)

from pascal_voc_writer import Writer
# Writer(path, width, height)
# writer = Writer('path/to/img.jpg', 800, 400)
# ::addObject(name, xmin, ymin, xmax, ymax)
# writer.addObject('cat', 100, 100, 200, 200)
# ::save(path)
# writer.save('path/to/img.xml')

# for ix in tqdm(keep, smoothing=0):
def prep_files(ix):
    path1 = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p396127/%s.jpeg" % clu_full_data['ID'][ix]
    path2 = "/home/users/ilic/ML/ML_clusters_project/keras-yolo3/data/imgs/%s.jpeg" % clu_full_data['ID'][ix]
    os.system('ln -s %s %s' % (path1, path2))
    writer = Writer(path2, 2048, 2048)
    for r in res[ix]:
        if r[-2] >= thres:
            writer.addObject(
                'cluster',
                max(0, r[0]),
                max(0, r[1]),
                min(2048, r[2]),
                min(2048, r[3]),
            )
    path3 = "/home/users/ilic/ML/ML_clusters_project/keras-yolo3/data/annots/%s.xml" % clu_full_data['ID'][ix]
    writer.save(path3)
    return 1
if __name__ == '__main__':
    pool = Pool(32)
    res = list(
        tqdm(
            pool.imap(prep_files, keep),
            total=len(keep),
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()

################################
########## FOR yolov4 ##########
################################

keep = []
for i, r in enumerate(res):
    tmp = np.array(r)
    g = tmp[:, -1] == 1
    if tmp[g, -2][0] == 1.:
        keep.append(i)
cut = len(keep) // 2

np.random.seed(42)
np.random.shuffle(keep)

fname_train = "/home/users/ilic/ML/ML_clusters_project/yolov4/train_2048x2048_0p396_pad50.txt"
fname_valid = "/home/users/ilic/ML/ML_clusters_project/yolov4/valid_2048x2048_0p396_pad50.txt"

for ix in tqdm(keep[:cut], smoothing=0):
    path = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p396127/%s.jpeg" % clu_full_data['ID'][ix]
    str_to_write = ''
    str_to_write += path
    for r in res[ix]:
        if r[-2] >= thres:
            ctr_x = 0.5 * (min(2048, r[2]) + max(0, r[0])) / 2048.
            ctr_y = 0.5 * (min(2048, r[3]) + max(0, r[1])) / 2048.
            w = (min(2048, r[2]) - max(0, r[0])) / 2048.
            h = (min(2048, r[3]) - max(0, r[1])) / 2048.
            str_to_write += ' 0,%s,%s,%s,%s' % (ctr_x, ctr_y, w, h)
    str_to_write += '\n'
    with open(fname_train, 'a') as f:
        f.write(str_to_write)

for ix in tqdm(keep[cut:], smoothing=0):
    path = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p396127/%s.jpeg" % clu_full_data['ID'][ix]
    str_to_write = ''
    str_to_write += path
    for r in res[ix]:
        if r[-2] >= thres:
            ctr_x = 0.5 * (min(2048, r[2]) + max(0, r[0])) / 2048.
            ctr_y = 0.5 * (min(2048, r[3]) + max(0, r[1])) / 2048.
            w = (min(2048, r[2]) - max(0, r[0])) / 2048.
            h = (min(2048, r[3]) - max(0, r[1])) / 2048.
            str_to_write += ' 0,%s,%s,%s,%s' % (ctr_x, ctr_y, w, h)
    str_to_write += '\n'
    with open(fname_valid, 'a') as f:
        f.write(str_to_write)
