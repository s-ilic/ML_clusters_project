import os, sys
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from multiprocessing import Pool


##########################
# COMPUTE BOUNDING BOXES #
##########################

# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)
IDs, ixs, cts = np.unique(mem_full_data['ID'], return_index=True, return_counts=True)
imc = {ID:np.arange(ix,ix+ct) for ID,ix,ct in zip(IDs,ixs,cts)}
ics = {ID:i for i, ID in enumerate(clu_full_data['ID'])}

# All clusters sky coordinates
c = SkyCoord(
    ra=clu_full_data['RA']*u.degree,
    dec=clu_full_data['DEC']*u.degree,
    frame='icrs',
)

# All member galaxies coordinates
c2 = SkyCoord(
    ra=mem_full_data['RA']*u.degree,
    dec=mem_full_data['DEC']*u.degree,
    frame='icrs',
)

'''
# Fix special cases:
spe = [11, 69, 73, 165, 313, 4540]
for s in spe:
    ID = clu_full_data['ID'][s]
    offs = c[s].spherical_offsets_to(c2[imc[ID]])
    delta_ra = 0.5 * (offs[0].deg.min() + offs[0].deg.max())
    delta_dec = 0.5 * (offs[1].deg.min() + offs[1].deg.max())
    clu_full_data['RA'][s] += delta_ra
    clu_full_data['DEC'][s] += delta_dec
'''

# All clusters sky coordinates again after fix
c = SkyCoord(
    ra=clu_full_data['RA']*u.degree,
    dec=clu_full_data['DEC']*u.degree,
    frame='icrs',
)

# Parameters:
# - input images resolution in arcsec/pix
reso = 0.396127
# reso = 0.792254
# - half width of input images in pix
half_width = 2048 // 2
# - padding around bbox in pix
pad = 0


# Function for computing bounding boxes of all clusters present in given image
def get_boxes(i):

    # Angular separation of all member galaxies around ith cluster
    seps = c[i].separation(c2)

    # First pruning to keep only galaxies in disk of radius 1.5x the image width
    g = np.where(seps.arcsec < (1.5 * half_width * reso))[0]

    # Spherical offsets of those galaxies around current cluster
    offs = c[i].spherical_offsets_to(c2[g])

    # Testing if the galaxies are in the image
    test = (np.abs(offs[0].arcsec) <= (half_width * reso)) & (np.abs(offs[1].arcsec) <= (half_width * reso))

    # Grab ids of clusters to which belong the galaxies inside the image
    IDs_clus_inside = np.unique(mem_full_data['ID'][g[test]])

    # For each cluster in the image (at least partially), compute:
    # - bounding box coordinates
    # - area of box inside image
    # - if cluster is the main one at the center
    # - (lambda) redshift of cluster
    out = []
    for ID in IDs_clus_inside:
        # Check if it is the main cluster of the image
        is_main_clus = ID == clu_full_data['ID'][i]
        # Grab correct offsets
        offs = c[i].spherical_offsets_to(c2[imc[ID]])
        # Compute their positions in the image in pixels
        xs = half_width - offs[0].arcsec / reso
        ys = half_width - offs[1].arcsec / reso
        # Compute bbox coordinates
        x1 = int(np.floor(np.min(xs))) - pad
        y1 = int(np.floor(np.min(ys))) - pad
        x2 = int(np.ceil(np.max(xs))) + pad
        y2 = int(np.ceil(np.max(ys))) + pad
        # Return bbox coords + cluster index
        out.append([is_main_clus, ID, x1, y1, x2, y2])
    return out

# Main loop with parallelization
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


# Dump results in a file
'''
import json
with open("all_bb_2048x2048_0p396_pad50.txt", "w") as fp:
    json.dump(res, fp)
'''

# Load results from file
'''
import json
fname = 'all_bb_2048x2048_0p396_pad50.txt'
with open(fname, "r") as fp:
    res = json.load(fp)
'''


############################
# MAKE BB FILES FOR YOLOV3 #
############################

# Input/output folder and files names
#root = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50_zcut0p3"
#root = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50_zbins"
# root = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x2048_ds4_0p396_pad50_zbins_ovl"
# root = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x0p396_ds1_mb32_nopad_A"
root = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/"
root += "2048x0p396_ds4_mb8_nopad_nocl"
fname_train = root + "/train.txt"
fname_valid = root + "/valid.txt"
imgs_path = '/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p396127'
# imgs_path = '/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p792254'
# imgs_path = '/home/users/ilic/ML/SDSS_image_data/redmapper_2016x2016_0p396127'

# First checks on the main cluster in the image
keep = []
for i, r in enumerate(res):
    for clus in r:
        if clus[0]:
            # Some variables
            ix, x1, y1, x2, y2 = clus[1:]
            L = x2 - x1
            H = y2 - y1
            fL = min(2 * half_width, x2) - max(0, x1)
            fH = min(2 * half_width, y2) - max(0, y1)
            # List of conditions to apply:
            conds = [True]
            # - main cluster has to be fully in image
            # conds.append((L == fL) & (H == fH))
            # - main cluster has to be at z>0.3
            #conds.append(clu_full_data['Z_LAMBDA'][ix] > 0.3)
            if all(conds):
                keep.append(i)
            # else:
            #     if not ((L > 2 * half_width) | (H > 2 * half_width)):
            #         print(i, L, H, fL * fH / L / H)
            #     else:
            #         print("Cluster %s is out!" % i)
cut = len(keep) // 2

# Shuffle for good measure
np.random.seed(42)
np.random.shuffle(keep)

# For z binning
# z_bins = np.linspace(0.12, 0.6, 9)
# z_mid = np.linspace(0.15, 0.6, 10)

# For variable regression
# regvar = clu_full_data['Z_LAMBDA']
# regvarerr = clu_full_data['Z_LAMBDA_ERR']
# g = clu_full_data['Z_SPEC'] != -1.
# regvar[g] = clu_full_data['Z_SPEC'][g]
# regvarerr[g] = 0.001
# me = regvar.mean()
# st = regvar.std()
# regvar = (regvar - me) / st
# regvarerr = regvarerr / st


# Main loop which writes the training and validation file
for i, ix in enumerate(keep):
    path = "%s/%s.jpeg" % (imgs_path, clu_full_data['ID'][ix])
    str_to_write = ''
    str_to_write += path
    for clus in res[ix]:
        # Some variables
        ID, x1, y1, x2, y2 = clus[1:]
        L = x2 - x1
        H = y2 - y1
        fL = min(2 * half_width, x2) - max(0, x1)
        fH = min(2 * half_width, y2) - max(0, y1)
        # List of conditions to apply to clusters:
        conds = [True]
        # - cluster has to fit in an image
        # conds.append(max(L, H) <= 2 * half_width)
        # - cluster bbox center is in the image
        conds.append(
            (0 < (0.5 * (x1 + x2)) < (2 * half_width)) &
            (0 < (0.5 * (y1 + y2)) < (2 * half_width))
        )
        if all(conds):
            # class_num = 0
            # class_num = np.digitize(clus[7], z_bins) - 1
            # class_num = np.abs(z_mid-clus[7]).argmin()
            str_to_write += ' %s,%s,%s,%s,%s' % (
            # str_to_write += ' %s,%s,%s,%s,%s,%s' % (
                # max(0, x1),
                # max(0, y1),
                # min(2 * half_width, x2),
                # min(2 * half_width, y2),
                x1, y1, x2, y2,
                # class_num,
                # regvar[ics[ID]], regvarerr[ics[ID]],
                fL * fH / L / H,
            )
    str_to_write += '\n'
    with open(fname_train if i < cut else fname_valid, 'a') as f:
        f.write(str_to_write)

# Summary of cluster rules
with open(root + "/which_clusters.txt", 'w') as f:
    f.write("Main cluster:\n")
    # f.write("- has to be fully in image\n")
    f.write("- no rules\n")
    f.write("\n")
    f.write("Secondary cluster(s):\n")
    # f.write("-  has to be able to fit in an image\n")
    f.write("-  cluster center is in the image\n")
    f.write("\n")
    # f.write("Reg var 1 - redshift:\n")
    # f.write("-  mean = %s\n" % me)
    # f.write("-  std = %s\n" % st)


### Add empty imgs
import healpy as hp
from tqdm import tqdm

n_want = 13056
#12203
out_fname = "/home/users/ilic/ML/ML_clusters_project/YOLOV3/runs/2048x0p396_ds4_mb8_nopad_nocl/valid_empty.txt"
nside = 2048
m = np.load('poly_map.npz')['arr_0']

# !!! only bad file : '39719858979976774.txt'

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
        if (g == 1.) & (fn != '39719858979976774.txt'):
            ct += 1
            f.write(empty_path + '/' + fn[:-3] + 'jpeg\n')
        if ct == n_want:
            break



'''
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
'''