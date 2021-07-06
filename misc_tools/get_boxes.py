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
pix_size = 1024.
# pix_size = 2048.
pix_pad = 50
reso = 0.396127

for i in tqdm(range(n_clu), smoothing=0):
    # if (clu_full_data['Z_LAMBDA'][i] >= 0.3) & (clu_full_data['Z_LAMBDA'][i] <= 0.45):
    c = SkyCoord(
        ra=clu_full_data['RA'][i]*u.degree,
        dec=clu_full_data['DEC'][i]*u.degree,
        frame='icrs',
    )
    g = np.where(mem_full_data['ID'] == clu_full_data['ID'][i])[0]

    c2 = SkyCoord(
        ra=mem_full_data['RA'][g]*u.degree,
        dec=mem_full_data['DEC'][g]*u.degree,
        frame='icrs',
    )
    off = c.spherical_offsets_to(c2)
    xs = pix_size - off[0].arcsec / reso
    ys = pix_size - off[1].arcsec / reso
    x1 = int(np.floor(np.min(xs))) - pix_pad
    y1 = int(np.floor(np.min(ys))) - pix_pad
    x2 = int(np.ceil(np.max(xs))) + pix_pad
    y2 = int(np.ceil(np.max(ys))) + pix_pad
    boxes.append([x1,y1,x2,y2])
    # path1 = "/home/users/ilic/ML/SDSS_image_data/train/has_cluster/%s.jpeg" % clu_full_data['ID'][i]
    # path2 = "/home/users/ilic/ML/SDSS_image_data/valid/has_cluster/%s.jpeg" % clu_full_data['ID'][i]
    # path3 = "/home/users/ilic/ML/SDSS_image_data/test/has_cluster/%s.jpeg" % clu_full_data['ID'][i]
    # test = (x1 >= 0) & (x2 < 2048) & (y1 >= 0) & (y2 < 2048)
    # if test:
    #     if os.path.isfile(path1):
    #         with open('train.txt', 'a') as f:
    #             f.write('%s %s,%s,%s,%s,0\n' % (path1,x1,y1,x2,y2))
    #     if os.path.isfile(path2):
    #         with open('valid.txt', 'a') as f:
    #             f.write('%s %s,%s,%s,%s,0\n' % (path2,x1,y1,x2,y2))
    #     if os.path.isfile(path3):
    #         with open('valid.txt', 'a') as f:
    #             f.write('%s %s,%s,%s,%s,0\n' % (path3,x1,y1,x2,y2))

# img = mpimg.imread('7bis.jpeg')
# plt.imshow(img)

# r = 1./np.tan(reso/3600.*np.pi/180.)

# refs = [
#     [54, 1331, 820],
#     [244, 841.133, 1656.53],
#     [39, 810.374, 1291.84],
#     [122, 792.483, 1015.82],
#     [137, 1055, 883],
#     [117, 797, 832],
#     [210, 853, 387],
#     [48, 1350, 666],
#     [2, 1282, 773],
#     [165, 832.193, 1139.91],
# ]

# plt.savefig('toto2.png',dpi=600)
# plt.clf()


#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################


catalog = SkyCoord(ra=clu_full_data['RA']*u.degree, dec=clu_full_data['DEC']*u.degree)
ran_ra = np.random.rand(200000) * 360.
ran_dec = (np.arccos(2. * np.random.rand(200000) - 1.) - np.pi / 2.) * 180. / np.pi
c = SkyCoord(ra=ran_ra*u.degree, dec=ran_dec*u.degree)
idx, d2d, d3d = c.match_to_catalog_sky(catalog)
g = ((np.array(d2d) > (20./60.)) & (np.array(d2d) < (5.)))
ra = ran_ra[g]
dec = ran_dec[g]
ct = 0
for i in range(len(ra)):
    print(ct)
    path = "/home/silic/Downloads/imgs_clusters/randoms/%s.jpeg" % ct
    if os.path.isfile(path):
        ct += 1
    else:
        res = os.system('wget "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" -O %s' % (ra[i], dec[i], path))
        if res == 0:
            np.savetxt("/home/silic/Downloads/imgs_clusters/randoms/%s.txt" % ct, np.array([ra[i], dec[i]]))
            ct += 1
        else:
            os.system('rm %s' % path)
    if ct > 14047:
        break


#################################################################################
#################################################################################


all_ixs = np.arange(n_clu)
all_n = np.zeros(n_clu)
for i in tqdm(range(n_clu), smoothing=0):
    c = SkyCoord(
        ra=clu_full_data['RA'][i]*u.degree,
        dec=clu_full_data['DEC'][i]*u.degree,
        frame='icrs',
    )
    g = all_ixs != i
    c2 = SkyCoord(
        ra=clu_full_data['RA'][g]*u.degree,
        dec=clu_full_data['DEC'][g]*u.degree,
        frame='icrs',
    )
    off = c.spherical_offsets_to(c2)
    test = (np.abs(off[0].arcsec) <= (1024. * reso)) & (np.abs(off[1].arcsec) <= (1024. * reso))
    n = test.sum()
    all_n[i] = n + 1
    # if n > 0.:
    #     print("there is %s clusters in image %s" % (n, i))


#################################################################################
#################################################################################

reso = 0.396127
pix_size = 1024.
# pix_size = 2048.
all_ixs = np.arange(n_clu)
def get_nclus(i):
    c = SkyCoord(
        ra=clu_full_data['RA'][i]*u.degree,
        dec=clu_full_data['DEC'][i]*u.degree,
        frame='icrs',
    )
    g = all_ixs != i
    c2 = SkyCoord(
        ra=clu_full_data['RA'][g]*u.degree,
        dec=clu_full_data['DEC'][g]*u.degree,
        frame='icrs',
    )
    off = c.spherical_offsets_to(c2)
    test = (np.abs(off[0].arcsec) <= (pix_size * reso)) & (np.abs(off[1].arcsec) <= (pix_size * reso))
    n = test.sum()
    return n
if __name__ == '__main__':
    pool = Pool(32)
    all_n = list(
        tqdm(
            pool.imap(get_nclus, range(n_clu)),
            total=n_clu,
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()


#################################################################################
#################################################################################

# reso = 0.396127
reso = 0.792254
pix_size = 1024.
# pix_size = 2048.

all_ixs = np.arange(n_clu)
def get_nclus(i):
    c = SkyCoord(
        ra=clu_full_data['RA'][i]*u.degree,
        dec=clu_full_data['DEC'][i]*u.degree,
        frame='icrs',
    )
    g = np.where(mem_full_data['ID'] != clu_full_data['ID'][i])[0]
    c2 = SkyCoord(
        ra=mem_full_data['RA'][g]*u.degree,
        dec=mem_full_data['DEC'][g]*u.degree,
        frame='icrs',
    )
    off = c.spherical_offsets_to(c2)
    test = (np.abs(off[0].arcsec) <= (pix_size * reso)) & (np.abs(off[1].arcsec) <= (pix_size * reso))
    ids_clus_inside = np.unique(mem_full_data['ID'][g][test])
    out = []
    for ids in ids_clus_inside:
        g = np.where(mem_full_data['ID'] == ids)[0]
        c2 = SkyCoord(
            ra=mem_full_data['RA'][g]*u.degree,
            dec=mem_full_data['DEC'][g]*u.degree,
            frame='icrs',
        )
        off = c.spherical_offsets_to(c2)
        xs = pix_size - off[0].arcsec / reso
        ys = pix_size - off[1].arcsec / reso
        x1 = int(np.floor(np.min(xs)))
        y1 = int(np.floor(np.min(ys)))
        x2 = int(np.ceil(np.max(xs)))
        y2 = int(np.ceil(np.max(ys)))
        L = min(pix_size*2, x2) - max(0, x1)
        H = min(pix_size*2, y2) - max(0, y1)
        if (L<0) | (H<0):
            out.append(0.)
        else:
            A = (x2 - x1) * (y2 - y1)
            out.append(L * H / A)
    return out
if __name__ == '__main__':
    pool = Pool(32)
    all_n = list(
        tqdm(
            pool.imap(get_nclus, range(n_clu)),
            total=n_clu,
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()


#################################################################################
#################################################################################


# reso = 0.396127
reso = 0.792254
pix_size = 1024.
def get_box(i):
    c = SkyCoord(
        ra=clu_full_data['RA'][i]*u.degree,
        dec=clu_full_data['DEC'][i]*u.degree,
        frame='icrs',
    )
    g = np.where(mem_full_data['ID'] == clu_full_data['ID'][i])[0]
    c2 = SkyCoord(
        ra=mem_full_data['RA'][g]*u.degree,
        dec=mem_full_data['DEC'][g]*u.degree,
        frame='icrs',
    )
    off = c.spherical_offsets_to(c2)
    xs = pix_size - off[0].arcsec / reso
    ys = pix_size - off[1].arcsec / reso
    x1 = int(np.floor(np.min(xs)))
    y1 = int(np.floor(np.min(ys)))
    x2 = int(np.ceil(np.max(xs)))
    y2 = int(np.ceil(np.max(ys)))
    return [x1,y1,x2,y2]
if __name__ == '__main__':
    pool = Pool(32)
    res = list(
        tqdm(
            pool.imap(get_box, range(n_clu)),
            total=n_clu,
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()
res = np.array(res)
test = np.any(res < 0., axis=1) | np.any(res > 2048., axis=1)

# Among the 26111 redmapper clusters:
# - 1390 (5.32%) are not fully inside the 2048x2048 0.396'' pictures
# - 9 (0.0345%) are not fully inside the 2048x2048 0.792'' pictures

# Among the 26111 redmapper clusters:
# - 18242 (69.9%) have 0 other box in the 2048x2048 0.396'' pictures
# - 6340 (24.3%) have 1 other box in the 2048x2048 0.396'' pictures
# - 1287 (4.93%) have 2 other box in the 2048x2048 0.396'' pictures
# - 210/25/6/1 have 3/4/5/6 other box in the 2048x2048 0.396'' pictures

# Among the 26111 redmapper clusters:
# - 9626 (36.9%) have 0 other box in the 2048x2048 0.792'' pictures
# - 8718 (33.4%) have 1 other box in the 2048x2048 0.792'' pictures
# - 4630 (17.7%) have 2 other box in the 2048x2048 0.792'' pictures
# - 2111 (8.08%) have 3 other box in the 2048x2048 0.792'' pictures
# - 701 (2.68%)  have 4 other box in the 2048x2048 0.792'' pictures
# - 227/70/22/4/0/2 have 5/6/7/8/9/10 other box in the 2048x2048 0.792'' pictures

9626 36.865688790165066
8718 33.38822718394546
4630 17.731990348895103
2111 8.084715254107465
701 2.684692275286278
227 0.8693654015549002
70 0.26808624717551993
22 0.08425567768373482
4 0.015319214124315424
0 0.0
2 0.007659607062157712
