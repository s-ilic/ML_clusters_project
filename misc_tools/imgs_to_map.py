import os
import healpy as hp
import numpy as np
from tqdm import tqdm
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from healpy.projector import GnomonicProj as GP

# fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)

'''
empty_path = "/home/users/ilic/ML/SDSS_image_data/empty_2048x2048_0p396127"
ra, dec = [], []
nf = len(os.listdir(empty_path))
for ix, fn in tqdm(enumerate(os.listdir(empty_path)),total=nf):
    if '.txt' in fn:
        tmp_ra, tmp_dec = np.loadtxt(empty_path + '/' + fn, unpack=True)
        ra.append(tmp_ra)
        dec.append(tmp_dec)

nside = 2048
m = np.zeros(hp.nside2npix(nside))
for r, d in tqdm(zip(ra, dec)):
    v = hp.ang2vec(r, d, lonlat=True)
    disk = hp.query_disc(2048, v, 1024*0.4/3600.*np.pi/180.)
    m[disk] += 1
hp.mollview(m, xsize=2048)
plt.savefig('empty_map.png', dpi=600)
'''

'''
nside = 2048
m = np.zeros(hp.nside2npix(nside))
for r, d in tqdm(zip(clu_full_data['RA'], clu_full_data['DEC'])):
    v = hp.ang2vec(r, d, lonlat=True)
    disk = hp.query_disc(2048, v, 1024*0.4/3600.*np.pi/180.)
    m[disk] += 1
hp.mollview(m, xsize=2048)
plt.savefig('full_map.png', dpi=600)
'''


m = np.zeros(hp.nside2npix(nside))
poly = np.loadtxt("polys.txt")
poly1 = hp.ang2vec(poly[:12, 0], poly[:12, 1], lonlat=True)
poly2 = hp.ang2vec(poly[12:, 0], poly[12:, 1], lonlat=True)
ixs = hp.query_polygon(
    nside,
    [poly1[i] for i in [-1, 0, 1, 2]],
)
m[ixs] = 1
ixs = hp.query_polygon(
    nside,
    [poly1[i] for i in [-1, 2, 3, 4, 7, 8, 9, 10]],
)
m[ixs] = 1
ixs = hp.query_polygon(
    nside,
    [poly1[i] for i in[4, 5, 6, 7]],
)
m[ixs] = 1
for i in range(len(poly2)):
    i2 = (i+1) % len(poly2)
    for j in range(len(poly2)):
        if (j != i) & (j != i2):
            ixs = hp.query_polygon(
                nside,
                [
                    poly2[i],
                    poly2[i2],
                    poly2[j],
                ],
            )
            m[ixs] = 1
np.savez_compressed("poly_map", m)

for i, v in enumerate(poly1):
    disk = hp.query_disc(2048, v, 1024*0.4/3600.*np.pi/180.*10)
    if i==0:
        m[disk] += 3
    elif i==1:
        m[disk] += 2
    else:
        m[disk] += 1
for i, v in enumerate(poly2):
    disk = hp.query_disc(2048, v, 1024*0.4/3600.*np.pi/180.*10)
    if i==0:
        m[disk] += 3
    elif i==1:
        m[disk] += 2
    else:
        m[disk] += 1

hp.mollview(m, rot=[180, 0])
plt.show()


