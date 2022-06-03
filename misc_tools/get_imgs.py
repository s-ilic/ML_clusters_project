import os
import sys
import time
import numpy as np
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from healpy.projector import GnomonicProj as GP


#################################################################################
#################################################################################


# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read redmapper data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)

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

# Special cases:
'''
spe = [11, 69, 73, 165, 313, 4540]
for s in spe:
    print("Before: RA=%s, DEC=%s" % (clu_full_data['RA'][s], clu_full_data['DEC'][s]))
    g = np.where(mem_full_data['ID'] == clu_full_data['ID'][s])
    offs = c[s].spherical_offsets_to(c2[g])
    delta_ra = 0.5 * (offs[0].deg.min() + offs[0].deg.max())
    delta_dec = 0.5 * (offs[1].deg.min() + offs[1].deg.max())
    clu_full_data['RA'][s] += delta_ra
    clu_full_data['DEC'][s] += delta_dec
    print("After: RA=%s, DEC=%s" % (clu_full_data['RA'][s], clu_full_data['DEC'][s]))
'''


#################################################################################
#################################################################################


# # # # # # # # # # #
# Get images:
# - source : SDSS DR16 ImgCutout service
# - positions : redmapper clusters
# - size : 2048x2048
# - resolution : 0.396127 arcsec/pix (native SDSS)
# # # # # # # # # # #
'''
# for i in tqdm(range(n_clu)[int(sys.argv[1])::32]):
for i in tqdm(range(n_clu)):
    com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" % (clu_full_data['RA'][i], clu_full_data['DEC'][i])
    path = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p396127/%s.jpeg" % clu_full_data['ID'][i]
    if os.path.isfile(path):
        print("%s.jpeg already in new folder" % clu_full_data['ID'][i])
    else:
        os.system('wget -q "%s" -O %s' % (com, path))
'''


# # # # # # # # # # #
# Get images:
# - source : SDSS DR16 ImgCutout service
# - positions : redmapper clusters
# - size : 2048x2048
# - resolution : 0.792254 arcsec/pix (native SDSS x 2)
# # # # # # # # # # #
'''
for i in tqdm(range(n_clu)):
    com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&scale=0.792254&width=2048&height=2048" % (clu_full_data['RA'][i], clu_full_data['DEC'][i])
    path = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p792254/%s.jpeg" % clu_full_data['ID'][i]
    if os.path.isfile(path):
        print("%s.jpeg already in new folder" % clu_full_data['ID'][i])
    else:
        os.system('wget -q "%s" -O %s' % (com, path))
'''


# # # # # # # # # # #
# Get images:
# - source : SDSS DR16 ImgCutout service
# - positions : redmapper clusters
# - size : 2048x2048
# - resolution : 1.188381 arcsec/pix (native SDSS x 3)
# # # # # # # # # # #
'''
for i in tqdm(range(n_clu)):
    com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&scale=1.188381&width=2048&height=2048" % (clu_full_data['RA'][i], clu_full_data['DEC'][i])
    path = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_1p188381/%s.jpeg" % clu_full_data['ID'][i]
    if os.path.isfile(path):
        print("%s.jpeg already in new folder" % clu_full_data['ID'][i])
    else:
        os.system('wget -q "%s" -O %s' % (com, path))
'''


# # # # # # # # # # #
# Get images:
# - source : SDSS DR16 ImgCutout service
# - positions : redmapper clusters
# - size : 2048x2048
# - resolution : 1.584508 arcsec/pix (native SDSS x 4)
# # # # # # # # # # #
'''
for i in tqdm(range(n_clu)):
    com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&scale=1.584508&width=2048&height=2048" % (clu_full_data['RA'][i], clu_full_data['DEC'][i])
    path = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_1p584508/%s.jpeg" % clu_full_data['ID'][i]
    if os.path.isfile(path):
        print("%s.jpeg already in new folder" % clu_full_data['ID'][i])
    else:
        os.system('wget -q "%s" -O %s' % (com, path))
'''


# # # # # # # # # # #
# Get images:
# - source : SDSS DR16 ImgCutout service
# - positions : random within SDSS footprint
# - size : 2048x2048
# - resolution : 0.396127 arcsec/pix (native SDSS)
# # # # # # # # # # #
'''
ct = 0
while ct < 1000:
    ran_ra = np.random.rand() * 360.
    ran_dec = (np.arccos(2. * np.random.rand() - 1.) - np.pi / 2.) * 180. / np.pi
    ran_str = str(np.random.rand())[2:]
    path = "/home/users/ilic/ML/SDSS_image_data/random_2048x2048_0p396127/%s.jpeg" % ran_str
    if os.path.isfile(path):
        continue
    else:
        res = os.system('wget -q "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" -O %s' % (ran_ra, ran_dec, path))
        if res == 0:
            ct += 1
            np.savetxt("/home/users/ilic/ML/SDSS_image_data/random_2048x2048_0p396127/%s.txt" % ran_str, np.array([ran_ra, ran_dec]))
        else:
            os.system('rm %s' % path)
'''


# # # # # # # # # # #
# Get images:
# - source : SDSS DR16 ImgCutout service
# - positions : random but at most 5 deg away and at least 20 arcmin from the nearest redmapper cluster
# - size : 2048x2048
# - resolution : 0.396127 arcsec/pix (native SDSS)
# # # # # # # # # # #
'''
catalog = SkyCoord(ra=clu_full_data['RA']*u.degree, dec=clu_full_data['DEC']*u.degree)
ct = 0
while ct < 1000:
    ran_ra = np.random.rand() * 360.
    ran_dec = (np.arccos(2. * np.random.rand() - 1.) - np.pi / 2.) * 180. / np.pi
    ran_str = str(np.random.rand())[2:]
    path = "/home/users/ilic/ML/SDSS_image_data/empty_2048x2048_0p396127/%s.jpeg" % ran_str
    if os.path.isfile(path):
        continue
    else:
        c = SkyCoord(ra=ran_ra*u.degree, dec=ran_dec*u.degree)
        idx, d2d, d3d = c.match_to_catalog_sky(catalog)
        test = ((np.array(d2d[0]) > (20./60.)) & (np.array(d2d[0]) < (5.)))
        if test:
            res = os.system('wget -q "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" -O %s' % (ran_ra, ran_dec, path))
            if res == 0:
                ct += 1
                np.savetxt("/home/users/ilic/ML/SDSS_image_data/empty_2048x2048_0p396127/%s.txt" % ran_str, np.array([ran_ra, ran_dec]))
            else:
                os.system('rm %s' % path)
'''


# # # # # # # # # # #
# Get images:
# - source : SDSS DR16 ImgCutout service
# - positions : PCCS SZ union clusters
# - size : 2048x2048
# - resolution : 0.396127 arcsec/pix (native SDSS)
# # # # # # # # # # #
'''
pathData="/home/users/ilic/ML/SZ_data/"
hdul = fits.open(pathData+'HFI_PCCS_SZ-union_R2.08.fits')
clu_full_data = hdul[1].data
n_clu = len(clu_full_data)
# for i in tqdm(range(n_clu)[int(sys.argv[1])::32]):
for i in tqdm(range(n_clu)):
    com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" % (clu_full_data['RA'][i], clu_full_data['DEC'][i])
    path = "/home/users/ilic/ML/SDSS_image_data/PSZunion_2048x2048_0p396127/%s.jpeg" % clu_full_data['INDEX'][i]
    if os.path.isfile(path) & (os.path.getsize(path) != 0):
        print("%s.jpeg already in new folder" % clu_full_data['INDEX'][i])
    else:
        os.system('wget -q "%s" -O %s' % (com, path))
'''


# # # # # # # # # # #
# Get images:
# - source : SDSS DR16 ImgCutout service
# - positions : MCXC clusters
# - size : 2048x2048
# - resolution : 0.396127 arcsec/pix (native SDSS)
# # # # # # # # # # #
pathData="/home/users/ilic/ML/other_cats/"
hdul = fits.open(pathData+'mcxc.fits')
clu_full_data = hdul[1].data
n_clu = len(clu_full_data)
# for i in tqdm(range(n_clu)[int(sys.argv[1])::32]):
for i in tqdm(range(n_clu)):
    com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" % (clu_full_data['RAdeg'][i], clu_full_data['DEdeg'][i])
    path = "/home/users/ilic/ML/SDSS_image_data/MCXC_2048x2048_0p396127/%s.jpeg" % i
    test1 = os.path.isfile(path)
    test2 = True
    if test1:
        test2 = os.path.getsize(path) != 0
    if test1 & test2:
        print("%s.jpeg already in new folder" % i)
    else:
        os.system('wget -q "%s" -O %s' % (com, path))


# # # # # # # # # # #
# Get images:
# - source : SDSS DR16 ImgCutout service
# - positions : COMPRASS clusters
# - size : 2048x2048
# - resolution : 0.396127 arcsec/pix (native SDSS)
# # # # # # # # # # #
pathData="/home/users/ilic/ML/other_cats/"
hdul = fits.open(pathData+'comprass.fits')
clu_full_data = hdul[1].data
n_clu = len(clu_full_data)
# for i in tqdm(range(n_clu)[int(sys.argv[1])::32]):
for i in tqdm(range(n_clu)):
    com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" % (clu_full_data['RAJ2000'][i], clu_full_data['DEJ2000'][i])
    path = "/home/users/ilic/ML/SDSS_image_data/COMPRASS_2048x2048_0p396127/%s.jpeg" % i
    test1 = os.path.isfile(path)
    test2 = True
    if test1:
        test2 = os.path.getsize(path) != 0
    if test1 & test2:
        print("%s.jpeg already in new folder" % i)
    else:
        os.system('wget -q "%s" -O %s' % (com, path))
