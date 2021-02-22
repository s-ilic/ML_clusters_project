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

# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)

'''
for i in tqdm(range(n_clu)[int(sys.argv[1]):][::int(sys.argv[2])]):
    if (clu_full_data['Z_LAMBDA'][i] >= 0.3) & (clu_full_data['Z_LAMBDA'][i] <= 0.45):
        com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" % (clu_full_data['RA'][i], clu_full_data['DEC'][i])
        path = "/home/silic/Downloads/imgs_clusters/%s.jpeg" % clu_full_data['ID'][i]
        if not os.path.isfile(path):
            os.system('wget "%s" -O %s' % (com, path))
'''

'''
# for i in tqdm(range(n_clu)[::-1][int(sys.argv[1])::10]):
for i in tqdm(range(n_clu)):
    com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" % (clu_full_data['RA'][i], clu_full_data['DEC'][i])
    path1 = "/home/users/ilic/ML/SDSS_image_data/train/has_cluster/%s.jpeg" % clu_full_data['ID'][i]
    path2 = "/home/users/ilic/ML/SDSS_image_data/valid/has_cluster/%s.jpeg" % clu_full_data['ID'][i]
    path3 = "/home/users/ilic/ML/SDSS_image_data/new/%s.jpeg" % clu_full_data['ID'][i]
    if os.path.isfile(path1):
        # print("%s.jpeg already in train folder" % clu_full_data['ID'][i])
        toto = 0
    elif os.path.isfile(path2):
        # print("%s.jpeg already in valid folder" % clu_full_data['ID'][i])
        toto = 0
    elif os.path.isfile(path3):
        # print("%s.jpeg already in new folder" % clu_full_data['ID'][i])
        toto = 0
    else:
        os.system('wget -q "%s" -O %s' % (com, path3))
'''

for i in tqdm(range(n_clu)[int(sys.argv[1])::32]):
    com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&scale=0.792254&width=2048&height=2048" % (clu_full_data['RA'][i], clu_full_data['DEC'][i])
    path = "/home/users/ilic/ML/SDSS_image_data/redmapper_2048x2048_0p792254/%s.jpeg" % clu_full_data['ID'][i]
    if os.path.isfile(path):
        print("%s.jpeg already in new folder" % clu_full_data['ID'][i])
    else:
        os.system('wget "%s" -O %s' % (com, path))

sys.exit()

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