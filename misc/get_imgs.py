import numpy as np
from astropy.io import fits

from tqdm import tqdm

from healpy.projector import GnomonicProj as GP

import os, sys

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy import units as u

#################################################################################

# Path to fits files
pathData="/home/silic/Downloads/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)

for i in tqdm(range(n_clu)[int(sys.argv[1]):][::int(sys.argv[2])]):
    if (clu_full_data['Z_LAMBDA'][i] >= 0.3) & (clu_full_data['Z_LAMBDA'][i] <= 0.45):
        com = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=%s&dec=%s&width=2048&height=2048" % (clu_full_data['RA'][i], clu_full_data['DEC'][i])
        path = "/home/silic/Downloads/imgs_clusters/%s.jpeg" % clu_full_data['ID'][i]
        if not os.path.isfile(path):
            os.system('wget "%s" -O %s' % (com, path))


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