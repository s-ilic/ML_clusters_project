import numpy as np
from astropy.io import fits
from tqdm import tqdm
from healpy.projector import GnomonicProj as GP
import os, sys
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.image as mpimg


#################################################################################

# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)

reso = 0.4 # arcsec/pix ACTUALLY 0.396 # ACTUALLY ACTUALLY 0.396127

boxes = []

for i in tqdm(range(n_clu), smoothing=0):
    if (clu_full_data['Z_LAMBDA'][i] >= 0.3) & (clu_full_data['Z_LAMBDA'][i] <= 0.45):
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
        xs = 1024 - off[0].arcsec / reso * 1.01
        ys = 1024 - off[1].arcsec / reso * 1.01
        x1 = int(np.floor(np.min(xs))) - 50
        y1 = int(np.floor(np.min(ys))) - 50
        x2 = int(np.ceil(np.max(xs))) + 50
        y2 = int(np.ceil(np.max(ys))) + 50
        boxes.append([x1,y1,x2,y2])
        path1 = "/home/users/ilic/ML/SDSS_image_data/train/has_cluster/%s.jpeg" % clu_full_data['ID'][i]
        path2 = "/home/users/ilic/ML/SDSS_image_data/valid/has_cluster/%s.jpeg" % clu_full_data['ID'][i]
        path3 = "/home/users/ilic/ML/SDSS_image_data/test/has_cluster/%s.jpeg" % clu_full_data['ID'][i]
        test = (x1 >= 0) & (x2 < 2048) & (y1 >= 0) & (y2 < 2048)
        if test:
            if os.path.isfile(path1):
                with open('train.txt', 'a') as f:
                    f.write('%s %s,%s,%s,%s,0\n' % (path1,x1,y1,x2,y2))
            if os.path.isfile(path2):
                with open('valid.txt', 'a') as f:
                    f.write('%s %s,%s,%s,%s,0\n' % (path2,x1,y1,x2,y2))
            if os.path.isfile(path3):
                with open('valid.txt', 'a') as f:
                    f.write('%s %s,%s,%s,%s,0\n' % (path3,x1,y1,x2,y2))

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