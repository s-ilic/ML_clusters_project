from astropy.io.fits.convenience import append
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
from itertools import combinations

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


# Coordinates of all clusters
c = SkyCoord(
    ra=clu_full_data['RA']*u.degree,
    dec=clu_full_data['DEC']*u.degree,
    frame='icrs',
)

# Coordinates of all member galaxies
c2 = SkyCoord(
    ra=mem_full_data['RA']*u.degree,
    dec=mem_full_data['DEC']*u.degree,
    frame='icrs',
)

def get_intsec(bb1, bb2):
    x_dist = min(bb1[1], bb2[1]) - max(bb1[0], bb2[0])
    y_dist = min(bb1[3], bb2[3]) - max(bb1[2], bb2[2])
    if (x_dist <= 0) or (y_dist <= 0):
        return 0.
    else:
        return x_dist * y_dist

def get_intsec_bb(bb1, bb2):
    x_min = max(bb1[0], bb2[0])
    x_max = min(bb1[1], bb2[1])
    y_min = max(bb1[2], bb2[2])
    y_max = min(bb1[3], bb2[3])
    if ((x_max - x_min) <= 0) or ((y_max - y_min) <= 0):
        print("Strange!")
    return [x_min, x_max, y_min, y_max]

def get_overlap_fraction(i):
    # Compute offsets of all galaxies from center of cluster "i"
    off = c[i].spherical_offsets_to(c2)

    # Indices of all member galaxies of cluster "i"
    ix = np.where(mem_full_data['ID'] == clu_full_data['ID'][i])[0]

    # Compute bounding box coordinates of cluster "i"
    x_min = np.min(off[0][ix].arcsec) - pix_pad * reso
    x_max = np.max(off[0][ix].arcsec) + pix_pad * reso
    y_min = np.min(off[1][ix].arcsec) - pix_pad * reso
    y_max = np.max(off[1][ix].arcsec) + pix_pad * reso
    bb = [x_min, x_max, y_min, y_max]

    # Grab all member galaxies that fall into that box
    test = (
        (off[0].arcsec >= (x_min - pix_pad * reso)) &
        (off[0].arcsec <= (x_max + pix_pad * reso)) &
        (off[1].arcsec >= (y_min - pix_pad * reso)) &
        (off[1].arcsec <= (y_max + pix_pad * reso))
    )

    # Find the IDs of all the clusters to which those galaxies belong
    ids_clus_inside = np.unique(mem_full_data['ID'][test])

    # If there is only one, the cluster has no overlap
    if len(ids_clus_inside) == 1:
        return 0.

    # Else, make a loop over those clusters to get their bounding boxes
    bbs = []
    for ids in ids_clus_inside:
        if ids != clu_full_data['ID'][i]:
            ix = np.where(mem_full_data['ID'] == ids)[0]
            cx_min = np.min(off[0][ix].arcsec) - pix_pad * reso
            cx_max = np.max(off[0][ix].arcsec) + pix_pad * reso
            cy_min = np.min(off[1][ix].arcsec) - pix_pad * reso
            cy_max = np.max(off[1][ix].arcsec) + pix_pad * reso
            bbs.append([cx_min, cx_max, cy_min, cy_max])

    # Total area of cluster "i" BB overlapped is:
    # - the sum of intersection areas of cluster "i" with above clusters BBs
    # - minus all the intersection between aforementioned intersection areas

    # All intersection BBs (and coresponding areas) with cluster "i"
    iBB = [get_intsec_bb(bb, b) for b in bbs]
    aiBB = [(ibb[1] - ibb[0]) * (ibb[3] - ibb[2]) for ibb in iBB]

    # Areas of intersections between intersections
    aiiBB = [get_intsec(iBB[i], iBB[j]) for i, j in combinations(range(len(iBB)), 2)]

    # Total area of intersection
    A_tot = sum(aiBB) - sum(aiiBB)

    if A_tot < 0.:
        print("Warning: A_tot = %s" % A_tot)

    return A_tot / (x_max - x_min) / (y_max - y_min)






if __name__ == '__main__':
    pool = Pool(32)
    res = list(
        tqdm(
            pool.imap(get_overlap_fraction, range(n_clu)),
            total=n_clu,
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()


############################################################################

def get_overlaps(i):
    # Compute offsets of all galaxies from center of cluster "i"
    off = c[i].spherical_offsets_to(c2)

    # Indices of all member galaxies of cluster "i"
    ix = np.where(mem_full_data['ID'] == clu_full_data['ID'][i])[0]

    # Compute bounding box coordinates of cluster "i"
    x_min = np.min(off[0][ix].arcsec) - pix_pad * reso
    x_max = np.max(off[0][ix].arcsec) + pix_pad * reso
    y_min = np.min(off[1][ix].arcsec) - pix_pad * reso
    y_max = np.max(off[1][ix].arcsec) + pix_pad * reso
    bb = [x_min, x_max, y_min, y_max]

    # Grab all member galaxies that fall into that box
    test = (
        (off[0].arcsec >= (x_min - pix_pad * reso)) &
        (off[0].arcsec <= (x_max + pix_pad * reso)) &
        (off[1].arcsec >= (y_min - pix_pad * reso)) &
        (off[1].arcsec <= (y_max + pix_pad * reso))
    )

    # Find the IDs of all the clusters to which those galaxies belong
    ids_clus_inside = np.unique(mem_full_data['ID'][test])

    # If there is only one, the cluster has no overlap
    if len(ids_clus_inside) == 1:
        return []

    # Else, make a loop over those clusters to get their bounding boxes
    bbs = []
    IDs = []
    for ids in ids_clus_inside:
        if ids != clu_full_data['ID'][i]:
            IDs.append(np.where(clu_full_data['ID'] == ids)[0][0])
            ix = np.where(mem_full_data['ID'] == ids)[0]
            cx_min = np.min(off[0][ix].arcsec) - pix_pad * reso
            cx_max = np.max(off[0][ix].arcsec) + pix_pad * reso
            cy_min = np.min(off[1][ix].arcsec) - pix_pad * reso
            cy_max = np.max(off[1][ix].arcsec) + pix_pad * reso
            bbs.append([cx_min, cx_max, cy_min, cy_max])

    # Total area of cluster "i" BB overlapped is:
    # - the sum of intersection areas of cluster "i" with above clusters BBs
    # - minus all the intersection between aforementioned intersection areas

    # All intersection BBs (and coresponding areas) with cluster "i"
    iBB = [get_intsec_bb(bb, b) for b in bbs]
    aiBB = [(ibb[1] - ibb[0]) * (ibb[3] - ibb[2]) for ibb in iBB]

    out = []
    for ids, aibb in zip(IDs, aiBB):
            out.append(
                [
                    clu_full_data['Z_LAMBDA'][i],
                    clu_full_data['Z_LAMBDA'][ids],
                    aibb / (x_max - x_min) / (y_max - y_min),
                ]
            )

    return out

if __name__ == '__main__':
    pool = Pool(32)
    res = list(
        tqdm(
            pool.imap(get_overlaps, range(n_clu)),
            total=n_clu,
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()
