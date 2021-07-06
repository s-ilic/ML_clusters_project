http://skyserver.sdss.org/dr16/SkyServerWS/ImagingQuery/Rectangular?limit=500000&raMin=352.98439859114563&raMax=353.2097508400345&decMin=16.9832627593305&decMax=17.20861500821939&magType=model&imgparams=objid,ra,dec,u,g,r,i,z,type&format=csv

http://skyserver.sdss.org/dr8/en/tools/search/x_rect.asp?min_ra=352.98439859114563&max_ra=353.2097508400345&min_dec=16.9832627593305&max_dec=17.20861500821939&format=csv&topnum=50000

http://skyserver.sdss.org/dr8/en/tools/search/x_rect.asp?min_ra=352.998&max_ra=353.196&min_dec=16.996&max_dec=17.194&format=csv&topnum=50000

http://skyserver.sdss.org/dr16/SkyServerWS/ImagingQuery/Rectangular?limit=500000&raMin=352.98439859114563&raMax=353.2097508400345&decMin=16.9832627593305&decMax=17.20861500821939&imgparams=objid,ra,dec,modelMag_u,modelMag_g,modelMag_r,cModelMag_i,modelMag_z,type&format=csv

http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch?cmd=+SELECT+TOP+500000+%0D%0A+objID%2C+ra+%2Cdec+%0D%0A+FROM+%0D%0A+PhotoPrimary+%0D%0A+WHERE+%0D%0A+objID+%3D+1237671939804562353%0D%0A+OR+objID+%3D+1237671939804561886&format=json


with open("sql.txt", "w") as f:
    f.write("SELECT TOP 500000\n")
    f.write("objID, dered_u\n")
    f.write("FROM\n")
    f.write("PhotoPrimary\n")
    f.write("WHERE\n")
    f.write("objID = 123\n")
    for i in tqdm(range(400000)):
        f.write("OR objID = %s\n" % mem_full_data['OBJID'][i])


 SELECT TOP 500000 
 objID, dered_u 
 FROM 
 PhotoPrimary 
 WHERE 
 objID = 123
 OR objID = 1237661358071612005 OR objID = 1237661358071612011 OR objID = 1237661358071612173 OR objID = 1237661358071612175 OR objID = 1237661358071612186 OR objID = 1237661358071612196 OR objID = 1237661358071612198 OR objID = 1237661358071612216 OR objID = 1237661358071612221 OR objID = 1237661358071612223 OR objID = 1237661358071612235 OR objID = 1237661358071611455 OR objID = 1237661358071611887 OR objID = 1237661358071611934 OR objID = 1237661358071611975 OR objID = 1237661358071611996 OR objID = 1237661358071612177 OR objID = 1237661358071612193 OR objID = 1237661358071611539 OR objID = 1237661358071677264 OR objID = 1237661358071677287 OR objID = 1237661358071677340 OR objID = 1237661358071677347 OR objID = 1237661358071677653 OR objID = 1237661358071677678 OR objID = 1237661358071677690 OR objID = 1237661358071611980 OR objID = 1237661358071612203 OR objID = 1237661358071612231 OR objID = 1237661358071612238 OR objID = 1237661358071677284 OR objID = 1237661358071677646 OR objID = 1237661358071677652 OR objID = 1237661358071677133 OR objID = 1237661358071677253 OR objID = 1237661358071677263 OR objID = 1237661358071677293 OR objID = 1237661358071677296 OR objID = 1237661358071677306 OR objID = 1237667782814270015 OR objID = 1237667782814270016 OR objID = 1237667782814270042 OR objID = 1237667782814270056 OR objID = 1237667782814270059 OR objID = 1237667782814270060 OR objID = 1237667782814270068 OR objID = 1237667782814270080 OR objID = 1237667782814270096 OR objID = 1237667782814270399 OR objID = 1237667782814270401 OR objID = 1237667782814270408 OR objID = 1237667782814270414 OR objID = 1237667782814270441 OR objID = 1237667782814270459 OR objID = 1237667782814270463 OR objID = 1237667782814270473 OR objID = 1237670449448157676 OR objID = 1237670449448157686 OR objID = 1237667782814269490 OR objID = 1237667782814269492 OR objID = 1237667782814269496 OR objID = 1237667782814269497 OR objID = 1237667782814269737 OR objID = 1237667782814270083 OR objID = 1237667782814270176 OR objID = 1237667782814270179 OR objID = 1237667782814270192 OR objID = 1237667782814270193 OR objID = 1237667782814270215 OR objID = 1237667782814270228 OR objID = 1237667782814270247 OR objID = 1237667782814270277 OR objID = 1237667782814270465 OR objID = 1237667782814270501 OR objID = 1237667782814270507 OR objID = 1237667782814270533 OR objID = 1237670449448158450 OR objID = 1237670449448158060 OR objID = 1237670449448158071 OR objID = 1237670449448157832 OR objID = 1237670449448158102 OR objID = 1237670449448158135 OR objID = 1237670449448158138 OR objID = 1237670449448158143 OR objID = 1237670449448157757



import urllib.request, json, time

n_tot = len(mem_full_data)
ix_start = 0
ix_end = 200
all_dered_u = []
while True:
    print(ix_end)
    sql = "http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch?cmd=+SELECT+TOP+500000+%0D%0A+objID%2C+dered_u+%0D%0A+FROM+%0D%0A+PhotoPrimary+%0D%0A+WHERE+%0D%0A+objID+%3D+123%0D%0A"
    for i in range(ix_start, ix_end):
        sql += "+OR+objID+%3D"
        sql += "+%s" % mem_full_data['OBJID'][i]
    sql += "&format=json"
    with urllib.request.urlopen(sql) as url:
        data = json.loads(url.read().decode())
    objID = [row['objID'] for row in data[0]['Rows']]
    dered_u = [row['dered_u'] for row in data[0]['Rows']]
    for i in range(ix_start, ix_end):
        try:
            all_dered_u.append(dered_u[objID.index(mem_full_data['OBJID'][i])])
        except:
            all_dered_u.append(-1.)
    if ix_end == n_tot:
        break
    else:
        ix_start = ix_end
        ix_end += min(200, n_tot - ix_end)
    time.sleep(0.5)




time.sleep(2)

for i in tqdm(range(200)):
    sql = "http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch?cmd=+SELECT+TOP+500000+%0D%0A+objID%2C+dered_u+%0D%0A+FROM+%0D%0A+PhotoPrimary+%0D%0A+WHERE+%0D%0A+objID+%3D+"
    sql += "%s" % mem_full_data['OBJID'][i]
    ""
    sql += "&format=json"
    with urllib.request.urlopen(sql) as url:
        data = json.loads(url.read().decode())



modelMag_r

t = np.genfromtxt(
    'test_sql.txt',
    delimiter=',',
    names=True,
    dtype=[
        ('OBJID', '>i8'),
        ('RA', '>f8'),
        ('DEC', '>f8'),
        ('U', '>f4'),
        ('G', '>f4'),
        ('R', '>f4'),
        ('I', '>f4'),
        ('Z', '>f4'),
        ('TYPE', '>i4'),
    ]
)

t2 = np.genfromtxt(
    'result.csv',
    delimiter=',',
    names=True,
    dtype=[
        ('objid', '>i8'),
        ('run', '>i4'),
        ('rerun', '>i4'),
        ('camcol', '>i4'),
        ('field', '>i4'),
        ('obj', '>i4'),
        ('type', '>i4'),
        ('ra', '>f4'),
        ('dec', '>f4'),
        ('u', '>f4'),
        ('g', '>f4'),
        ('r', '>f4'),
        ('i', '>f4'),
        ('z', '>f4'),
        ('Err_u', '>f4'),
        ('Err_g', '>f4'),
        ('Err_r', '>f4'),
        ('Err_i', '>f4'),
        ('Err_z', '>f4'),
    ]
)


##############################################################################

from SciServer import CasJobs
from SciServer import Authentication

Authentication_loginName = 'silic';
Authentication_loginPassword = "t\ke5FwR%uKH_;S:L?{RGP$fEN:m#Kv\\"
token1 = Authentication.login(Authentication_loginName, Authentication_loginPassword);
token2 = Authentication.getToken()
token3 = Authentication.getKeystoneToken()
token4 = Authentication.token.value

user = Authentication.getKeystoneUserWithToken(token1)
print("userName=" + user.userName)
print("id=" + user.id)
iden = Authentication.identArgIdentifier()
print("ident="+iden)

import os, sys, gc
import numpy as np
from tqdm import tqdm
from astropy.io import fits
# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"
# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
# clu_full_data = hdul1[1].data
clu_full_data = {}
for n in hdul1[1].data.dtype.names:
    clu_full_data[n] = hdul1[1].data[n]
ras = clu_full_data['RA'] / 180. * np.pi
decs = clu_full_data['DEC'] / 180. * np.pi
clu_full_data['X'] = np.cos(decs) * np.cos(ras)
clu_full_data['Y'] = np.cos(decs) * np.sin(ras)
clu_full_data['Z'] = np.sin(decs)
clu_full_data['M500'] = np.zeros(len(hdul1[1].data))
clu_full_data['M500_ERR_LOW'] = np.zeros(len(hdul1[1].data))
clu_full_data['M500_ERR_UPP'] = np.zeros(len(hdul1[1].data))
# mem_full_data = hdul2[1].data
mem_full_data = {}
for n in hdul2[1].data.dtype.names:
    mem_full_data[n] = hdul2[1].data[n]
ras = mem_full_data['RA'] / 180. * np.pi
decs = mem_full_data['DEC'] / 180. * np.pi
mem_full_data['X'] = np.cos(decs) * np.cos(ras)
mem_full_data['Y'] = np.cos(decs) * np.sin(ras)
mem_full_data['Z'] = np.sin(decs)
print("Variables in cluster catalog:")
print(clu_full_data.keys())
print("Variables in member catalog:")
print(mem_full_data.keys())

########

CasJobs_TestDatabase = "MyDB"
CasJobs_TestTableName1 = "MyTable_A"

CasJobs_TestQuery = "SELECT TOP 500000 objID, ra, dec, dered_u, dered_i, dered_g, dered_r, dered_z into mydb.%s from PhotoPrimary WHERE " % CasJobs_TestTableName1
for i in tqdm(range(10000)):
    if i == 0:
        CasJobs_TestQuery += 'objID = %s' % mem_full_data['OBJID'][i]
    else:
        CasJobs_TestQuery += ' OR objID = %s' % mem_full_data['OBJID'][i]
jobId = CasJobs.submitJob(sql=CasJobs_TestQuery, context="DR16")
jobDescription = CasJobs.waitForJob(jobId=jobId, verbose=False)
#print(jobId)
#print(jobDescription)
array = CasJobs.getNumpyArrayFromQuery(queryString=CasJobs_TestQuery, context=CasJobs_TestDatabase)
print(array)


# delete table
df = CasJobs.executeQuery(
    sql="DROP TABLE " + CasJobs_TestTableName1,
    context="MyDB",
    format="pandas",
)

# get numpy array containing the results of a query
array = CasJobs.getNumpyArrayFromQuery(
    queryString=CasJobs_TestQuery,
    context="DR16",
)
print(array)

#########

from SciServer import SkyServer

n_obj = 100
n_tot = len(mem_full_data['OBJID'])
ct = 0

final_arr = -np.ones((n_tot, 7))

while ct < n_tot:
    CasJobs_TestQuery = "select top %s objid, ra, dec, dered_u, dered_i, dered_g, dered_r, dered_z from photoprimary where " % n_obj
    ct_up = min(ct + n_obj, n_tot)
    for i in range(ct, ct_up):
        if i == ct:
            CasJobs_TestQuery += 'objID = %s' % mem_full_data['OBJID'][i]
        else:
            CasJobs_TestQuery += ' or objID = %s' % mem_full_data['OBJID'][i]
    # df = SkyServer.sqlSearch(sql=CasJobs_TestQuery, dataRelease="DR16")
    df = SkyServer.sqlSearch(sql=CasJobs_TestQuery, dataRelease="DR8")
    for idx, objid in enumerate(mem_full_data['OBJID'][ct:ct_up]):
        ix = np.where(df["objid"] == objid)[0]
        if len(ix) == 1:
            final_arr[ct+idx, :] = df.to_numpy()[ix[0], 1:]
    #########
    ct = ct_up
    print(ct, n_tot)


#######################################################################

import os, sys, gc
import numpy as np
from tqdm import tqdm
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt

# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"
# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
# clu_full_data = hdul1[1].data
clu_full_data = {}
for n in hdul1[1].data.dtype.names:
    clu_full_data[n] = hdul1[1].data[n]
ras = clu_full_data['RA'] / 180. * np.pi
decs = clu_full_data['DEC'] / 180. * np.pi
clu_full_data['X'] = np.cos(decs) * np.cos(ras)
clu_full_data['Y'] = np.cos(decs) * np.sin(ras)
clu_full_data['Z'] = np.sin(decs)
clu_full_data['M500'] = np.zeros(len(hdul1[1].data))
clu_full_data['M500_ERR_LOW'] = np.zeros(len(hdul1[1].data))
clu_full_data['M500_ERR_UPP'] = np.zeros(len(hdul1[1].data))
# mem_full_data = hdul2[1].data
mem_full_data = {}
for n in hdul2[1].data.dtype.names:
    mem_full_data[n] = hdul2[1].data[n]
ras = mem_full_data['RA'] / 180. * np.pi
decs = mem_full_data['DEC'] / 180. * np.pi
mem_full_data['X'] = np.cos(decs) * np.cos(ras)
mem_full_data['Y'] = np.cos(decs) * np.sin(ras)
mem_full_data['Z'] = np.sin(decs)
print("Variables in cluster catalog:")
print(clu_full_data.keys())
print("Variables in member catalog:")
print(mem_full_data.keys())
tab = np.load('final_arr.npz')['final_arr']

g = tab[:, -1] != -1.
plt.figure(figsize=(20, 5))
for i, l in enumerate('UIGRZ'):
    print(i, l)
    mi = min(min(mem_full_data['MODEL_MAG_%s' % l][g]), min(tab[:, 2+i][g]))
    ma = max(max(mem_full_data['MODEL_MAG_%s' % l][g]), max(tab[:, 2+i][g]))
    plt.subplot(2, 5, i +1)
    plt.title("%s band mag" % l)
    plt.hist2d(mem_full_data['MODEL_MAG_%s' % l][g], tab[:, 2+i][g], bins=64, range=[[mi, ma], [mi, ma]])
    plt.plot([mi, ma], [mi, ma], color='red')
    plt.xlabel("redmapper cat")
    plt.ylabel("DR16 cat")
    plt.colorbar()
    plt.subplot(2, 5, i +1 + 5)
    plt.title("%s band mag"% l)
    plt.hist2d(mem_full_data['MODEL_MAG_%s' % l][g], tab[:, 2+i][g], bins=64, range=[[mi, ma], [mi, ma]], norm=matplotlib.colors.LogNorm())
    plt.plot([mi, ma], [mi, ma], color='red')
    plt.xlabel("redmapper cat")
    plt.ylabel("DR16 cat")
    plt.colorbar()
plt.tight_layout()
plt.savefig("phot.pdf")
plt.figure(figsize=(20, 5))
for i, l in enumerate('UIGRZ'):
    print(i, l)
    mi = min(min(mem_full_data['MODEL_MAG_%s' % l][g]), min(tab[:, 2+i][g]))
    ma = max(max(mem_full_data['MODEL_MAG_%s' % l][g]), max(tab[:, 2+i][g]))
    plt.subplot(2, 5, i +1)
    plt.title("%s band mag" % l)
    plt.hist(mem_full_data['MODEL_MAG_%s' % l][g] - tab[:, 2+i][g], bins=64)
    plt.xlabel("redmapper - DR16")
    plt.subplot(2, 5, i +1 + 5)
    plt.title("%s band mag"% l)
    plt.hist(mem_full_data['MODEL_MAG_%s' % l][g] - tab[:, 2+i][g], bins=64)
    plt.yscale('log')
    plt.xlabel("redmapper - DR16")
plt.tight_layout()
plt.savefig("phot.pdf")

#######################################

import corner
st = np.vstack(
    (
        mem_full_data["MODEL_MAG_U"],
        mem_full_data["MODEL_MAG_I"],
        mem_full_data["MODEL_MAG_G"],
        mem_full_data["MODEL_MAG_R"],
        mem_full_data["MODEL_MAG_Z"],
    )
).T

fig = corner.corner(st, labels=["U","I","G","R","Z"])
g = np.where(tab[:, -1] == -1.)[0]
corner.overplot_points(fig, st[g, :], marker='+', color='red')
plt.savefig("corner.png")

#######################################

g = np.where(mem_full_data['ID'] == 1388)
ra_min = mem_full_data['RA'][g].min()
ra_max = mem_full_data['RA'][g].max()
dec_min = mem_full_data['DEC'][g].min()
dec_max = mem_full_data['DEC'][g].max()
delta_ra = ra_max - ra_min
delta_dec = dec_max - dec_min


import urllib.request, json, time

qu = "http://skyserver.sdss.org/dr16/SkyServerWS/ImagingQuery/Rectangular?limit=500000&raMin=%s&raMax=%s&decMin=%s&decMax=%s&imgparams=objid,ra,dec,dered_u,dered_g,dered_r,dered_i,dered_z,err_u,err_g,err_r,err_i,err_z,type&flagsOffList=SATUR_CENTER,BRIGHT&format=json" % (ra_min - 0.01 * delta_ra, ra_max + 0.01 * delta_ra, dec_min - 0.01 * delta_dec, dec_max + 0.01 * delta_dec)
with urllib.request.urlopen(qu) as url:
    data = json.loads(url.read().decode())
ids = []
objs = []
for d in data[0]['Rows']:
    if d['type'] == 3:
        ids.append(d['objid'])
        objs.append(d)
for i, ix in enumerate(mem_full_data['OBJID'][g]):
    if ix not in ids:
        print(mem_full_data['RA'][g][i],mem_full_data['DEC'][g][i])
        print("ouch!")
    else:
        idx = ids.index(ix)
        print(ix)
        for l in ["U","I","G","R","Z"]:
            print(
                l,
                # objs[idx]['dered_' + l.lower()],
                # objs[idx]['err_' + l.lower()],
                # mem_full_data['MODEL_MAG_' + l][g][i],
                # mem_full_data['MODEL_MAGERR_' + l][g][i],
                objs[idx]['dered_' + l.lower()]/mem_full_data['MODEL_MAG_' + l][g][i]-1.,
                objs[idx]['err_' + l.lower()]/mem_full_data['MODEL_MAGERR_' + l][g][i]-1.,
            )


theurl = "http://skyserver.sdss.org/dr8/en/tools/search/x_sql.asp?format=csv&cmd=SELECT%20TOP%20500000%20p.objid,cast(str(p.ra,13,8)%20as%20float)%20as%20ra,p.dec,p.dered_u,p.dered_g,p.dered_r,p.dered_i,p.dered_z,p.err_u,p.err_g,p.err_r,p.err_i,p.err_z,p.type%20FROM%20..PhotoObj%20AS%20p%20JOIN%20dbo.fGetObjFromRect(38.8185369331657,39.0273778257017,-5.52310328628162,-5.29267509982209)%20AS%20b%20ON%20p.objID%20=%20b.objID%20WHERE%20(p.flags%20&%20(dbo.fPhotoFlags('SATUR_CENTER')%20+%20dbo.fPhotoFlags('BRIGHT'))%20=%200)"

http://skyserver.sdss.org/dr8/en/tools/search/x_sql.asp?format=html&cmd=SELECT%20TOP%20500%20p.objid,cast(str(p.ra,13,8)%20as%20float)%20as%20ra,p.dec,p.dered_u,p.dered_g,p.dered_r,p.dered_i,p.dered_z,p.err_u,p.err_g,p.err_r,p.err_i,p.err_z,p.type%20FROM%20..PhotoObj%20AS%20p%20JOIN%20dbo.fGetObjFromRect(38.8185369331657,39.0273778257017,-5.52310328628162,-5.29267509982209)%20AS%20b%20ON%20p.objID%20=%20b.objID%20WHERE%20(p.flags%20AND%20((dbo.fPhotoFlags(%27SATUR_CENTER%27)%20%2b%20dbo.fPhotoFlags(%27BRIGHT%27))%20EQ%200))

http://skyserver.sdss.org/dr8/en/tools/search/x_sql.asp?format=html&cmd=SELECT%20TOP%20500%20p.objid,cast(str(p.ra,13,8)%20as%20float)%20as%20ra,p.dec,p.dered_u,p.dered_g,p.dered_r,p.dered_i,p.dered_z,p.err_u,p.err_g,p.err_r,p.err_i,p.err_z,p.type,p.flags%20FROM%20..PhotoObj%20AS%20p%20JOIN%20dbo.fGetObjFromRect(38.8185369331657,39.0273778257017,-5.52310328628162,-5.29267509982209)%20AS%20b%20ON%20p.objID%20=%20b.objID%20WHERE%20((p.type%20=%203))

http://skyserver.sdss.org/dr8/en/tools/search/x_sql.asp?format=html&cmd=SELECT%20TOP%20500%20p.objid,cast(str(p.ra,13,8)%20as%20float)%20as%20ra,p.dec,p.dered_u,p.dered_g,p.dered_r,p.dered_i,p.dered_z,p.err_u,p.err_g,p.err_r,p.err_i,p.err_z,p.type,p.flags,dbo.fPhotoFlagsN(flags)%20FROM%20..PhotoObj%20AS%20p%20JOIN%20dbo.fGetObjFromRect(38.8185369331657,39.0273778257017,-5.52310328628162,-5.29267509982209)%20AS%20b%20ON%20p.objID%20=%20b.objID



http://skyserver.sdss.org/dr8/en/tools/search/x_sql.asp?format=html&cmd=SELECT%20TOP%20500%20p.objid,cast(str(p.ra,13,8)%20as%20float)%20as%20ra,p.dec,p.dered_u,p.dered_g,p.dered_r,p.dered_i,p.dered_z,p.err_u,p.err_g,p.err_r,p.err_i,p.err_z,p.type,p.flags%20FROM%20..PhotoObj%20AS%20p%20JOIN%20dbo.fGetObjFromRect(38.8185369331657,39.0273778257017,-5.52310328628162,-5.29267509982209)%20AS%20b%20ON%20p.objID%20=%20b.objID%20WHERE (b.flags AND (dbo.fPhotoFlags('SATURATED'))) != 0 WHERE


http://skyserver.sdss.org/dr8/en/tools/search/x_sql.asp?format=html&cmd=SELECT%20TOP%20500%20p.objid,cast(str(p.ra,13,8)%20as%20float)%20as%20ra,p.dec,p.dered_u,p.dered_g,p.dered_r,p.dered_i,p.dered_z,p.err_u,p.err_g,p.err_r,p.err_i,p.err_z,p.type,p.flags,dbo.fPhotoFlagsN(flags)%20FROM%20..PhotoObj%20AS%20p%20JOIN%20dbo.fGetObjFromRect(38.8185369331657,39.0273778257017,-5.52310328628162,-5.29267509982209)%20AS%20b%20ON%20p.objID%20=%20b.objID%20WHERE%20((p.type%20=%203))