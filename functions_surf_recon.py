import trimesh as trm
import matplotlib.pyplot as plt
import functions
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree 
import imp
import trimesh as trm
from skimage.measure import marching_cubes

def idl_slp(x, x_mean): # ideal Slope model
'''
    Parameters
    ----------
    x : scalar value of X[i] coordinate.
    x_mean: mean value of X coordinates  
    Returns
    -------
    elevation to corresponding X[i] coordinate and area (below surf., tilted plane, or above surf.)
'''
    mnx = x_mean - .5*x_mean
    mxx = x_mean + .5*x_mean
    if (x<=mxx) & (x>=mnx):
        z0 = np.abs(mnx - x) * 1/(np.tan(np.pi/5)) + 1.5
        area = 1
    elif x < mnx:
        z0 = 1.5
        area = 0
    else: #x >= mxx
        z0 = mxx + 1.5
        area = 2
    return(z0, area)

def der_gaus(x, mu, sigma): # Derivative of Gaussian function
'''
    Parameters
    ----------
    x : array of x axis of derivative of Gaussian function
    mu: mean value
    sigma: STD value 
    Returns
    -------
    2D array of derivative of Gaussian values
'''
    a = 1/(np.sqrt(2*np.pi)*sigma)
    return (-((x-mu)/(np.sqrt(2*np.pi)*sigma**3))*np.exp(-((x-mu)**2)/2*sigma**2))

def normal(x,mu,sigma): # Gaussian function
'''
    Parameters
    ----------
    x : array of x axis of Gaussian function
    mu: mean value
    sigma: STD value 
    Returns
    -------
    2D array of Gaussian values
'''
    return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )


def ideal_vxl_slp(width): # Ideal voxel space for Slope model
'''
    Parameters
    ----------
    width: desired voxel width
    (xs, ys, zs): input coordinates of starting position of voxel space. Have to be defined globally!
    (xe, ye, ze): input coordinates of ending position of voxel space. Have to be defined globally!
    s0: STD value for der_gaus function. Has to be defined globally!
    Returns
    -------
    1) points - 2D array of coordinates of voxels
    2) vxl - signed values of voxels 3D array
    3) ideal - coordinates of ideal surface
    4) (xr.mean(), yr.mean()) - tuple of mean value of xr and yr
'''
    xr = np.arange(xs, xe, width)
    yr = np.arange(ys, ye, width)
    zr = np.arange(zs, ze, width)
    vxl = np.zeros((xr.shape[0], yr.shape[0], zr.shape[0]))
    xx, yy, zz = np.meshgrid(xr, yr, zr, indexing='ij')
    ideal = []
    for i, x in enumerate(xr):
        for j, y in enumerate(yr):
            z0, _ = idl_slp(x, xr.mean(), y, yr.mean())
            ideal.append(np.array([x, y, z0]))
            drg = der_gaus(zr, z0, s0)
            drg[np.argmin(drg):] = drg[np.argmin(drg)]
            drg[:np.argmax(drg)] = drg[np.argmax(drg)]
            vxl[i, j, :] = drg
    points = np.stack((xx, yy, zz))
    ideal = np.stack(ideal)
    return(points, vxl, ideal, (xr.mean(), yr.mean()))


def ideal_vxl_pln(width): # Ideal voxel space for Slope model
'''
    Parameters
    ----------
    width: desired voxel width
    (xs, ys, zs): input coordinates of starting position of voxel space. Have to be defined globally!
    (xe, ye, ze): input coordinates of ending position of voxel space. Have to be defined globally!
    s0: STD value for der_gaus function. Has to be defined globally!
    Returns
    -------
    1) points - 2D array of coordinates of voxels
    2) vxl - signed values of voxels 3D array
    3) ideal - coordinates of ideal surface
'''
    xr = np.arange(xs, xe, width)
    yr = np.arange(ys, ye, width)
    zr = np.arange(zs, ze, width)
    vxl = np.zeros((xr.shape[0], yr.shape[0], zr.shape[0]))
    xx, yy, zz = np.meshgrid(xr, yr, zr, indexing='ij')
    ideal = []
    for i, x in enumerate(xr):
        for j, y in enumerate(yr):
            #zzz = np.zeros(1)
            s0=1
            z0 = 3.3
            ideal.append(np.array([x, y, z0]))
            drg = der_gaus(zr, z0, s0)
            drg[np.argmin(drg):] = drg[np.argmin(drg)]
            drg[:np.argmax(drg)] = drg[np.argmax(drg)]
            vxl[i, j, :] = drg
    points = np.stack((xx, yy, zz))
    ideal = np.stack(ideal)
    #print(xx.shape, yy.shape, zz.shape)
    return(points, vxl, ideal)


def sint_slp(pts, xno, pls_len, yno = False): # Synthetic data generator (Slope model)
'''
    Parameters
    ----------
    pts: coordinates of peaks. 2D array - shape = (n, 3) 
    xno: scalar value for timing error. (.025 for 2.5cm e.g.)
    pls_len: number of samples for each pulse
    yno: amplitude noise. Default - False. given noise shape = (n, pls_len)
    Returns
    -------
    1) pnts - 2D array of coordinates of pulses
    2) area - area of cooresponding pulse (below surf., tilted plane, or above surf.)
'''
    noisx = np.random.randint(0, 100, xx.shape)*(xno*2)*.01 - xno
    #noisy = np.random.random(xx.shape)*(yno*2) - yno
    beam = np.zeros((pls_len, 5))
    beam[:, 2] = (np.arange(0, pls_len)*0.15)-3.3
    beam[:, 2] -= np.median(beam[:pls_len, 2])
    pnts = np.zeros((pts.shape[0]*beam.shape[0], 5))
    area=[]
    for i in range(pts.shape[0]):
        no1 = normal(beam[:, 2]+noisx[i], 0, 0.85)
        # rn1 = np.random.random(pls_len)*no1.max() * noisy # 
        no1/=no1.max()
        g = np.zeros(pls_len)
        for k in range(pls_len):
            g[k] = np.random.choice(yno[:, k], 1)
        z_stp, area_ = idl_slp(pts[i, 0], pts[:, 0].mean())
        no1 = no1+g
        pnts[i*pls_len:(i+1)*pls_len, :] = beam + np.array([pts[i, 0], pts[i, 1], z_stp+noisx[i], 0, 0])
        pnts[i*pls_len:(i+1)*pls_len, 3:] = np.c_[no1, np.gradient(no1)]
        area.append(area_)
    return(pnts, area)


def sint(pts, xno, pls_len, yno = False): # Synthetic data generator (Plane model)
'''
    Parameters
    ----------
    pts: coordinates of peaks. 2D array - shape = (n, 3) 
    xno: scalar value for timing error. (.025 for 2.5cm e.g.)
    pls_len: number of samples for each pulse
    yno: amplitude noise. Default - False. given noise shape = (n, pls_len)
    Returns
    -------
    1) pnts - 2D array of coordinates of pulses
'''
    noisx = np.random.randint(0, 100, xx.shape)*(xno*2)*.01 - xno
    beam = np.zeros((pls_len, 5))
    beam[:, 2] = (np.arange(0, pls_len)*0.15)-3.3
    beam[:, 2] -= np.median(beam[:pls_len, 2])
    pnts = np.zeros((pts.shape[0]*beam.shape[0], 5))
    for i in range(pts.shape[0]):
        no1 = normal(beam[:, 2]+noisx[i], 0, 0.85) 
        no1/=no1.max()
        g = np.zeros(pls_len)
        for k in range(pls_len):
            g[k] = np.random.choice(yno[:, k], 1)
        no1 = no1+g
        pnts[i*pls_len:(i+1)*pls_len, :] = beam + np.array([pts[i, 0], pts[i, 1], pts[i, 2]+noisx[i], 0, 0])
        pnts[i*pls_len:(i+1)*pls_len, 3:] = np.c_[no1, np.gradient(no1)]
    return(pnts)


def angle(ar, angl, pls_len): # angle function
    '''
    

    Parameters
    ----------
    ar : numpy array
        2 dimensional array which represents 3D unstructured data.
    angl: scanning angle. integer or float
    pls_len: number of samples
    Returns
    -------
    ar - pulses with scanning angles

    '''
    y = ar[:, 1]
    bin_w = int((y.max()-y.min())*25)
    bins = pd.interval_range(start=y.min(), end = y.max(), periods=bin_w, closed='both')  
    angle = np.linspace(-5, 5, bin_w)
    for i in range(len(bins)):
        for j in range(int(len(ar)/pls_len)):
            if ar[j*pls_len:(j+1)*pls_len][0, 1] in bins[i]:
                y_ = ar[j*pls_len:(j+1)*pls_len][:, 1]
                z = ar[j*pls_len:(j+1)*pls_len][:, 2]
                b2 = ((z.max()-z.min())-(z.max()-z.min())*math.cos(np.radians(angle[i])))/2
                xx = z*math.sin(np.radians(angle[i]))
                xxx = xx + y_[0]
                zz = np.linspace(b2, (z.max()-z.min())-b2, pls_len)
                ar[j*pls_len:(j+1)*pls_len][:, 2] = zz[::-1]
                ar[j*pls_len:(j+1)*pls_len][:, 1] = xxx
                ar[j*pls_len:(j+1)*pls_len][:, 4] = ar[j*pls_len:(j+1)*pls_len][:, 4][::-1]
    return(ar)


def thr(ar, ln): # thresholding for pulses
'''
    Parameters
    ----------
    ar: coordinates of pulses. 2D array - shape = (n, 3) 
    ln: number of samples for each pulse
    Returns
    -------
    1) pnts - 2D array of coordinates of thresholded pulses (40 percent of maximum value)
'''
    l = int(len(ar)/ln)
    thr_ = []
    for i in range(l):
        amp = ar[i*ln:(i+1)*ln, 3]
        thr_.append(ar[i*ln:(i+1)*ln][amp>amp.max()*0.6])
    print(thr_[0].shape)
    thr_ = np.concatenate(thr_)
    return(thr_)


def voxel(ar, vwidth): # voxelization of the data
    '''
    

    Parameters
    ----------
    ar : numpy array
        2 dimensional array which represents 3D unstructured data.
    vwidth: integer or float
        width of the desired voxel
    Returns
    -------
    voxels - voxelized 3D array.

    '''
    x = ar[:, 0]
    y = ar[:, 1]
    z = ar[:, 2]
    a = ar[:, 4]
    xi = ((x-x.min())/vwidth).astype('int')
    yi = ((y-y.min())/vwidth).astype('int')
    zi = (((z-z.min()))/vwidth).astype('int')
    vcnt = np.zeros((xi.max()+1, yi.max()+1, zi.max()+1))
    vsum= np.zeros((xi.max()+1, yi.max()+1, zi.max()+1))
    for k in range(len(xi)):
        vcnt[xi[k], yi[k], zi[k]] += 1
        vsum[xi[k], yi[k], zi[k]] += a[k]
    vavg = vsum / vcnt
    vavg[vcnt < 1] = np.nan
    vx = np.arange(x.min(), x.max(), vwidth) 
    vy = np.arange(y.min(), y.max(), vwidth) 
    vz = np.arange(z.min(), z.max(), vwidth)
    vxofs,vyofs,vzofs=np.meshgrid(vx, vy, vz, indexing='ij')
    voxels = np.stack((vxofs, vyofs, vzofs, vavg))
    return(voxels)


def nnbh_intrp(ar, vxl): # nearest neighbour interpolation
    '''
    

    Parameters
    ----------
    ar : numpy array
        2 dimensional array which represents 3D unstructured data (5th column is derivative of the pulse).
    vxl: voxel coordinates - 4D data (3, n, m, p)
    Returns
    -------
    pnt_ofs - interpolated voxel values with coordinates

    '''
    x = ar[:, 0]
    y = ar[:, 1]
    z = ar[:, 2]
    a = ar[:, 4]
    tree = cKDTree(np.c_[x, y, z])
    dd, ii = tree.query(np.c_[vxl[0].ravel(), vxl[1].ravel(), vxl[2].ravel()], k=1)
    vann = a[ii].reshape(vxl[0].shape)
    pnt_ofs = np.stack((vxl[0], vxl[1], vxl[2], vann))
    return(pnt_ofs)


def gki(ar, vxl, vwidth): # Gaussian Kerne Interpolation
    '''
    

    Parameters
    ----------
    ar : numpy array
        2 dimensional array which represents 3D unstructured data (5th column is derivative of the pulse).
    vxl: voxel coordinates - 4D data (3, n, m, p)
    vwidth: voxel width
    Returns
    -------
    pp - interpolated voxel values with coordinates

    '''
    x = ar[:, 0]
    y = ar[:, 1]
    z = ar[:, 2]
    a = ar[:, 4]
    sigma = .15
    tree = cKDTree(np.c_[x, y, z])
    dd, ii = tree.query(np.c_[vxl[0].ravel(), vxl[1].ravel(), vxl[2].ravel()], k=10, n_jobs=-1)
    wi = []
    for i in range(len(dd)):
        wi.append(gaussian_weights(dd[i], sigma))
    wi = np.array(wi, dtype = object)
    vi = []
    for i in range(len(ii)):
            vi.append(a[ii[i]])
    vi = np.array(vi, dtype = object)
    vmw = vi*wi
    sm = []
    for i in range(len(vi)):
        sm.append(np.sum(vmw[i])/np.sum(wi[i]))
    var_gki = np.array(sm)
    var_gki.shape = vxl[0].shape
    pp = np.stack((vxl[0], vxl[1], vxl[2], var_gki))    
    return(pp)




def inverse_dist(ar, vxl, vwidth): # Inverse distance interpolation
    '''
    

    Parameters
    ----------
    ar : numpy array
        2 dimensional array which represents 3D unstructured data (5th column is derivative of the pulse).
    vxl: voxel coordinates - 4D data (3, n, m, p)
    vwidth: voxel width
    Returns
    -------
    pp - interpolated voxel values with coordinates

    '''
    x = ar[:, 0]
    y = ar[:, 1]
    z = ar[:, 2]
    a = ar[:, 4]
    tree = cKDTree(np.c_[x, y, z])
    dd, ii = tree.query(np.c_[vxl[0].ravel(), vxl[1].ravel(), vxl[2].ravel()], k=10, n_jobs=-1)
    dd[dd<vwidth/4] = vwidth/4
    wi = 1/dd
    dw = np.sum(wi*a[ii], axis = 1)/np.sum(wi, axis = 1)
    dw.shape = vxl[0].shape
    pp = np.stack((vxl[0], vxl[1], vxl[2], dw))  
    return(pp)


def lin_intrp_vxl(ar, vxl, vwidth): # Linear Interpolation
    '''
    

    Parameters
    ----------
    ar : numpy array
        2 dimensional array which represents 3D unstructured data (5th column is derivative of the pulse).
    vxl: voxel coordinates - 4D data (3, n, m, p)
    vwidth: voxel width
    Returns
    -------
    pp - interpolated voxel values with coordinates

    '''
    x = ar[:, 0]
    y = ar[:, 1]
    z = ar[:, 2]
    a = ar[:, 4]
    a_int = griddata(np.c_[x, y, z], a, np.c_[vxl[0].ravel(), vxl[1].ravel(), vxl[2].ravel()], method = 'linear')
    a_int.shape = vxl[0].shape
    pp = np.stack((vxl[0], vxl[1], vxl[2], a_int))  
    return(pp)


def mcubes(pnt_ofs, level, name):  # Surface Reconstruction with Marching Cubes
    '''
    

    Parameters
    ----------
    pnt_ofs : 4D array (4, n, m, p)
    level: Contour value
    name: string for output file
    Returns
    -------
    pp - interpolated voxel values with coordinates

    '''
    vxofs, vyofs, vzofs, vann = pnt_ofs[0], pnt_ofs[1], pnt_ofs[2], pnt_ofs[3]
    verts, faces, normals, values = marching_cubes(vann, level)
    verts[:, 0] = verts[:, 0]*(vxofs.max() - vxofs.min())/(vann.shape[0]-1) + vxofs.min()
    verts[:, 1] = verts[:, 1]*(vyofs.max() - vyofs.min())/(vann.shape[1]-1) + vyofs.min()
    verts[:, 2] = verts[:, 2]*(vzofs.max() - vzofs.min())/(vann.shape[2]-1) + vzofs.min()       
    mesh = trm.Trimesh(vertices = verts, faces = faces,
                    vertex_normals = normals, vertex_colors = values,
                    process = False)
    return(mesh.export('/home/shaig93/Documents/internship_FWF/images#/exct_'+name+'.ply', file_type = 'ply'))





