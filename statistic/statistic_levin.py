import numpy as np
from metrics.psnr_ssim import calculate_psnr, calculate_ssim


def interp2(X, Y, V, Xq, Yq):
    """
    Linear interpolation equivalent to interp2(V, Xq, Yq) in MATLAB

    @param X: matrix of x coordinates where function V is defined
    @param Y: matrix of y coordinates where function V is defined
    @param V: function defined on square lattice [0, ..., width(V)] X [0, ..., height(V)]
    @param Xq: matrix of x coordinates where interpolation is required
    @param Yq: matrix of y coordinates where interpolation is required
    @return: interpolated values in [Xq, Yq] points
    """
    xq = Xq.copy()
    yq = Yq.copy()
    nrows, ncols = V.shape
    x0 = X[0,0]
    y0 = Y[0,0]

    if nrows < 2 or ncols < 2:
        raise Exception('V shape is too small')
    
    if not xq.shape == yq.shape:
        raise Exception('sizes of Xq indices and Yq indices must match')
    
    # find x values out of range
    xq_bad = ((xq - x0 < 0) | (xq - x0 > ncols - 1))
    if xq_bad.any():
        xq[xq_bad] = 0
    
    # find y values out of range
    yq_bad = ((yq - y0 < 0) | (yq - y0 > nrows - 1))
    if yq_bad.any():
        yq[yq_bad] = 0
    
    # linear indexing. V must be in 'C' order
    ndx = (np.floor(yq) - y0) * ncols + np.floor(xq) - x0
    ndx = ndx.astype('int32')

    # fix parameters on x border
    d = (xq - x0 == ncols - 1)
    xq = xq - np.floor(xq)
    if d.any():
        xq[d] += 1
        ndx[d] -= 1
    
    # fix parameters on y border
    d = (yq - y0 == nrows - 1)
    yq = yq - np.floor(yq)
    if d.any():
        yq[d] += 1
        ndx[d] -= ncols
    
    # interpolate
    one_minus_t = 1 - yq
    V = V.ravel()
    f = (V[ndx] * one_minus_t + V[ndx+ncols] * yq) * (1-xq) + (V[ndx+1] * one_minus_t + V[ndx+ncols+1] * yq) * xq
    f = f.astype(np.uint8)

    # Set out of range positions to extrapval
    extrapval = 0
    if xq_bad.any():
        f[xq_bad] = extrapval
    if yq_bad.any():
        f[yq_bad] = extrapval
    
    return f
    

def comp_upto_shift(img1, img2, maxshift=5):
    """
    Args:
        img1, img2: images to compare. (H, W)
        maxshift: usually maxshift=5 is enough. If you find very low PSNR and SSIM
                  for images with visually good results, maxshift should be set as
                  a larger value.
    Return:
        psnr, ssim.
        img1_shift: img1 at best shift toward img2
    """
    
    shifts = np.arange(-maxshift, maxshift+0.25, 0.25)

    img2 = img2[15:-15, 15:-15]
    img1 = img1[15-maxshift : -15+maxshift, 15-maxshift : -15+maxshift]
    H, W = img2.shape
    gx, gy = np.meshgrid(np.arange(1-maxshift, W+maxshift+1, 1), np.arange(1-maxshift, H+maxshift+1))
    gx0, gy0 = np.meshgrid(np.arange(1, W+1, 1), np.arange(1, H+1, 1))

    # sum of square differences matrix
    ssdem = np.zeros((len(shifts), len(shifts)))
    for i in range(len(shifts)):
        for j in range(len(shifts)):
            gxn = gx0 + shifts[i]
            gyn = gy0 + shifts[j]
            img1_shift = interp2(gx, gy, img1, gxn, gyn)
            ssdem[i, j] = np.sum((img1_shift - img2) ** 2)
    
    ssde = np.min(ssdem)
    i, j = np.where(ssdem == ssde)
    i = i[0]
    j = j[0]

    gxn = gx0 + shifts[i]
    gyn = gy0 + shifts[j]
    img1_shift = interp2(gx, gy, img1, gxn, gyn)
    psnr = calculate_psnr(img1_shift, img2, crop_border=0)
    ssim = calculate_ssim(img1_shift, img2, crop_border=0, ssim3d=False)

    return psnr, ssim, img1_shift
