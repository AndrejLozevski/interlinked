import sys
import logging
import numpy as np
import scipy as sp
import skimage as ski
import SimpleITK as sitk

import interlinked as lnk

log = logging.getLogger(__name__)


#--| Constants |------------------------------------------------------------------------#

ROW_TILES = 5


#--| Utility |------------------------------------------------------------------------#

# Normalizes an array
def norm(arr, vmin=None, vmax=None, clip=False):
    arr = arr.astype(np.float32)
    if not vmin:
        vmin = arr.min()
    if not vmax:
        vmax = arr.max()
    if vmin == vmax:
        lnk.meta.Error("Cannot normalize array where vmin equals vmax", error=ZeroDivisionError)

    if clip:
        arr = np.clip(arr, vmin, vmax)

    arr -= vmin
    arr /= (vmax - vmin)
    return arr

# Normalizes an array using percentiles
def pnorm(arr, pmin=1, pmax=99, clip=False):
    if not pmin < pmax:
        lnk.meta.Error("Min percentile must be less than max percentile", error=ValueError)

    vmin = np.percentile(arr, pmin)
    vmax = np.percentile(arr, pmax)
    arr = norm(arr, vmin, vmax, clip)
    return arr

# Normalizes two arrays together
def norm_pair(arr1, arr2, clip=False):
    vmin = min(arr1.min(), arr2.min())
    vmax = max(arr1.max(), arr2.max())
    arr1 = norm(arr1, vmin, vmax, clip)
    arr2 = norm(arr2, vmin, vmax, clip)
    return arr1, arr2

# Normalizes two arrays together
def pnorm_pair(arr1, arr2, pmin=1, pmax=99, clip=False):
    if not pmin < pmax:
        lnk.meta.Error("Min percentile must be less than max percentile", error=ValueError)

    flat = np.concatenate(arr1.flatten(), arr2.flatten())
    vmin = np.percentile(flat, pmin)
    vmax = np.percentile(flat, pmax)

    arr1 = norm(arr1, vmin, vmax, clip)
    arr2 = norm(arr2, vmin, vmax, clip)
    return arr1, arr2


#--| Reshaping |------------------------------------------------------------------------#

# Forms a volume from a tiled image
def form_volume(img, shape):
    Lz, Ly, Lx = shape
    img = img.astype(np.float32)
    Oy, Ox = img.shape
    assert (Oy//Ly) * (Ox//Lx) >= Lz

    z = 0
    volume = np.empty(shape, np.float32)
    for iy in range(Oy//Ly):
        for ix in range(Ox//Lx):
            if z >= Lz:
                break
            volume[z,:,:] = img[iy*Ly:(iy+1)*Ly, ix*Lx:(ix+1)*Lx]
            z += 1
    assert volume.shape == (Lz, Ly, Lx)
    return volume

# Forms a tiled image from a volume
def form_tiles(vol, shape):
    Lz, Ly, Lx = shape
    Tx = ROW_TILES
    q, r = divmod(Lz, Tx)
    Ty = q if r == 0 else q + 1

    z = 0
    img = np.ones((Ty*Ly, Tx*Lx), vol.dtype) * vol.min()
    for iy in range(Ty):
        for ix in range(Tx):
            if z >= Lz:
                break
            img[iy*Ly:(iy+1)*Ly, ix*Lx:(ix+1)*Lx] = vol[z,:,:]
            z += 1
    assert img.shape == (Ty*Ly, Tx*Lx)
    return img


#--| ROIs |-----------------------------------------------------------------------------#

# Labels ROIs in a volume from a Suite2p combined stat.npy file
def label_rois(stat, shape):
    Lz, Ly, Lx = shape
    Ln = len(stat)

    count = 0
    rois = np.zeros((Lz,Ly,Lx), np.int32)
    for n in range(Ln):
        ypix = stat[n]["ypix"]    
        xpix = stat[n]["xpix"]    
        zpln = stat[n]["iplane"]    
        assert len(ypix) == len(xpix)

        for py, px in zip(ypix, xpix):
            rois[zpln, py%Ly, px%Lx] = count + 1
        count += 1
    return rois

# Shifts ROIs in a labeled array to fill any missing label IDs
def adjust_rois(arr):
    uniq = np.unique(arr)
    uniq = uniq[uniq != -1]
    if uniq.size == 0:
        lnk.meta.Error("No labels found in array", error=ValueError)
    assert all(uniq >= 0)
    uniq = uniq.astype(np.int64)

    lmin = 0
    lmax = int(uniq.max())

    offset = lmin
    lut = np.arange(lmax - lmin + 1, dtype=arr.dtype)
    lut[:] = -999
    lut[uniq - offset] = np.arange(len(uniq)) + offset
    
    out = arr.copy()
    mask = (arr >= lmin) & (arr <= lmax)
    out_vals = arr[mask] - offset
    out[mask] = lut[out_vals.astype(np.int64)]

    full = np.arange(lmin, lmax+1)
    mask = np.in1d(full, uniq, assume_unique=True)
    missing = full[~mask].tolist()
    return out, uniq, missing

# Removes the specified ROIs from a labeled array
def remove_rois(arr, rois, keep=False):
    if keep:
        keep = np.array(rois)
        mask = np.isin(arr, rois)
        arr[~mask] = arr.min()
    else:
        lookup = np.zeros(arr.max()+1, bool)
        lookup[rois] = True
        arr[lookup[arr]] = arr.min()
    return arr

# Checks if ROIs in a labeled array are valid and without gaps
def validate_rois(rois, Lc=0, throw_err=True):
    if not rois.min() == -1:
        if not throw_err:
            return False
        lnk.meta.Error("Background label should be -1, but found: (Min: %s)", rois.min(), error=ValueError)
    if not len(np.unique(rois)) == rois.max()+2:
        if not throw_err:
            return False
        lnk.meta.Error("Unique label count should be two more than max label, but found: (Unique: %s, Max: %s)", len(np.unique(rois)), rois.max(), error=ValueError)
    if Lc != 0:
        if not rois.max() == Lc-1:
            if not throw_err:
                return False
            lnk.meta.Error("Max label should be one more than roi count, but found: (Max: %s, Lc: %s)", rois.max(), Lc, error=ValueError)
        if not rois.max() - rois.min() == Lc:
            if not throw_err:
                return False
            lnk.meta.Error("Max label minus min label should equal roi count, but found: (Max: %s, Min: %s, Lc: %s)", rois.max(), rois.min(), Lc, error=ValueError)
        if not len(np.unique(rois)) == Lc+1:
            if not throw_err:
                return False
            lnk.meta.Error("Unique label count should one more than roi count, but found: (Max: %s, Min: %s, Lc: %s)", rois.max(), rois.min(), Lc, error=ValueError)
    if not throw_err:
        return True
    return

# Replaces ROIs in a labeled array with their corresponding weights from a list
def weight_rois(rois, weights):
    weighted = np.zeros(rois.shape, np.float32)
    mask = (rois > -1)
    weighted[mask] = weights[rois[mask]]
    return weighted


#--| Alignment |-----------------------------------------------------------------------------#

# Aligns one array to a reference array in real units
def register_array(mov, ref, mov_res, ref_res=None, allow_rotation=False, cval=0, order=3):
    if not (mov.ndim == ref.ndim and len(mov_res) == len(ref_res)):
        lnk.meta.Error("Array and reference array must have same dimensionality", error=ValueError)

    if not ref_res:
        ref_res = mov_res
    if not (mov.ndim == len(mov_res) and ref.ndim == len(ref_res)):
        lnk.meta.Error("Array and its resolution must have same dimensionality", error=ValueError)

    if order not in [0, 1, 2, 3]:
        lnk.meta.Error("Interpolation order must be an integer between 0 and 3", error=ValueError)

    ref = sitk.GetImageFromArray(ref.astype(np.float32))
    mov = sitk.GetImageFromArray(mov.astype(np.float32))
    ref.SetSpacing(ref_res[::-1])
    mov.SetSpacing(mov_res[::-1])

    reg = sitk.ImageRegistrationMethod()
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)

    rot = 1.0 if allow_rotation else 0.0
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetOptimizerWeights([
        rot, rot, rot,
        1.0, 1.0, 1.0
    ])
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=100
    )

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    transform = sitk.CenteredTransformInitializer(
        ref,
        mov,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg.SetInitialTransform(transform, inPlace=False)
    tx = reg.Execute(ref, mov)

    tx_params = transform.GetParameters()
    transform = np.empty((2, 3), np.float32)
    transform[0,:] = tx_params[:3][::-1]
    transform[1,:] = tx_params[-3:][::-1]

    mov = move_array(mov, ref, transform, mov_res, ref_res, cval, order)
    return mov, transform

# Applies a 3D Euler transformation to the given array in real units
def move_array(mov, ref, transform, mov_res, ref_res=None, cval=0, order=3, flip_dims=False):
    if not (mov.ndim == 3 and ref.ndim == 3):
        lnk.meta.Error("Moving array and reference array must have dimensionality of 3", error=ValueError)

    if not (mov.ndim == ref.ndim and len(mov_res) == len(ref_res)):
        lnk.meta.Error("Moving array and reference array must have same dimensionality", error=ValueError)

    if not ref_res:
        ref_res = mov_res
    if not (mov.ndim == len(mov_res) and ref.ndim == len(ref_res)):
        lnk.meta.Error("Array and its resolution must have same dimensionality", error=ValueError)

    if order not in [0, 1, 2, 3]:
        lnk.meta.Error("Interpolation order must be an integer between 0 and 3", error=ValueError)

    tx = sitk.Euler3DTransform()
    if transform.shape == (2, 3):
        rotZ, rotY, rotX = transform[0,:][::-1] if flip_dims else transform[0,:]
        offZ, offY, offX = transform[1,:][::-1] if flip_dims else transform[1,:]
    elif len(transform) == 3:
        rotZ, rotY, rotX = 0.0, 0.0, 0.0
        offZ, offY, offX = transform[::-1] if flip_dims else transform
    else:
        lnk.meta.Error("Transform must be a (2, 3) or (1, 3) array)", error=ValueError)
    tx.SetRotation(rotX, rotY, rotZ)
    tx.setTranslation((offX, offY, offZ))

    mov = sitk.GetImageFromArray(mov.astype(np.float32))
    ref = sitk.GetImageFromArray(ref.astype(np.float32))
    mov.SetSpacing(mov_res[::-1])
    ref.SetSpacing(ref_res[::-1])

    methods = [sitk.sitkNearestNeighbor, sitk.sitkLinear, sitk.sitkBSpline2, sitk.sitkBSpline]
    order   = methods[order]

    adj = sitk.ResampleImageFilter()
    adj.SetReferenceImage(ref)
    adj.SetInterpolator(order)
    adj.SetDefaultPixelValue(cval)
    adj.SetOutputPixelType(mov.GetPixelID())
    adj.SetTransform(tx)

    mov = adj.Execute(mov)
    mov = sitk.GetArrayFromImage(mov)
    return mov


