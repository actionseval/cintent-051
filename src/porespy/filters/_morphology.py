import numpy.typing as npt
from scipy.signal import fftconvolve

from porespy.tools import get_edt, ps_round

__all__ = [
    'erode',
    'dilate',
]


edt = get_edt()


def erode(
    im: npt.NDArray,
    r: int,
    dt: npt.NDArray = None,
    method: str = 'dt',
    smooth: bool = True,
):
    r"""
    Perform erosion with a round structuring element

    Parameters
    ----------
    im : ndarray
        A boolean image with the foreground (to be eroded) indicated by `True`
    r : int
        The radius of the round structuring element to use
    dt : ndarray
        The distance transform of the foreground. If not provided it will be
        computed. This argument is only relevant if `method='dt'`.
    smooth : boolean
        If `True` (default) the single voxel protrusion on the face of the
        structuring element are removed.
    method : str
        Controls which method is used. Options are:

        ========= =============================================================
        method    Description
        ========= =============================================================
        `'dt'`    Uses a distance transform to find all voxels within `r` of
                  the background, then removes them to affect an erosion
        `'conv'`  Uses a FFT based convolution to find all voxels within `r`
                  of the background (voxels with a value smaller than the sum
                  of the structuring element), then removes them to affect an
                  erosion.
        ========= =============================================================

    Returns
    -------
    erosion : ndarray
        An image the same size as `im` with the foreground eroded by the specified
        amount.
    """
    from porespy.tools import settings
    if method == 'dt':
        if dt is None:
            dt = edt(im, parallel=settings.ncores)
        ero = dt >= r if smooth else dt > r
    elif method.startswith('conv'):
        se = ps_round(r=r, ndim=im.ndim, smooth=smooth)
        ero = ~(fftconvolve(~im, se, mode='same') > 0.1)
    return ero


def dilate(
    im: npt.NDArray,
    r: int,
    dt: npt.NDArray = None,
    method: str = 'dt',
    smooth: bool = True,
):
    r"""
    Perform dilation with a round structuring element

    Parameters
    ----------
    im : ndarray
        A boolean image with the foreground (to be dilated) indicated by `True`
    r : int
        The radius of the round structuring element to use
    dt : ndarray
        The distance transform of the foreground. If not provided it will be
        computed. This argument is only relevant if `method='dt'`.
    smooth : boolean
        If `True` (default) the single voxel protrusion on the face of the
        structuring element are removed.
    method : str
        Controls which method is used. Options are:

        ========= =============================================================
        method    Description
        ========= =============================================================
        `'dt'`    Uses a distance transform to find all voxels within `r` of
                  the foreground, then adds them to affect a dilation
        `'conv'`  Using a FFT based convolution to find all voxels within `r`
                  of the foreground (voxels with a value larger than 0), then adds
                  them to affect a dilation.
        ========= =============================================================

    Returns
    -------
    dilation : ndarray
        An image the same size as `im` with the foreground eroded by the specified
        amount.
    """
    from porespy.tools import settings
    im = im == 1
    if method == 'dt':
        if dt is None:
            dt = edt(~im, parallel=settings.ncores)
        dil = dt < r if smooth else dt <= r
        dil += im
    elif method.startswith('conv'):
        se = ps_round(r=r, ndim=im.ndim, smooth=smooth)
        dil = fftconvolve(im, se, mode='same') > 0.1
    return dil


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import porespy as ps

    im = ps.generators.blobs([200, 200], porosity=0.6, seed=5)

    ero1 = erode(im, 5, method='dt').astype(int)
    ero1[~im] = -1

    ero2 = erode(im, 5, method='conv').astype(int)
    ero2[~im] = -1

    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(ero1)
    ax[0][1].imshow(ero2)

    ero1 = erode(im, 10, method='dt').astype(int)
    dil1 = dilate(ero1, 10, method='dt').astype(int)
    dil1[~im] = -1

    dil2 = dilate(ero1, 10, method='conv').astype(int)
    dil2[~im] = -1

    ax[1][0].imshow(dil1)
    ax[1][1].imshow(dil2)
