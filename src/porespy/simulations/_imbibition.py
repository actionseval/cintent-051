import inspect

import numpy as np
from edt import edt
from numba import njit, prange

from porespy.filters import (
    erode,
    fftmorphology,
    find_small_clusters,
    find_trapped_clusters,
    seq_to_satn,
    trim_disconnected_voxels,
)
from porespy.metrics import pc_map_to_pc_curve
from porespy.tools import (
    Results,
    _insert_disk_at_points,
    _insert_disk_at_points_parallel,
    _insert_disks_at_points_parallel,
    get_tqdm,
    make_contiguous,
    parse_steps,
    ps_round,
    settings,
)

tqdm = get_tqdm()


__all__ = [
    'imbibition',
    'imbibition_dt',
    'imbibition_dt_fft',
    'imbibition_fft',
    'imbibition_dsi',
]


def imbibition_dsi(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=True,
):
    r"""
    Performs a distance transform based imbibition simulation using direct sphere
    insertion to accomplish dilation and distance transform thresholding for erosion

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (non-wetting) phase would exit the domain. If not provided
        then trapping of the non-wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time. Also, it can be converted to
        integer type or round to fewer decimal places to reduce the number of unique
        sphere sizes to insert if `steps=None`.
    steps : scalar or array_like
        Controls which sphere sizes to invade. If an `int` then this many steps
        between 1 and the maximum size are used. A `tuple` is treated as the start
        and stop of the integer values. A `list` or `ndarray` is used directly. If
        `None` (default) then each unique value in the distance transform is used.
    smooth : boolean
        If `True` (default) then the spheres are drawn without any single voxel
        protrusions on the faces.

    Returns
    -------
    results : Dataclass-like object
        An object with the following attributes:

        =========== ===========================================================
        Attribute   Description
        =========== ===========================================================
        `im_seq`    The sequence map indicating the sequence or step number at
                    which each voxels was first invaded.
        `im_size`   The size map indicating the size of the sphere being drawn
                    when each voxel was first invaded.
        =========== ===========================================================

    Notes
    -----
    The sphere insertion steps will be executed in parallel if
    ``porespy.settings.ncores > 1``
    """
    if settings.ncores > 1:
        func = _insert_disk_at_points_parallel
    else:
        func = _insert_disk_at_points
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    dt_int = dt.astype(int)
    bins = parse_steps(steps=steps, vals=dt[im], descending=False)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    nwp = np.zeros_like(im, dtype=bool)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        if smooth:
            seeds = dt >= r
            edges = dt_int == r
        else:
            seeds = dt > r
            edges = (dt > r)*(dt_int <= (r + 1))
        coords = np.vstack(np.where(edges))
        nwp.fill(False)
        if coords.size > 0:
            nwp = func(
                im=nwp,
                coords=coords,
                r=int(r),
                v=True,
                smooth=smooth,
            )
        nwp[seeds] = True
        wp = (~nwp)*im
        if inlets is not None:
            wp = trim_disconnected_voxels(wp, inlets=inlets)
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i + 1
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn='min',
            method='labels',
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode='symmetric')
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results


def imbibition_dt_fft(
    im,
    inlets=None,
    outlets=None,
    residual=None,
    dt=None,
    steps=None,
    smooth=True,
):
    r"""
    Performs a distance transform based imbibition simulation using distance
    transform thresholding for the erosion step and fft-based convolution for
    the dilation step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with ``True`` values indicating the inlet locations for the
        invading (wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with ``True`` values indicating the outlet locations through
        which defending (non-wetting) phase would exit the domain. If not provided
        then trapping of the non-wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time. Also, it can be converted to
        integer type or round to fewer decimal places to reduce the number of unique
        sphere sizes to insert if `steps=None`.
    steps : scalar or array_like
        Controls which sphere sizes to invade. If an `int` then this many steps
        between 1 and the maximum size are used. A `tuple` is treated as the start
        and stop of the integer values. A `list` or `ndarray` is used directly. If
        `None` (default) then each unique value in the distance transform is used.
    smooth : boolean
        If ``True`` (default) then the spheres are drawn without any single voxel
        protrusions on the faces.

    Returns
    -------
    results : Dataclass-like object
        An object with the following attributes:

        =========== ===========================================================
        Attribute   Description
        =========== ===========================================================
        ``im_seq``  The sequence map indicating the sequence or step number at
                    which each voxel was first invaded.
        ``im_size`` The size map indicating the size of the sphere being drawn
                    when each voxel was first invaded.
        =========== ===========================================================

    Notes
    -----
    The distance transform will be executed in parallel if
    ``porespy.settings.ncores > 1``
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    bins = parse_steps(steps=steps, vals=dt[im], descending=False)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        # Perform erosion using dt
        seeds = dt >= r if smooth else dt > r
        # Perform dilation using convolution
        se = ps_round(r, ndim=im.ndim, smooth=smooth)
        wp = im*~fftmorphology(seeds, se, mode='dilation')
        # Trimming disconnected wetting phase
        if inlets is not None:
            wp = trim_disconnected_voxels(wp, inlets=inlets)
        # TODO: Not sure this residual code works
        # if residual is not None:
        #     blobs = trim_disconnected_voxels(residual, inlets=wp)
        #     seeds2 = trim_disconnected_voxels(seeds, inlets=blobs + inlets)
        #     wp = im*~fftmorphology(seeds2, se, mode='dilation')
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i+1
    # if residual is not None:
    #     im_seq[im_seq > 0] += 1
    #     im_seq[residual] = 1
    #     im_size[residual] = np.inf
    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn='min',
            method='labels',
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode='symmetric')
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results


def imbibition_dt(
    im,
    inlets=None,
    outlets=None,
    residual=None,
    dt=None,
    steps=None,
    smooth=True,
):
    r"""
    Performs a distance transform based imbibition simulation using distance
    transform thresholding for the erosion step and a second distance transform
    for the dilation step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (non-wetting) phase would exit the domain. If not provided
        then trapping of the non-wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time. Also, it can be converted to
        integer type or round to fewer decimal places to reduce the number of unique
        sphere sizes to insert if `steps=None`.
    steps : scalar or array_like
        Controls which sphere sizes to invade. If an `int` then this many steps
        between 1 and the maximum size are used. A `tuple` is treated as the start
        and stop of the integer values. A `list` or `ndarray` is used directly. If
        `None` (default) then each unique value in the distance transform is used.
    smooth : boolean
        If `True` (default) then the spheres are drawn without any single voxel
        protrusions on the faces.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== ============================================================
        Attribute  Description
        ========== ============================================================
        im_seq     A numpy array with each voxel value indicating the sequence
                   at which it was invaded.  Values of -1 indicate that it was
                   not invaded.
        im_size    A numpy array with each voxel value indicating the radius of
                   spheres being inserted when it was invaded.
        ========== ============================================================

    Notes
    -----
    The distance transforms will be executed in parallel if
    ``porespy.settings.ncores > 1``
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    bins = parse_steps(steps=steps, vals=dt[im], descending=False)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        # Perform erosion using dt
        seeds = dt >= r if smooth else dt > r
        # Perform dilation using dt
        tmp = edt(~seeds, parallel=settings.ncores)
        wp = ~(tmp < r) if smooth else ~(tmp <= r)
        wp[~im] = 0
        # Trimming disconnected wetting phase
        if inlets is not None:
            wp = trim_disconnected_voxels(wp, inlets=inlets)
        # TODO: Not sure this residual code works
        # if residual is not None:
        #     blobs = trim_disconnected_voxels(residual, inlets=wp)
        #     seeds2 = trim_disconnected_voxels(seeds, inlets=blobs + inlets)
        #     wp = im*~fftmorphology(seeds2, se, mode='dilation')
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i+1
    # if residual is not None:
    #     im_seq[im_seq > 0] += 1
    #     im_seq[residual] = 1
    #     im_size[residual] = np.inf
    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn='min',
            method='labels',
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode='symmetric')
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results


def imbibition_fft(
    im,
    inlets=None,
    outlets=None,
    residual=None,
    dt=None,
    steps=None,
    smooth=True,
):
    r"""
    Performs a distance transform based imbibition simulation using fft-based
    convolution for both the erosion and dilation steps

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (non-wetting) phase would exit the domain. If not provided
        then trapping of the non-wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time. Also, it can be converted to
        integer type or round to fewer decimal places to reduce the number of unique
        sphere sizes to insert if `steps=None`.
    steps : scalar or array_like
        Controls which sphere sizes to invade. If an `int` then this many steps
        between 1 and the maximum size are used. A `tuple` is treated as the start
        and stop of the integer values. A `list` or `ndarray` is used directly. If
        `None` (default) then each unique value in the distance transform is used.
    smooth : boolean
        If `True` (default) then the spheres are drawn without any single voxel
        protrusions on the faces.

    Returns
    -------
    results : Dataclass-like object
        An object with the following attributes:

        =========== ===========================================================
        Attribute   Description
        =========== ===========================================================
        ``im_seq``  The sequence map indicating the sequence or step number at
                    which each voxels was first invaded.
        ``im_size`` The size map indicating the size of the sphere being drawn
                    when each voxel was first invaded.
        =========== ===========================================================
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    bins = parse_steps(steps=steps, vals=dt[im], descending=False)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        # Perform erosion using convolution
        se = ps_round(r, ndim=im.ndim, smooth=smooth)
        seeds = ~fftmorphology(~im, se, mode='dilation')
        # Perform dilation using convolution
        wp = im*~fftmorphology(seeds, se, mode='dilation')
        # Trimming disconnected wetting phase
        if inlets is not None:
            wp = trim_disconnected_voxels(wp, inlets=inlets)
        # TODO: Not sure this residual code works
        # if residual is not None:
        #     blobs = trim_disconnected_voxels(residual, inlets=wp)
        #     seeds2 = trim_disconnected_voxels(seeds, inlets=blobs + inlets)
        #     wp = im*~fftmorphology(seeds2, se, mode='dilation')
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i+1
    # if residual is not None:
    #     im_seq[im_seq > 0] += 1
    #     im_seq[residual] = 1
    #     im_size[residual] = np.inf
    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn='min',
            method='labels',
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode='symmetric')
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results


def imbibition(
    im,
    pc=None,
    dt=None,
    inlets=None,
    outlets=None,
    residual=None,
    steps=25,
    min_size=0,
    conn='min',
):
    r"""
    Performs an imbibition simulation using image-based sphere insertion

    Parameters
    ----------
    im : ndarray
        The image of the porous materials with void indicated by ``True``
    pc : ndarray
        An array containing precomputed capillary pressure values in each
        voxel. This can include gravity effects or not. This can be generated
        by ``capillary_transform``. If not provided then `2/dt` is used.
    dt : ndarray (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time.
    inlets : ndarray
        An image the same shape as ``im`` with ``True`` values indicating the
        wetting fluid inlet(s).  If ``None`` then the wetting film is able to
        appear anywhere within the domain.
    residual : ndarray, optional
        A boolean mask the same shape as ``im`` with ``True`` values
        indicating to locations of residual wetting phase.
    steps : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given
        then steps will be created between the lowest and highest pressures
        in ``pc``. If a list is given, each value in the list is used
        directly in order.
    min_size : int
        Any clusters of trapped voxels smaller than this size will be set to not
        trapped. This argument is only used if `outlets` is given. This is useful
        to prevent small voxels along edges of the void space from being set to
        trapped. These can appear to be trapped due to the jagged nature of the
        digital image. The default is 0, meaning this adjustment is not applied,
        but a value of 3 or 4 is recommended to activate this adjustment.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default is `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    results : Result Object
        A dataclass-like object with the following attributes:

        =========== ===========================================================
        Attribute   Description
        =========== ===========================================================
        im_pc       An ndarray with each voxel indicating the step number at
                    which it was first invaded by wetting phase.
        im_seq      A numpy array with each voxel value indicating the sequence
                    at which it was invaded by the wetting phase.  Values of -1
                    indicate that it was not invaded, either because it was
                    trapped, inaccessbile, or sufficient pressure was not
                    reached.
        im_snwp     A numpy array with each voxel value indicating the global
                    non-wetting phase saturation at the point it was invaded.
        im_trapped  A numpy array with ``True`` values indicating trapped
                    voxels if `outlets` was provided, otherwise will be `None`.
        pc          1D array of capillary pressure values that were applied
        snwp        1D array of non-wetting phase saturations for each applied
                    value of capillary pressure (``pc``).
        =========== ===========================================================

    Notes
    -----
    The simulation proceeds as though the non-wetting phase pressure is very
    high and is slowly lowered. Then imbibition occurs into the smallest
    accessible regions at each step. Closed or inaccessible pores are
    assumed to be filled with wetting phase.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/simulations/reference/imbibition.html>`_
    to view online example.

    """
    im = np.array(im, dtype=bool)

    if dt is None:
        dt = edt(im)

    if pc is None:
        pc = 2/dt

    pc = np.copy(pc)
    pc[~im] = 0  # Remove any infs or nans from pc computation

    if isinstance(steps, int):
        mask = np.isfinite(pc)*im
        Ps = np.logspace(
            np.log10(pc[mask].max()),
            np.log10(pc[mask].min()*0.99),
            steps,
        )
    elif steps is None:
        Ps = np.unique(pc[im])[::-1]
    else:
        Ps = np.unique(steps)[::-1]  # To ensure they are in descending order

    # Initialize empty arrays to accumulate results of each loop
    im_pc = np.zeros_like(im, dtype=float)
    im_seq = np.zeros_like(im, dtype=int)

    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for step, P in enumerate(tqdm(Ps, desc=desc, **settings.tqdm)):
        # This can be made faster if I find a way to get only seeds on edge, so
        # less spheres need to be drawn
        invadable = (pc <= P)*im
        # Using FFT-based erosion to find edges.  When struct is small, this is
        # quite fast so it saves time overall by reducing the number of spheres
        # that need to be inserted.
        edges = (~erode(invadable, r=1, smooth=False, method='conv'))*invadable
        nwp_mask = np.zeros_like(im, dtype=bool)
        if np.any(edges):
            coords = np.where(edges)
            radii = dt[coords].astype(int) + 1
            nwp_mask = _insert_disks_at_points_parallel(
                im=nwp_mask,
                coords=np.vstack(coords),
                radii=radii,
                v=True,
                smooth=True,
                overwrite=True,
            )
            nwp_mask += invadable
        if inlets is not None:
            nwp_mask = ~trim_disconnected_voxels(
                im=(~nwp_mask)*im,
                inlets=inlets,
                conn=conn,
            )*im
        if residual is not None:
            nwp_mask = nwp_mask * ~residual

        mask = (nwp_mask == 0) * (im_seq == 0) * im
        if np.any(mask):
            im_seq[mask] = step
            im_pc[mask] = P
    im_seq = make_contiguous(im_seq)

    trapped = None  # Initialize trapped to None in case outlets not given
    if outlets is not None:
        if inlets is not None:
            outlets[inlets] = False  # Ensure outlets do not overlap inlets
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            method='labels' if len(Ps) < 100 else 'queue',
            conn=conn,
        )
        if min_size > 0:
            temp = find_small_clusters(
                im=im,
                trapped=trapped,
                min_size=min_size,
                conn=conn,
            )
            trapped = temp.im_trapped
        im_pc[trapped] = -np.inf
        im_seq[trapped] = -1

    if residual is not None:
        im_pc[residual] = np.inf
        im_seq[residual] = 0

    satn = seq_to_satn(im=im, seq=im_seq, mode='imbibition')
    # Collect data in a Results object
    results = Results()
    results.im_snwp = satn
    results.im_seq = im_seq
    results.im_pc = im_pc
    results.im_trapped = trapped

    pc_curve = pc_map_to_pc_curve(pc=im_pc, im=im, seq=im_seq, mode='imbibition')
    results.pc = pc_curve.pc
    results.snwp = pc_curve.snwp
    return results


@njit(parallel=True)
def _insert_disks_npoints_nradii_1value_parallel(
    im,
    coords,
    radii,
    v,
    overwrite=False,
    smooth=False,
):  # pragma: no cover
    if im.ndim == 2:
        xlim, ylim = im.shape
        for row in prange(len(coords[0])):
            i, j = coords[0][row], coords[1][row]
            r = radii[row]
            for a, x in enumerate(range(i-r, i+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(j-r, j+r+1)):
                        if (y >= 0) and (y < ylim):
                            R = ((a - r)**2 + (b - r)**2)**0.5
                            if (R <= r)*(~smooth) or (R < r)*(smooth):
                                if overwrite or (im[x, y] == 0):
                                    im[x, y] = v
    else:
        xlim, ylim, zlim = im.shape
        for row in prange(len(coords[0])):
            i, j, k = coords[0][row], coords[1][row], coords[2][row]
            r = radii[row]
            for a, x in enumerate(range(i-r, i+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(j-r, j+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(k-r, k+r+1)):
                                if (z >= 0) and (z < zlim):
                                    R = ((a - r)**2 + (b - r)**2 + (c - r)**2)**0.5
                                    if (R <= r)*(~smooth) or (R < r)*(smooth):
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
    return im


# %%

if __name__ == '__main__':
    from copy import copy

    import matplotlib.pyplot as plt
    import numpy as np
    from edt import edt

    import porespy as ps
    ps.visualization.set_mpl_style()

    cm = copy(plt.cm.turbo)
    cm.set_under('k')
    cm.set_over('grey')

    i = np.random.randint(1, 100000)  # bad: 38364, good: 65270, 71698
    i = 50591
    # i = 59477  # Bug in pc curve if lowest point is not 0.99 x min(pc)
    # i = 38364
    print(i)
    im = ps.generators.blobs([500, 500], porosity=0.65, blobiness=2, seed=i)
    im = ps.filters.fill_invalid_pores(im)

    inlets = ps.generators.faces(im.shape, inlet=0)
    outlets = ps.generators.faces(im.shape, outlet=0)
    lt = ps.filters.local_thickness_dt(im)
    residual = (lt < 8)*im
    pc = ps.filters.capillary_transform(im=im, voxel_size=1e-4)

    imb1 = imbibition(im=im, pc=pc, inlets=inlets, min_size=1)
    imb2 = imbibition(im=im, pc=pc, inlets=inlets, outlets=outlets, min_size=1)
    imb3 = imbibition(im=im, pc=pc, inlets=inlets, residual=residual, min_size=1)
    imb4 = imbibition(im=im, pc=pc, inlets=inlets, outlets=outlets, residual=residual, min_size=1)

    # %%

    fig, ax = plt.subplot_mosaic(
        [['(a)', '(b)', '(e)', '(e)'],
         ['(c)', '(d)', '(e)', '(e)']],
        figsize=[12, 8],
     )
    tmp = np.copy(imb1.im_seq).astype(float)
    vmax = tmp.max()
    tmp[tmp < 0] = vmax + 1
    tmp[tmp == 0] = np.nan
    tmp[~im] = -1
    ax['(a)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

    tmp = np.copy(imb2.im_seq).astype(float)
    vmax = tmp.max()
    tmp[tmp < 0] = vmax + 1
    tmp[tmp == 0] = np.nan
    tmp[~im] = -1
    ax['(b)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

    tmp = np.copy(imb3.im_seq).astype(float)
    vmax = tmp.max()
    tmp[tmp < 0] = vmax + 1
    tmp[tmp == 0] = np.nan
    tmp[~im] = -1
    ax['(c)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

    tmp = np.copy(imb4.im_seq).astype(float)
    vmax = tmp.max()
    tmp[tmp < 0] = vmax + 1
    tmp[tmp == 0] = np.nan
    tmp[~im] = -1
    ax['(d)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        pc=imb1.im_pc, seq=imb1.im_seq, im=im, mode='imbibition')
    ax['(e)'].semilogx(Pc, Snwp, 'b->', label='imbibition')

    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        pc=imb2.im_pc, seq=imb2.im_seq, im=im, mode='imbibition')
    ax['(e)'].semilogx(Pc, Snwp, 'r-<', label='imbibition w trapping')

    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        pc=imb3.im_pc, seq=imb3.im_seq, im=im, mode='imbibition')
    ax['(e)'].semilogx(Pc, Snwp, 'g-^', label='imbibition w residual')

    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        pc=imb4.im_pc, seq=imb4.im_seq, im=im, mode='imbibition')
    ax['(e)'].semilogx(Pc, Snwp, 'm-*', label='imbibition w residual & trapping')

    ax['(e)'].legend()
