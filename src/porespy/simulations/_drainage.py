import inspect
from typing import Literal

import numpy as np
import numpy.typing as npt

from porespy.filters import (
    fftmorphology,
    find_small_clusters,
    find_trapped_clusters,
    pc_to_satn,
    trim_disconnected_voxels,
)
from porespy.metrics import pc_map_to_pc_curve
from porespy.tools import (
    Results,
    _insert_disk_at_points,
    _insert_disk_at_points_parallel,
    _insert_disks_at_points_parallel,
    get_edt,
    get_strel,
    get_tqdm,
    make_contiguous,
    parse_steps,
    ps_round,
    settings,
)

__all__ = [
    "drainage",
    # The following are reference implementations using different techniques
    "drainage_dt",
    "drainage_fft",
    "drainage_dt_fft",
    "drainage_dsi",
]


edt = get_edt()
tqdm = get_tqdm()
strel = get_strel()


def drainage_dsi(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=True,
):
    r"""
    Performs a distance transform based drainage simulation using direct sphere
    insertion to accomplish dilation and distance transform thresholding for erosion

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (non-wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid cand appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (wetting) phase would exit the domain. If not provided then
        trapping of the wetting phase is ignored.
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

        =========== ================================================================
        Attribute   Description
        =========== ================================================================
        ``im_seq``  The sequence map indicating the sequence or step number at which
                    each voxels was first invaded.
        ``im_size`` The size map indicating the size of the sphere being drawn
                    when each voxel was first invaded.
        =========== ================================================================

    Notes
    -----
    The sphere insert steps will be executed in parallel if
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
    bins = parse_steps(steps=steps, vals=dt[im], descending=True)
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
            edges = (dt > r) * (dt_int <= (r + 1))
        if inlets is not None:
            seeds = trim_disconnected_voxels(seeds, inlets=inlets)
            edges *= seeds
        coords = np.vstack(np.where(edges))
        if coords.size > 0:
            nwp = func(
                im=nwp,
                coords=coords,
                r=int(r),
                v=True,
                smooth=smooth,
            )
        nwp[seeds] = True
        mask = nwp * (im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i + 1
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn="min",
            method="cluster",
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode="symmetric")
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq * im
    results.im_size = im_size * im
    return results


def drainage_dt_fft(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=True
):
    r"""
    Performs a distance transform based drainage simulation using distance transform
    thresholding for the erosion step and fft-based convolution for the dilation
    step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (non-wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid cand appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (wetting) phase would exit the domain. If not provided then
        trapping of the wetting phase is ignored.
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

        =========== ================================================================
        Attribute   Description
        =========== ================================================================
        `im_seq`    The sequence map indicating the sequence or step number at which
                    each voxels was first invaded.
        `im_size`   The size map indicating the size of the sphere being drawn
                    when each voxel was first invaded.
        =========== ================================================================

    Notes
    -----
    The distance transform will be executed in parallel if
    `porespy.settings.ncores > 1`
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    bins = parse_steps(steps=steps, vals=dt[im], descending=True)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        seeds = dt >= r if smooth else dt > r
        if inlets is not None:
            seeds = trim_disconnected_voxels(seeds, inlets=inlets)
        if not np.any(seeds):
            continue
        se = ps_round(int(r), ndim=im.ndim, smooth=smooth)
        nwp = fftmorphology(seeds, se, "dilation")
        mask = nwp * (im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i + 1
    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn="min",
            method="cluster",
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode="symmetric")
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq * im
    results.im_size = im_size * im
    return results


def drainage_fft(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=True,
):
    r"""
    Performs a distance transform based drainage simulation using fft-based
    convolution for both the erosion and dilation steps

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (non-wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid cand appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (wetting) phase would exit the domain. If not provided then
        trapping of the wetting phase is ignored.
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

        =========== ================================================================
        Attribute   Description
        =========== ================================================================
        `im_seq`    The sequence map indicating the sequence or step number at which
                    each voxels was first invaded.
        `im_size`   The size map indicating the size of the sphere being drawn
                    when each voxel was first invaded.
        =========== ================================================================
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    bins = parse_steps(steps=steps, vals=dt[im], descending=True)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        se = ps_round(int(r), ndim=im.ndim, smooth=smooth)
        seeds = ~fftmorphology(~im, se, "dilation")
        if inlets is not None:
            seeds = trim_disconnected_voxels(seeds, inlets=inlets)
        if not np.any(seeds):
            continue
        se = ps_round(int(r), ndim=im.ndim, smooth=smooth)
        nwp = fftmorphology(seeds, se, "dilation")
        mask = nwp * (im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i + 1
    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn="min",
            method="cluster",
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode="symmetric")
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq * im
    results.im_size = im_size * im
    return results


def drainage_dt(
    im,
    inlets,
    outlets=None,
    # residual=None,
    dt=None,
    steps=None,
    smooth=True,
):
    r"""
    Performs a distance transform based drainage simulation using distance transform
    thresholding for the erosion step and a second distance transform for the
    dilation step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (non-wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid cand appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (wetting) phase would exit the domain. If not provided then
        trapping of the wetting phase is ignored.
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

        ========== =================================================================
        Attribute  Description
        ========== =================================================================
        im_seq     A numpy array with each voxel value indicating the sequence
                   at which it was invaded.  Values of -1 indicate that it was
                   not invaded.
        im_size    A numpy array with each voxel value indicating the radius of
                   spheres being inserted when it was invaded.
        ========== =================================================================

    Notes
    -----
    The distance transforms will be executed in parallel if
    `porespy.settings.ncores > 1`
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    bins = parse_steps(steps=steps, vals=dt[im], descending=True)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        seeds = dt >= r if smooth else dt > r
        if inlets is not None:
            seeds = trim_disconnected_voxels(seeds, inlets=inlets)
        if not np.any(seeds):
            continue
        tmp = edt(~seeds, parallel=settings.ncores)
        nwp = tmp < r if smooth else tmp <= r
        # if residual is not None:
        #     blobs = trim_disconnected_voxels(residual, inlets=nwp)
        #     seeds = dt >= r
        #     seeds = trim_disconnected_voxels(seeds, inlets=blobs + inlets)
        #     nwp = edt(~seeds, parallel=settings.ncores) < r
        mask = nwp * (im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i + 1
    # if residual is not None:
    #     im_seq[im_seq > 0] += 1
    #     im_seq[residual] = 1
    #     im_size[residual] = -np.inf
    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn="min",
            method="cluster",
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode="symmetric")
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq * im
    results.im_size = im_size * im
    return results


def drainage(
    im: npt.NDArray,
    pc: npt.NDArray = None,
    dt: npt.NDArray = None,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    residual: npt.NDArray = None,
    steps: int = None,
    conn: Literal["min", "max"] = "min",
    min_size: int = 0,
):
    r"""
    Simulate drainage using image-based sphere insertion, optionally including
    gravity [4]_.

    Parameters
    ----------
    im : ndarray
        The image of the porous media with ``True`` values indicating the
        void space.
    pc : ndarray, optional
        Precomputed capillary pressure transform which is used to determine
        the invadability of each voxel. If not provided then it is calculated
        as `2/dt`.
    dt : ndarray (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time.
    inlets : ndarray, optional
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. If not specified then access limitations
        are not applied so the result is essentially a local thickness filter.
    outlets : ndarray, optional
        A boolean image with ``True`` values indicating the outlet locations.
        If this is provided then trapped voxels of wetting phase are found and
        all the output images are adjusted accordingly. Note that trapping can
        be assessed as a postprocessing step as well, so if this is not provided
        trapping can still be considered.
    residual : ndarray, optional
        A boolean array indicating the locations of any residual invading
        phase. This is added to the intermediate image prior to trimming
        disconnected clusters, so will create connections to some clusters
        that would otherwise be removed. The residual phase is indicated
        in the capillary pressure map by ``-np.inf`` values, since these voxels
        are invaded at all applied capillary pressures. Note that the presence of
        residual non-wetting phase makes it impossible to correctly deal with
        trapping, so if both `residual` and `outlets` are given then an error is
        raised.
    steps : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given then the given
        number of steps will be created between the lowest and highest values in
        ``pc``. If a list is given, each value in the list is used in ascending
        order. If `None` is given then all the possible values in `pc`
        are used.
    conn : str
        Controls the shape of the structuring element used to find neighboring
        voxels when looking at connectivity of invading blobs.  Options are:

        ========= =============================================================
        Option    Description
        ========= =============================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6
                  neighbors in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D
                  and 26 neighbors in 3D.
        ========= =============================================================

    min_size : int
        Any clusters of trapped voxels smaller than this size will be set to not
        trapped. This argument is only used if `outlets` is given. This is useful
        to prevent small voxels along edges of the void space from being set to
        trapped. These can appear to be trapped due to the jagged nature of the
        digital image. The default is 0, meaning this adjustment is not applied,
        but a value of 3 or 4 is recommended to activate this adjustment.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== ============================================================
        Attribute  Description
        ========== ============================================================
        im_seq     An ndarray with each voxel indicating the step number at
                   which it was first invaded by non-wetting phase
        im_snwp    A numpy array with each voxel value indicating the global
                   value of the non-wetting phase saturation at the point it
                   was invaded
        im_size    A numpy array with each voxel containing the radius of the
                   sphere, in voxels, that first overlapped it.
        im_pc      A numpy array with each voxel value indicating the
                   capillary pressure at which it was invaded.
        im_trapped A numpy array with ``True`` values indicating trapped voxels
                   if `outlets` was provided, otherwise will be `None`.
        pc         1D array of capillary pressure values that were applied
        swnp       1D array of non-wetting phase saturations for each applied
                   value of capillary pressure (``pc``).
        ========== ============================================================

    Notes
    -----
    This algorithm only provides sensible results for gravity stabilized
    configurations, meaning the more dense fluid is on the bottom. Be sure that
    ``inlets`` are specified accordingly.

    References
    ----------
    .. [4] Chadwick EA, Hammen LH, Schulz VP, Bazylak A, Ioannidis MA, Gostick JT.
       Incorporating the effect of gravity into image-based drainage simulations on
       volumetric images of porous media.
       `Water Resources Research. <https://doi.org/10.1029/2021WR031509>`__.
       58(3), e2021WR031509 (2022)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/simulations/reference/drainage.html>`__
    to view online example.

    """
    if (residual is not None) and (outlets is not None):
        raise Exception("Trapping cannot be properly assessed if residual present")

    im = np.array(im, dtype=bool)

    if dt is None:
        dt = edt(im)

    if outlets is not None:
        outlets = outlets * im
        if np.sum(inlets * outlets):
            raise Exception("Specified inlets and outlets overlap")

    if pc is None:
        pc = 2.0 / dt
    pc[~im] = 0  # Remove any infs or nans from pc computation

    if isinstance(steps, int):  # Use values in pc for invasion steps
        mask = np.isfinite(pc) * im
        Ps = np.logspace(
            np.log10(pc[mask].min()),
            np.log10(pc[mask].max()),
            steps,
        )
    elif steps is None:
        Ps = np.unique(pc[im])
    else:
        Ps = np.unique(steps)  # To ensure they are in ascending order

    # Initialize empty arrays to accumulate results of each loop
    nwp_mask = np.zeros_like(im, dtype=bool)
    im_seq = np.zeros_like(im, dtype=int)
    im_pc = np.zeros_like(im, dtype=float)
    im_size = np.zeros_like(im, dtype=float)
    seeds = np.zeros_like(im, dtype=bool)

    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for step, p in enumerate(tqdm(Ps, desc=desc, **settings.tqdm)):
        # Find all locations in image invadable at current pressure
        invadable = (pc <= p) * im  # Equivalent to erosion
        # Trim locations not connected to the inlets, if given
        if inlets is not None:
            invadable = trim_disconnected_voxels(
                im=invadable,
                inlets=inlets,
                conn=conn,
            )
        # Dilate the erosion to find locations of non-wetting phase
        temp = invadable * (~seeds)  # Isolate new locations to speed up inserting
        coords = np.where(temp)  # Find (i, j, k) coordinates of new locations
        radii = dt[coords]  # Extract sphere size to insert at each new location
        # Insert spheres of given radii at new locations
        nwp_mask = _insert_disks_at_points_parallel(
            im=nwp_mask,
            coords=np.vstack(coords),
            radii=radii.astype(int),
            v=True,
            smooth=True,
            overwrite=False,
        )

        # Deal with impact of residual, if present
        if residual is not None:
            if np.any(nwp_mask):
                # Find invadable pixels connected to surviving residual
                temp = trim_disconnected_voxels(residual, nwp_mask, conn=conn) * ~nwp_mask
                if np.any(temp):
                    # Trim invadable pixels not connected to residual
                    invadable = trim_disconnected_voxels(invadable, temp, conn=conn)
                    coords = np.where(invadable)
                    radii = dt[coords].astype(int)
                    nwp_mask = _insert_disks_at_points_parallel(
                        im=nwp_mask,
                        coords=np.vstack(coords),
                        radii=radii.astype(int),
                        v=True,
                        smooth=True,
                        overwrite=False,
                    )
        mask = nwp_mask * (im_seq == 0) * im
        if np.any(mask):
            im_seq[mask] = step + 1
            im_pc[mask] = p
            if np.size(radii) > 0:
                im_size[mask] = np.amin(radii)
        # Add new locations to list of invaded locations
        seeds += invadable

    # Set uninvaded voxels to inf
    im_pc[(im_seq == 0) * im] = np.inf

    # Add residual is given
    if residual is not None:
        im_pc[residual] = -np.inf
        im_seq[residual] = 0

    # Analyze trapping and adjust computed images accordingly
    trapped = None  # Initialize trapped to None in case outlets not given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            method="labels" if len(Ps) < 100 else "queue",
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
        trapped[im_seq == -1] = True
        im_pc[trapped] = np.inf  # Trapped defender only displaced as Pc -> inf
        if residual is not None:  # Re-add residual to inv
            im_pc[residual] = -np.inf  # Residual defender is always present

    # Initialize results object
    results = Results()
    results.im_snwp = pc_to_satn(pc=im_pc, im=im, mode="drainage")
    results.im_seq = im_seq
    # results.im_seq = pc_to_seq(pc=pc_inv, im=im, mode='drainage')
    results.im_pc = im_pc
    results.im_trapped = trapped
    if trapped is not None:
        results.im_seq[trapped] = -1
        results.im_snwp[trapped] = -1
        results.im_pc[trapped] = np.inf
    im_size[im_pc == np.inf] = np.inf
    im_size[im_pc == -np.inf] = -np.inf
    results.im_size = im_size
    results.pc, results.snwp = pc_map_to_pc_curve(
        im=im,
        pc=results.im_pc,
        seq=results.im_seq,
        mode="drainage",
    )
    return results


if __name__ == "__main__":
    from copy import copy

    import matplotlib.pyplot as plt

    import porespy as ps
    ps.visualization.set_mpl_style()

    cm = copy(plt.cm.turbo)
    cm.set_under("grey")
    cm.set_over("k")

    # %% Run this cell to regenerate the variables in drainage
    bg = "white"
    plots = True
    im = ps.generators.blobs(
        shape=[500, 500],
        porosity=0.7,
        blobiness=1.5,
        seed=16,
    )
    im = ps.filters.fill_invalid_pores(im)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    outlets = np.zeros_like(im)
    outlets[-1, :] = True

    lt = ps.filters.local_thickness(im)
    dt = edt(im)
    residual = lt > 25
    steps = 25
    pc = ps.filters.capillary_transform(
        im=im,
        dt=dt,
        sigma=0.072,
        theta=180,
        rho_nwp=1000,
        rho_wp=0,
        g=0,
        voxel_size=1e-4,
    )

    # %% Run different drainage simulations
    drn1 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        steps=30,
        min_size=5,
    )
    drn2 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        steps=30,
    )
    drn3 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        residual=residual,
        steps=30,
    )
    drn5 = ps.simulations.drainage(
        im=im,
        pc=pc,
        steps=30,
    )

    # %% Visualize the invasion configurations for each scenario
    if plots:
        fig, ax = plt.subplot_mosaic(
            [["(a)", "(b)", "(e)", "(e)"], ["(c)", "(d)", "(e)", "(e)"]],
            figsize=[12, 8],
        )
        # drn1.im_pc[~im] = -1
        ax["(a)"].imshow(drn1.im_seq / im, origin="lower", cmap=cm, vmin=0)

        vmax = drn2.im_seq.max()
        ax["(b)"].imshow(drn2.im_seq / im, origin="lower", cmap=cm, vmin=0, vmax=vmax)

        ax["(c)"].imshow(drn3.im_seq / im, origin="lower", cmap=cm, vmin=0)

        ax["(d)"].imshow(drn5.im_seq / im, origin="lower", cmap=cm, vmin=0)

        pc, s = ps.metrics.pc_map_to_pc_curve(
            pc=drn1.im_pc, seq=drn1.im_seq, im=im, mode="drainage"
        )
        ax["(e)"].plot(np.log10(pc), s, "b->", label="drainage")

        pc, s = ps.metrics.pc_map_to_pc_curve(
            pc=drn2.im_pc, seq=drn2.im_seq, im=im, mode="drainage"
        )
        ax["(e)"].plot(np.log10(pc), s, "r-<", label="drainage w trapping")

        pc, s = ps.metrics.pc_map_to_pc_curve(
            pc=drn3.im_pc, seq=drn3.im_seq, im=im, mode="drainage"
        )
        ax["(e)"].plot(np.log10(pc), s, "g-^", label="drainage w residual")

        pc, s = ps.metrics.pc_map_to_pc_curve(
            pc=drn5.im_pc, seq=drn5.im_seq, im=im, mode="drainage"
        )
        ax["(e)"].plot(np.log10(pc), s, "m-*", label="local thickness")

        ax["(e)"].legend()
