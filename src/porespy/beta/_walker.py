import ctypes as c
import math
import os
import platform

import numpy as np
import scipy as sp

__all__ = ["walk"]


# Load C library functions
def _load_c_walker_lib():
    base_dir = os.path.dirname(__file__)

    system = platform.system()
    if system == "Windows":
        lib_name = "libwalker.dll"
    elif system == "Darwin":
        lib_name = "libwalker.dylib"
    elif system == "Linux":
        lib_name = "libwalker.so"
    else:
        raise Exception("Invalid OS")

    lib_path = os.path.join(base_dir, lib_name)

    return c.CDLL(lib_path)


class HeatMap2D(c.Structure):
    _fields_ = [
        ("voidMap", c.POINTER(c.c_int)),
        ("solidMap", c.POINTER(c.c_int)),
        ("height", c.c_int),
        ("width", c.c_int),
        ("minLen", c.c_double),
        ("maxLen", c.c_double),
        ("minAngle", c.c_double),
        ("maxAngle", c.c_double),
    ]


class HeatMap3D(c.Structure):
    _fields_ = [
        ("voidMap", c.POINTER(c.c_int)),
        ("solidMap", c.POINTER(c.c_int)),
        ("layers", c.c_int),
        ("height", c.c_int),
        ("width", c.c_int),
        ("minLen", c.c_double),
        ("maxLen", c.c_double),
        ("minTheta", c.c_double),
        ("maxTheta", c.c_double),
        ("minPhi", c.c_double),
        ("maxPhi", c.c_double),
    ]


walker_lib = _load_c_walker_lib()

# Allocates memory: caller must call destroyHeatMap2D
walker_lib.createHeatMap2D.argtypes = [
    c.POINTER(c.c_int),
    c.POINTER(c.c_int),
    c.c_int,
    c.c_int,
    c.c_double,
    c.c_double,
    c.c_double,
    c.c_double,
]

walker_lib.createHeatMap2D.restype = c.POINTER(HeatMap2D)

walker_lib.destroyHeatMap2D.argtypes = [c.POINTER(HeatMap2D)]

walker_lib.walk2D.argtypes = [
    c.POINTER(c.c_bool),
    c.c_int,
    c.c_int,
    c.POINTER(c.c_double),
    c.c_int,
    c.POINTER(HeatMap2D),
    c.POINTER(c.c_int),
]

# Allocates memory: caller must call destroyHeatMap3D
walker_lib.createHeatMap3D.argtypes = [
    c.POINTER(c.c_int),
    c.POINTER(c.c_int),
    c.c_int,
    c.c_int,
    c.c_int,
    c.c_double,
    c.c_double,
    c.c_double,
    c.c_double,
    c.c_double,
    c.c_double,
]

walker_lib.createHeatMap3D.restype = c.POINTER(HeatMap3D)

walker_lib.destroyHeatMap3D.argtypes = [c.POINTER(HeatMap3D)]

walker_lib.walk3D.argtypes = [
    c.POINTER(c.c_bool),
    c.c_int,
    c.c_int,
    c.c_int,
    c.POINTER(c.c_double),
    c.c_int,
    c.POINTER(HeatMap3D),
    c.POINTER(c.c_int),
]


def walk(
    im: np.ndarray,
    iterations: int = 1000000,
    length_bins: int = 100,
    theta_bins: int = 100,
    phi_bins: int = 100,
    min_len: float = None,
    max_len: float = None,
    min_theta: float = -math.pi / 2,
    max_theta: float = math.pi / 2,
    min_phi: float = -math.pi,
    max_phi: float = math.pi,
) -> tuple[np.ndarray, np.ndarray]:
    if min_len is None:
        min_len = 0

    if im.ndim == 2:
        if max_len is None:
            max_len = 0.35 * math.sqrt(
                len(im) * len(im) + len(im[0]) * len(im[0])
            )  # 35% of max possible length
        return _walk2d(
            im, iterations, length_bins, theta_bins, min_len, max_len, min_theta, max_theta
        )
    elif im.ndim == 3:
        if max_len is None:
            max_len = 0.35 * math.sqrt(
                len(im) * len(im) + len(im[0]) * len(im[0]) + len(im[0][0]) * len(im[0][0])
            )
        return _walk3d(
            im,
            iterations,
            length_bins,
            theta_bins,
            phi_bins,
            min_len,
            max_len,
            min_theta,
            max_theta,
            min_phi,
            max_phi,
        )
    else:
        raise Exception("Image must be 2D or 3D")


def _walk2d(im, iterations, map_height, map_width, min_len, max_len, min_angle, max_angle):
    height = len(im)
    width = len(im[0])

    # Distance transform
    im_transform = sp.ndimage.distance_transform_edt(im)

    # Blank image to trace
    path = np.full(shape=[height, width], fill_value=0, dtype=np.int32)

    # Set up buffers and ctypes pointers
    im_c_ptr = np.ascontiguousarray(im).ctypes.data_as(c.POINTER(c.c_bool))
    im_transform_c_ptr = np.ascontiguousarray(im_transform).ctypes.data_as(
        c.POINTER(c.c_double)
    )
    path_c_ptr = np.ascontiguousarray(path).ctypes.data_as(c.POINTER(c.c_int))
    void_map = np.full(shape=[map_height, map_width], fill_value=0, dtype=np.int32)
    solid_map = np.full(shape=[map_height, map_width], fill_value=0, dtype=np.int32)
    void_map_c_ptr = np.ascontiguousarray(void_map).ctypes.data_as(c.POINTER(c.c_int))
    solid_map_c_ptr = np.ascontiguousarray(solid_map).ctypes.data_as(c.POINTER(c.c_int))

    # Allocate memory for heatmap struct
    heat_map_c_ptr = walker_lib.createHeatMap2D(
        void_map_c_ptr,
        solid_map_c_ptr,
        map_height,
        map_width,
        min_len,
        max_len,
        min_angle,
        max_angle,
    )

    # Walk
    walker_lib.walk2D(
        im_c_ptr, height, width, im_transform_c_ptr, iterations, heat_map_c_ptr, path_c_ptr
    )

    # Free heatmap struct
    walker_lib.destroyHeatMap2D(heat_map_c_ptr)

    return void_map, solid_map


def _walk3d(
    im,
    iterations,
    map_layers,
    map_height,
    map_width,
    min_len,
    max_len,
    min_theta,
    max_theta,
    min_phi,
    max_phi,
):
    layers = len(im)
    height = len(im[0])
    width = len(im[0][0])

    # Distance transform
    im_transform = sp.ndimage.distance_transform_edt(im)

    # Blank image to trace
    path = np.full(shape=[layers, height, width], fill_value=0, dtype=np.int32)

    # Set up buffers and ctypes pointers
    im_c_ptr = np.ascontiguousarray(im).ctypes.data_as(c.POINTER(c.c_bool))
    im_transform_c_ptr = np.ascontiguousarray(im_transform).ctypes.data_as(
        c.POINTER(c.c_double)
    )
    path_c_ptr = np.ascontiguousarray(path).ctypes.data_as(c.POINTER(c.c_int))

    void_map = np.full(
        shape=[map_layers, map_height, map_width], fill_value=0, dtype=np.int32
    )
    solid_map = np.full(
        shape=[map_layers, map_height, map_width], fill_value=0, dtype=np.int32
    )
    void_map_c_ptr = np.ascontiguousarray(void_map).ctypes.data_as(c.POINTER(c.c_int))
    solid_map_c_ptr = np.ascontiguousarray(solid_map).ctypes.data_as(c.POINTER(c.c_int))

    heat_map_c_ptr = walker_lib.createHeatMap3D(
        void_map_c_ptr,
        solid_map_c_ptr,
        map_layers,
        map_height,
        map_width,
        min_len,
        max_len,
        min_theta,
        max_theta,
        min_phi,
        max_phi,
    )

    # Walk
    walker_lib.walk3D(
        im_c_ptr,
        layers,
        height,
        width,
        im_transform_c_ptr,
        iterations,
        heat_map_c_ptr,
        path_c_ptr,
    )

    return void_map, solid_map
