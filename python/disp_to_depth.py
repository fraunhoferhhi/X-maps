import cv2
import numba
import numpy as np
from dataclasses import dataclass


@numba.jit(nopython=True, parallel=True, cache=True, error_model="numpy")
def clip_normalize_uint8_depth_frame(depth_frame: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """function to clip a depth map to min and max arguments, normalize to [0,255] and change dtype to np.uint8"""
    height, width = depth_frame.shape
    frame = np.zeros((height, width), dtype=np.uint8)
    min_value, max_value = np.float32(min_value), np.float32(max_value)  # convert min_value and max_value to float32
    range_value = max_value - min_value
    for i in numba.prange(height):
        for j in range(width):
            val = depth_frame[i, j]
            if val != 0:
                val = max(min(val, max_value), min_value)
                val = (val - min_value) / range_value * 255
            frame[i, j] = np.uint8(val)
    return frame


@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def apply_white_mask(frame, norm_frame):
    height, width = norm_frame.shape
    for i in numba.prange(height):
        for j in range(width):
            if norm_frame[i, j] == 0:
                frame[i, j, :] = 255
    return frame


def generate_color_map(norm_frame: np.ndarray) -> None:
    """Generate a colored visualization from the depth map"""
    frame = cv2.applyColorMap(norm_frame, cv2.COLORMAP_TURBO)

    # zero depth represents no depth value, to still be able to find depth at that pixel
    # at the next iteration if color map is projected back, undefined depth values are set to white
    # to create new events
    frame = apply_white_mask(frame, norm_frame)

    return frame


@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True, error_model="numpy")
def disparity_to_depth_rectified(disparity, P1):
    """Function for simplified calculation of depth from disparity.
    This calculation neglects the change in depth caused be the rotation of the rectification.
    If this rotation is small, the error is small."""

    height, width = disparity.shape
    depth = np.zeros((height, width), dtype=np.float32)

    for i in numba.prange(height):
        for j in range(width):
            val = disparity[i, j]
            if val == 0:
                depth[i, j] = 0.0
            else:
                depth[i, j] = max(P1[0, 3] / val, 1e-9)

    return depth


@dataclass
class DisparityToDepth:
    stats: "StatsPrinter"
    calib_params: "CalibParams"
    calib_maps: "CamProjMaps"
    z_near: float
    z_far: float

    dilate_kernel = np.ones((7, 7), dtype=np.uint8)

    def remap_rectified_disp_map_to_proj(self, rectified_disp_map):
        # if projector view is active, dilate pixels
        # projector view is the depth maps from the projectors perspective and with the projectors resolution.
        # Two problems, first, the resolution of the depth map is lower than the projector resolution.
        # Secondly, due how the dispraity search for the projectors perspective works, multiple camera pixels
        # can be mapped to the same projector pixel, while other projector pixel will be left out.
        # For a dense depth map from the projectors point of view, the pixels are dilated.

        # TODO perf this gets faster with a larger kernel.. why?
        with self.stats.measure_time("dilate"):
            disp_map = cv2.dilate(rectified_disp_map, self.dilate_kernel)

        with self.stats.measure_time("remap disp"):
            disp = cv2.remap(
                disp_map,
                map1=self.calib_maps.disp_proj_mapxy_i16,
                map2=None,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )

        return disp

    def colorize_depth_from_disp(self, disp_map: np.ndarray) -> np.ndarray:
        # NOTE: This depth calculatoin is quick but not correct. It does not take into account the
        # change of depth during the rotation back from the rectified coordinate system to the
        # unrectified coordinate system.
        with self.stats.measure_time("d2d_rect"):
            depth_map_f32 = disparity_to_depth_rectified(
                disp_map,
                self.calib_maps.P2,
            )

        with self.stats.measure_time("clip_norm"):
            depth_map_u8 = clip_normalize_uint8_depth_frame(depth_map_f32, min_value=self.z_near, max_value=self.z_far)

        with self.stats.measure_time("color_map"):
            frame = generate_color_map(depth_map_u8)

        return frame
