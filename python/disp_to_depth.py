import cv2
import numba
import numpy as np


def clip_depth_img(depth_img: np.ndarray, min_value: float, max_value: float):
    """function to clip frame to min and max arguments"""
    # as 0 depth represents no depth value, zeros must be preserved
    # get a boolean map of all zeros
    if min_value > 0:
        bool_map = depth_img == 0
    # clip values
    np.clip(depth_img, a_min=min_value, a_max=max_value, out=depth_img)
    # reset zeros according to boolean map
    if min_value > 0:
        depth_img[bool_map] = 0


def clip_normalize_uint8_depth_frame(
    depth_frame: np.ndarray, min_value: float, max_value: float, preserve_zeros: bool = True
) -> np.ndarray:
    """function to clip a depth map to min and max arguments, normalize to [0,255] and cahnge dtype to np.uint8"""
    # copy to new frame in memory
    frame = depth_frame.copy()
    # as 0 depth represents no depth value, zeros must be preserved
    # get a boolean map of all zeros
    if preserve_zeros:
        bool_map = frame == 0
    # clip frame to min and max values
    clip_depth_img(frame, min_value, max_value)
    # scale to [0,255]
    frame -= min_value
    frame = frame / (max_value - min_value)
    frame *= 255
    # reset zeros according to boolean map
    if preserve_zeros:
        frame[bool_map] = 0
    # convert dtype to np.uint8
    return frame.astype(np.uint8)


@numba.jit(nopython=True, parallel=False, fastmath=True)
def apply_white_mask(frame, norm_frame):
    height, width = norm_frame.shape
    for i in numba.prange(height):
        for j in numba.prange(width):
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



def disparity_to_depth_rectified(disparity, P1):
    """Function for simplified calculation of depth from disparity.
    This calculation neglects the change in depth caused be the rotation of the rectification.
    If this rotation is small, the error is small."""

    disparity_no_div_error = np.clip(disparity, 1e-9, None)

    depth = P1[0, 3] / disparity_no_div_error
    depth[disparity == 0] = 0.0

    return depth


class DisparityToDepth:
    def __init__(self, stats, calib, z_near, z_far):
        self.stats = stats
        self.calib = calib
        self.z_near = z_near
        self.z_far = z_far

        self.dilate_kernel = np.ones((7, 7), dtype=np.uint8)

    def compute_depth_map(self, disp_map):
        # if projector view is active, dilate pixels
        # projector view is the depth maps from the projectors perspective and with the projectors resolution.
        # Two problems, first, the resolution of the depth map is lower than the projector resolution.
        # Secondly, due how the dispraity search for the projectors perspective works, multiple camera pixels
        # can be mapped to the same projector pixel, while other projector pixel will be left out.
        # For a dense depth map from the projectors point of view, the pixels are dilated.

        # TODO perf this gets faster with a larger kernel.. why?
        with self.stats.measure_time("dilate"):
            disp_map = cv2.dilate(disp_map, self.dilate_kernel)

        # get depth map from rectified disparity map and append to list of depth maps
        # NOTE: This depth calculatoin is quick but not correct. It does not take into account the
        # change of depth during the rotation back from the rectified coordinate system to the
        # unrectified coordinate system.
        with self.stats.measure_time("d2d_rect"):

            depth_map_f32 = disparity_to_depth_rectified(
                cv2.remap(
                    disp_map, self.calib.disp_proj_mapx, self.calib.disp_proj_mapy, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT
                ),
                self.calib.P2,
            )

        # TODO perf Numba impl?
        with self.stats.measure_time("clip_norm"):
            depth_map_u8 = clip_normalize_uint8_depth_frame(depth_map_f32, min_value=self.z_near, max_value=self.z_far)

        with self.stats.measure_time("color_map"):
            frame = generate_color_map(depth_map_u8)

        return frame
