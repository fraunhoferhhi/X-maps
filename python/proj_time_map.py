import numpy as np
import cv2


class ProjectorTimeMap:
    def __init__(self, calib, proj_time_map_path=None, scan_upwards=True, remap_border_mode=cv2.BORDER_REPLICATE):
        if proj_time_map_path:
            self.projector_time_map_rectified = np.load(proj_time_map_path)
        else:
            self.projector_time_map = self.generate_linear_projector_time_map(
                (calib.projector_width, calib.projector_height), scan_upwards
            )
            self.projector_time_map_rectified = self.remap_proj_time_map(
                calib, self.projector_time_map, border_mode=remap_border_mode
            )

    def generate_linear_projector_time_map(self, projector_shape: tuple, scan_upwards: bool) -> np.ndarray:
        # x and y coordinates of projector pixels
        ys, xs = np.mgrid[0 : projector_shape[1], 0 : projector_shape[0]]

        if scan_upwards:
            # invert y axis to scan from bottom to top
            ys = ys[::-1]

        # scan in x direction (right) first, than the y direction (determined by scan_upwards)
        pixel_indeces = xs * projector_shape[1] + ys

        projector_time_map = pixel_indeces / (projector_shape[0] * projector_shape[1])

        return projector_time_map.astype(np.float32)

    def remap_proj_time_map(self, calib, proj_time_map, border_mode) -> np.ndarray:
        # NOTE: ESL implementation uses BORDER_CONSTANT
        return cv2.remap(
            proj_time_map,
            calib.projector_mapx,
            calib.projector_mapy,
            cv2.INTER_NEAREST,
            border_mode,
        )
