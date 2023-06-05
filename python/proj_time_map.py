import numpy as np
import cv2
from dataclasses import dataclass


def generate_linear_projector_time_map(proj_width: int, proj_height: int, scan_upwards: bool) -> np.ndarray:
    # x and y coordinates of projector pixels
    ys, xs = np.mgrid[0:proj_height, 0:proj_width]

    if scan_upwards:
        # invert y axis to scan from bottom to top
        ys = ys[::-1]

    # scan in x direction (right) first, than the y direction (determined by scan_upwards)
    pixel_indeces = xs * proj_height + ys

    projector_time_map = pixel_indeces / (proj_width * proj_height)

    return projector_time_map.astype(np.float32)


def remap_proj_time_map(calib, proj_time_map, border_mode) -> np.ndarray:
    return cv2.remap(
        proj_time_map,
        calib.projector_mapx,
        calib.projector_mapy,
        cv2.INTER_NEAREST,
        border_mode,
    )


@dataclass
class ProjectorTimeMap:
    projector_time_map_rectified: np.ndarray

    @staticmethod
    def from_calib(calib, scan_upwards=True, remap_border_mode=cv2.BORDER_REPLICATE):
        projector_time_map = generate_linear_projector_time_map(
            calib.projector_width, calib.projector_height, scan_upwards
        )
        projector_time_map_rectified = remap_proj_time_map(calib, projector_time_map, border_mode=remap_border_mode)
        return ProjectorTimeMap(projector_time_map_rectified)

    @staticmethod
    def from_file(proj_time_map_path):
        projector_time_map_rectified = np.load(proj_time_map_path)
        return ProjectorTimeMap(projector_time_map_rectified)
