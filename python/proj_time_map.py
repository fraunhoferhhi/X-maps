import numpy as np
import cv2


class ProjectorTimeMap:
    def __init__(self, calib, proj_time_map_path=None):
        if proj_time_map_path:
            self.projector_time_map_rectified = np.load(proj_time_map_path)
        else:
            projector_time_map = self.generate_linear_projector_time_map(
                (calib.projector_width, calib.projector_height)
            )
            self.projector_time_map_rectified = self.remap_proj_time_map(calib, projector_time_map)

    def generate_linear_projector_time_map(self, projector_shape: tuple) -> np.ndarray:
        """generate linear projector time map"""
        # create a range of width*height as ndarray
        projector_time_map: np.ndarray = np.arange(projector_shape[0] * projector_shape[1], dtype=np.float32)
        # normilize this range to intervall 0 to 1
        projector_time_map /= projector_shape[0] * projector_shape[1]
        # reshape to projector shape
        projector_time_map = projector_time_map.reshape((projector_shape[0], projector_shape[1]))
        # rotate 90 counterclockwise to account for projector rotation
        projector_time_map = np.rot90(projector_time_map)
        return projector_time_map

    def remap_proj_time_map(self, calib, proj_time_map) -> np.ndarray:
        # NOTE: ESL implementation uses BORDER_CONSTANT
        return cv2.remap(
            proj_time_map,
            calib.projector_mapx,
            calib.projector_mapy,
            cv2.INTER_NEAREST,
            cv2.BORDER_REPLICATE,
        )
