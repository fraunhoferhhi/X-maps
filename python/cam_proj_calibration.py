from typing import Optional

from dataclasses import dataclass

import numpy as np
import cv2
import os
import yaml


def open_calibration_data(calib_data_path: str) -> Optional[dict]:
    """function to open the calibration.yaml file"""
    if os.path.exists(calib_data_path):
        with open(calib_data_path, "r") as file:
            calibration_data = yaml.safe_load(file)
        return calibration_data
    else:
        print(f"File not found at {calib_data_path}")
        return None


def read_cv_matrix(calibration_data: dict, name: str) -> Optional[np.ndarray]:
    """function to read an opencv matrix from yaml-dict as written by the calibration application"""
    if (
        name in calibration_data
        and "type-id" in calibration_data[name]
        and calibration_data[name]["type-id"] == "opencv_matrix"
    ):
        cols = calibration_data[name]["cols"]
        rows = calibration_data[name]["rows"]
        return np.array(calibration_data[name]["data"]).reshape(rows, cols)
    else:
        print(f"Could not find matrix {name} in calibration data")
        return None


def initUndistortRectifyMapInverse(cameraMatrix, distCoeffs, R, newCameraMatrix, size):
    """Function to generate a map from a undistorted rectified image back to the original image."""
    # unpack size tuple
    W, H = size
    # create an array of all coordinate pairs in image with specified width and height
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).T.reshape((-1, 1, 2)).astype("float32")
    # Get points in camera space, undistort these points and project back to image plane
    points = cv2.undistortPoints(coords, cameraMatrix, distCoeffs, None, R, newCameraMatrix)
    # reshape to fit map_x and map_y format
    maps = points.reshape((H, W, 2))
    return maps[..., 0], maps[..., 1]


def map_to_i16(mapx_f32: np.ndarray, mapy_f32: np.ndarray) -> np.ndarray:
    assert mapx_f32.dtype == np.float32 and mapy_f32.dtype == np.float32
    mapx_i = np.round(mapx_f32)
    mapy_i = np.round(mapy_f32)
    assert mapx_i.min() >= np.iinfo(np.int16).min and mapx_i.max() <= np.iinfo(np.int16).max
    assert mapy_i.min() >= np.iinfo(np.int16).min and mapy_i.max() <= np.iinfo(np.int16).max
    return np.stack((mapx_i.astype(np.int16), mapy_i.astype(np.int16)), axis=-1)


@dataclass
class CamProjCalibration:
    def __init__(self, calibration_yaml_path, camera_width, camera_height, projector_width, projector_height):
        """initialize all stereo parameters from calibration file"""

        calibration_data = open_calibration_data(calibration_yaml_path)

        self.camera_width = camera_width
        self.camera_height = camera_height

        self.projector_width = projector_width
        self.projector_height = projector_height

        # TODO make this parameter configurable, compute from camera and projector resolution?
        rectification_scale = 2.75
        self.rect_image_width = round(camera_width * rectification_scale)
        self.rect_image_height = round(camera_height * rectification_scale)

        self.camera_K = read_cv_matrix(calibration_data, "camera_intrinsic_matrix")
        self.camera_D = read_cv_matrix(calibration_data, "camera_distortion_coefficients")

        self.projector_K = read_cv_matrix(calibration_data, "projector_intrinsic_matrix")
        self.projector_D = read_cv_matrix(calibration_data, "projector_distortion_coefficients")

        self.R = read_cv_matrix(calibration_data, "relative_rotation")
        self.T = read_cv_matrix(calibration_data, "relative_translation")

        if "F" in calibration_data:
            self.F = read_cv_matrix(calibration_data, "F")
        else:
            self.F = read_cv_matrix(calibration_data, "fundamental_matrix")

        # TODO: projector distortion coefficients are currently ignored
        self.projector_D_backup = self.projector_D.copy()
        self.projector_D = np.zeros((5,))

        # calculate stereo rectification from camera and projector instrinsics and extrinsics
        (
            self.R1,
            self.R2,
            self.P1,
            self.P2,
            self.Q,
            self.validPixROI1,
            self.validPixROI2,
        ) = cv2.stereoRectify(
            cameraMatrix1=self.camera_K,
            distCoeffs1=self.camera_D,
            cameraMatrix2=self.projector_K,
            distCoeffs2=self.projector_D,
            imageSize=(self.rect_image_width, self.rect_image_height),
            R=self.R,
            T=self.T,
            alpha=-1,
        )

        # calculate inverse of rectification rotation to be able to project back to unrectified space
        self.R1_inv = np.linalg.inv(self.R1)
        self.R2_inv = np.linalg.inv(self.R2)

        # utils.write_cv_matrix(calibration_data, 'R1', self.R1)
        # utils.write_cv_matrix(calibration_data, 'P1', self.P1)
        # utils.write_cv_matrix(calibration_data, 'R2', self.R2)
        # utils.write_cv_matrix(calibration_data, 'P2', self.P2)
        # utils.write_cv_matrix(calibration_data, 'Q', self.Q)
        # utils.write_cv_size(calibration_data, 'rectification_image_size', (self.image_width, self.image_height))

        # calculate lookup table to easily rectify camera image/time map with cv2.remap
        self.camera_mapx, self.camera_mapy = cv2.initUndistortRectifyMap(
            self.camera_K,
            self.camera_D,
            self.R1,
            self.P1,
            (self.rect_image_width, self.rect_image_height),
            cv2.CV_32FC1,
        )

        # calculate lookup table to easily rectify projector image/time map with cv2.remap
        self.projector_mapx, self.projector_mapy = cv2.initUndistortRectifyMap(
            self.projector_K,
            self.projector_D,
            self.R2,
            self.P2,
            (self.rect_image_width, self.rect_image_height),
            cv2.CV_32FC1,
        )

        # calculate lookup table to easily unrectify camera image/time map with cv2.remap
        self.disp_cam_mapx, self.disp_cam_mapy = initUndistortRectifyMapInverse(
            self.camera_K, self.camera_D, self.R1, self.P1, (self.camera_width, self.camera_height)
        )

        # calculate lookup table to easily calculate undistorted camera image
        self.disp_cam_mapx_undist, self.disp_cam_mapy_undist = initUndistortRectifyMapInverse(
            self.camera_K, self.camera_D, np.eye(3), self.camera_K, (self.camera_width, self.camera_height)
        )

        # calculate lookup table to easily unrectify projector image/time map with cv2.remap
        self.disp_proj_mapx, self.disp_proj_mapy = initUndistortRectifyMapInverse(
            self.projector_K,
            self.projector_D,
            self.R2,
            self.P2,
            (self.projector_width, self.projector_height),
        )

        self.disp_proj_mapxy_i16 = map_to_i16(self.disp_proj_mapx, self.disp_proj_mapy)
