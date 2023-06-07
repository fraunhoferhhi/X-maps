from typing import Optional

from dataclasses import dataclass

import numpy as np
import cv2
import os
import yaml


def open_calibration_data(calib_data_path: str) -> dict:
    """function to open the calibration.yaml file"""
    with open(calib_data_path, "r") as file:
        calibration_data = yaml.safe_load(file)
    return calibration_data


def read_cv_matrix(calibration_data: dict, name: str) -> np.ndarray:
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
        raise ValueError(f"Could not read matrix {name} from calibration data")


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
    mapx_i = np.rint(mapx_f32)
    mapy_i = np.rint(mapy_f32)
    assert mapx_i.min() >= np.iinfo(np.int16).min and mapx_i.max() <= np.iinfo(np.int16).max
    assert mapy_i.min() >= np.iinfo(np.int16).min and mapy_i.max() <= np.iinfo(np.int16).max
    return np.stack((mapx_i.astype(np.int16), mapy_i.astype(np.int16)), axis=-1)


@dataclass
class CamProjCalibrationParams:
    camera_width: int
    camera_height: int

    projector_width: int
    projector_height: int

    rect_image_width: int
    rect_image_height: int

    camera_K: np.ndarray
    camera_D: np.ndarray

    projector_K: np.ndarray
    projector_D: np.ndarray

    cam2proj_R: np.ndarray
    cam2proj_T: np.ndarray

    F: Optional[np.ndarray] = None

    @staticmethod
    def from_yaml(
        calibration_yaml_path: str, camera_width: int, camera_height: int, projector_width: int, projector_height: int
    ):
        calibration_data = open_calibration_data(calibration_yaml_path)

        # TODO make this parameter configurable, compute from camera and projector resolution?
        rectification_scale: float = 2.75

        # TODO: projector distortion coefficients are currently ignored
        projector_D = read_cv_matrix(calibration_data, "projector_distortion_coefficients")
        projector_D_backup = projector_D.copy()
        projector_D = np.zeros((5,))

        return CamProjCalibrationParams(
            camera_width=camera_width,
            camera_height=camera_height,
            projector_width=projector_width,
            projector_height=projector_height,
            # TODO move to init
            rect_image_width=round(camera_width * rectification_scale),
            rect_image_height=round(camera_height * rectification_scale),
            camera_K=read_cv_matrix(calibration_data, "camera_intrinsic_matrix"),
            camera_D=read_cv_matrix(calibration_data, "camera_distortion_coefficients"),
            projector_K=read_cv_matrix(calibration_data, "projector_intrinsic_matrix"),
            projector_D=projector_D,
            cam2proj_R=read_cv_matrix(calibration_data, "relative_rotation"),
            cam2proj_T=read_cv_matrix(calibration_data, "relative_translation"),
            F=read_cv_matrix(calibration_data, "F")
            if "F" in calibration_data
            else read_cv_matrix(calibration_data, "fundamental_matrix"),
        )

    @staticmethod
    def from_ESL_yaml(
        calibration_yaml_path: str, camera_width: int, camera_height: int, projector_width: int, projector_height: int
    ):
        print("Reading calibration from file: {0}".format(calibration_yaml_path))

        # TODO make this parameter configurable, compute from camera and projector resolution?
        rectification_scale: float = 2.75

        fs = cv2.FileStorage(calibration_yaml_path, cv2.FILE_STORAGE_READ)
        cam_int = fs.getNode("cam_K").mat()
        cam_dist = fs.getNode("cam_kc").mat()
        proj_int = fs.getNode("proj_K").mat()
        proj_dist = fs.getNode("proj_kc").mat()
        cam_proj_rmat = fs.getNode("R").mat()
        cam_proj_tvec = fs.getNode("T").mat()

        return CamProjCalibrationParams(
            camera_width=camera_width,
            camera_height=camera_height,
            projector_width=projector_width,
            projector_height=projector_height,
            rect_image_width=round(camera_width * rectification_scale),
            rect_image_height=round(camera_height * rectification_scale),
            camera_K=cam_int,
            camera_D=cam_dist,
            projector_K=proj_int,
            projector_D=proj_dist,
            cam2proj_R=cam_proj_rmat,
            cam2proj_T=cam_proj_tvec,
        )
        # R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        #     proj_int,
        #     proj_dist,
        #     cam_int,
        #     cam_dist,
        #     shape,
        #     cam_proj_rmat,
        #     cam_proj_tvec,
        #     # flags=cv2.CALIB_ZERO_DISPARITY,
        #     alpha=alpha,
        # )
        # T = np.zeros((4, 4), np.float32)
        # T[:3, :3] = cam_proj_rmat
        # T[:3, 3] = cam_proj_tvec[:, 0]
        # # T_inv = np.zeros((4, 4), np.float32)
        # # T_inv[:3, :3] = cam_proj_rmat.T
        # # T_inv[:3, 3] = -np.dot(cam_proj_rmat.T, cam_proj_tvec[:, 0])
        # return calib(cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, R1, R2, P1, P2, Q, T)


@dataclass
class CamProjMaps:
    # TODO move to params
    R1: np.ndarray
    R2: np.ndarray

    P1: np.ndarray
    P2: np.ndarray

    Q: np.ndarray

    # lookup table to easily rectify camera image/time map with cv2.remap
    camera_mapx: np.ndarray
    camera_mapy: np.ndarray

    # lookup tables to easily rectify projector image/time map with cv2.remap
    projector_mapx: np.ndarray
    projector_mapy: np.ndarray

    # lookup table to easily unrectify camera image/time map with cv2.remap
    disp_cam_mapx: np.ndarray
    disp_cam_mapy: np.ndarray

    # lookup table to easily unrectify projector image/time map with cv2.remap
    disp_proj_mapxy_i16: np.ndarray

    def __init__(self, calib: CamProjCalibrationParams):
        # calculate stereo rectification from camera and projector instrinsics and extrinsics
        (
            self.R1,
            self.R2,
            self.P1,
            self.P2,
            self.Q,
            validPixROI1,
            validPixROI2,
        ) = cv2.stereoRectify(
            cameraMatrix1=calib.camera_K,
            distCoeffs1=calib.camera_D,
            cameraMatrix2=calib.projector_K,
            distCoeffs2=calib.projector_D,
            imageSize=(calib.rect_image_width, calib.rect_image_height),
            R=calib.R,
            T=calib.T,
            alpha=-1,
        )

        # calculate inverse of rectification rotation to be able to project back to unrectified space
        R1_inv = np.linalg.inv(self.R1)
        R2_inv = np.linalg.inv(self.R2)

        # utils.write_cv_matrix(calibration_data, 'R1', R1)
        # utils.write_cv_matrix(calibration_data, 'P1', P1)
        # utils.write_cv_matrix(calibration_data, 'R2', R2)
        # utils.write_cv_matrix(calibration_data, 'P2', P2)
        # utils.write_cv_matrix(calibration_data, 'Q', Q)
        # utils.write_cv_size(calibration_data, 'rectification_image_size', (image_width, image_height))

        # TODO why isn't this signed i16?
        self.camera_mapx, self.camera_mapy = cv2.initUndistortRectifyMap(
            calib.camera_K,
            calib.camera_D,
            self.R1,
            self.P1,
            (calib.rect_image_width, calib.rect_image_height),
            cv2.CV_32FC1,
        )

        # TODO why isn't this signed i16?
        self.projector_mapx, self.projector_mapy = cv2.initUndistortRectifyMap(
            calib.projector_K,
            calib.projector_D,
            self.R2,
            self.P2,
            (calib.rect_image_width, calib.rect_image_height),
            cv2.CV_32FC1,
        )

        self.disp_cam_mapx, self.disp_cam_mapy = initUndistortRectifyMapInverse(
            calib.camera_K, calib.camera_D, self.R1, self.P1, (calib.camera_width, calib.camera_height)
        )

        # calculate lookup table to easily calculate undistorted camera image
        disp_cam_mapx_undist, disp_cam_mapy_undist = initUndistortRectifyMapInverse(
            calib.camera_K, calib.camera_D, np.eye(3), calib.camera_K, (calib.camera_width, calib.camera_height)
        )

        # calculate lookup table to easily unrectify projector image/time map with cv2.remap
        disp_proj_mapx, disp_proj_mapy = initUndistortRectifyMapInverse(
            calib.projector_K,
            calib.projector_D,
            self.R2,
            self.P2,
            (calib.projector_width, calib.projector_height),
        )

        self.disp_proj_mapxy_i16 = map_to_i16(disp_proj_mapx, disp_proj_mapy)
