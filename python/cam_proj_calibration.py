from typing import Optional

from dataclasses import dataclass, field, InitVar

import numpy as np
import cv2
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


def mapf_to_i16(map_f32: np.ndarray) -> np.ndarray:
    assert map_f32.dtype == np.float32
    map_i = np.rint(map_f32)
    assert map_i.min() >= np.iinfo(np.int16).min and map_i.max() <= np.iinfo(np.int16).max
    return map_i.astype(np.int16)


def mapxy_to_i16s2(mapx_f32: np.ndarray, mapy_f32: np.ndarray) -> np.ndarray:
    return np.stack((mapf_to_i16(mapx_f32), mapf_to_i16(mapy_f32)), axis=-1)


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
        rectification_scale: float = 3

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
            rect_image_width=round(projector_width * rectification_scale),
            rect_image_height=round(projector_height * rectification_scale),
            camera_K=cam_int,
            camera_D=cam_dist,
            projector_K=proj_int,
            projector_D=proj_dist,
            cam2proj_R=cam_proj_rmat,
            cam2proj_T=cam_proj_tvec,
        )


@dataclass
class CamProjMaps:
    calib: CamProjCalibrationParams

    cam_is_left: InitVar[bool] = False
    zero_undistort_proj_map: InitVar[bool] = False

    # TODO move to params
    R1: np.ndarray = field(init=False)
    R2: np.ndarray = field(init=False)

    P1: np.ndarray = field(init=False)
    P2: np.ndarray = field(init=False)

    Q: np.ndarray = field(init=False)

    # lookup table to rectify camera image/time map with cv2.remap
    camera_mapx: np.ndarray = field(init=False)
    camera_mapy: np.ndarray = field(init=False)

    # lookup tables to rectify projector image/time map with cv2.remap
    projector_mapx: np.ndarray = field(init=False)
    projector_mapy: np.ndarray = field(init=False)

    # lookup table to unrectify camera image/time map with cv2.remap
    disp_cam_mapx: np.ndarray = field(init=False)
    disp_cam_mapy: np.ndarray = field(init=False)

    # lookup table to unrectify projector image/time map with cv2.remap
    disp_proj_mapxy_i16: np.ndarray = field(init=False)

    def __post_init__(self, cam_is_left, zero_undistort_proj_map):
        """Provide maps to map camera and projector images to rectified images and vice versa.

        Default params are used in X-maps.

        Invert the bool params to mirror ESL processing.

        Args:
            calib: calibration parameters
            cam_is_first_cam: True if camera is first camera in stereo pair, False if projector is first camera for cv2.stereoRectify
            zero_undistort_proj_map: True if projector distortion should be ignored in projector rectification map
        """

        if cam_is_left:
            rectify_params = {
                "cameraMatrix1": self.calib.camera_K,
                "distCoeffs1": self.calib.camera_D,
                "cameraMatrix2": self.calib.projector_K,
                "distCoeffs2": self.calib.projector_D,
            }
        else:
            rectify_params = {
                "cameraMatrix1": self.calib.projector_K,
                "distCoeffs1": self.calib.projector_D,
                "cameraMatrix2": self.calib.camera_K,
                "distCoeffs2": self.calib.camera_D,
            }

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
            imageSize=(self.calib.rect_image_width, self.calib.rect_image_height),
            R=self.calib.cam2proj_R,
            T=self.calib.cam2proj_T,
            alpha=-1,
            **rectify_params,
        )

        # calculate inverse of rectification rotation to be able to project back to unrectified space
        # R1_inv = np.linalg.inv(self.R1)
        # R2_inv = np.linalg.inv(self.R2)

        # TODO perf: why isn't this signed i16?
        self.camera_mapx, self.camera_mapy = cv2.initUndistortRectifyMap(
            self.calib.camera_K,
            self.calib.camera_D,
            self.R1,
            self.P1,
            (self.calib.rect_image_width, self.calib.rect_image_height),
            cv2.CV_32FC1,
        )

        # ESL compatibility: projector distortion is ignored here, but still used in cv2.stereoRectify
        proj_dist = np.zeros(5) if zero_undistort_proj_map else self.calib.projector_D

        # TODO perf: why isn't this signed i16?
        self.projector_mapx, self.projector_mapy = cv2.initUndistortRectifyMap(
            self.calib.projector_K,
            proj_dist,
            self.R2,
            self.P2,
            (self.calib.rect_image_width, self.calib.rect_image_height),
            cv2.CV_32FC1,
        )

        self.disp_cam_mapx_f32, self.disp_cam_mapy_f32 = initUndistortRectifyMapInverse(
            self.calib.camera_K,
            self.calib.camera_D,
            self.R1,
            self.P1,
            (self.calib.camera_width, self.calib.camera_height),
        )
        self.disp_cam_mapx_i16 = mapf_to_i16(self.disp_cam_mapx_f32)
        self.disp_cam_mapy_i16 = mapf_to_i16(self.disp_cam_mapy_f32)

        # calculate lookup table to calculate undistorted camera image
        # disp_cam_mapx_undist, disp_cam_mapy_undist = initUndistortRectifyMapInverse(
        #     self.calib.camera_K, self.calib.camera_D, np.eye(3), self.calib.camera_K, (self.calib.camera_width, self.calib.camera_height)
        # )

        # calculate lookup table to unrectify projector image/time map with cv2.remap
        disp_proj_mapx, disp_proj_mapy = initUndistortRectifyMapInverse(
            self.calib.projector_K,
            self.calib.projector_D,
            self.R2,
            self.P2,
            (self.calib.projector_width, self.calib.projector_height),
        )

        self.disp_proj_mapxy_i16 = mapxy_to_i16s2(disp_proj_mapx, disp_proj_mapy)

    def rectify_cam_coords_f32(self, events):
        xcr_f32 = self.disp_cam_mapx_f32[events["y"], events["x"]]
        ycr_f32 = self.disp_cam_mapy_f32[events["y"], events["x"]]
        return xcr_f32, ycr_f32

    def rectify_cam_coords_i16(self, events):
        # TODO perf ys =; xs=?
        xcr_i16 = self.disp_cam_mapx_i16[events["y"], events["x"]]
        ycr_i16 = self.disp_cam_mapy_i16[events["y"], events["x"]]
        return xcr_i16, ycr_i16

    def round_rectified_y_coords(self, ev_y_rect_f32, inlier_mask):
        # TODO + 0.5?
        yr_i16 = np.rint(ev_y_rect_f32[inlier_mask]).astype(np.int16)
        return yr_i16

    def compute_disp_map(self):
        # ypr_dispf = np.rint(ypr_f32[inlier_mask]).astype(np.int16)
        # xpr_dispf = np.rint(xpr_f32[inlier_mask]).astype(np.int16)

        # TODO: choice between ypr and ycr
        # disp_map[ypr_dispf, xpr_dispf] = disp[inlier_mask]

        # TODO should one of the disparities actually be negative, assuption: no, since then the baseline also must be negative and it would cancel out ? (N.G.)
        # TODO instead of using the rounded rectified coordinates, be could also use the input event coordinates
        pass

    def compute_disp_map_projector_view(self, ev_x_rect_i16, ev_y_rect_i16, inlier_mask, ev_disparity_f32):
        xpr_i16 = np.rint(ev_x_rect_i16[inlier_mask] + ev_disparity_f32).astype(np.int16)
        disp_map = np.zeros((self.calib.rect_image_height, self.calib.rect_image_width), dtype=np.float32)
        disp_map[ev_y_rect_i16[inlier_mask], xpr_i16] = ev_disparity_f32
        return disp_map

    # def compute_disp_map_rect_camera_view(self, ev_x_rect_f32, ev_y_rect_f32, inlier_mask, ev_disparity_f32):
    #     ycr_i16 = self.round_rectified_y_coords(ev_y_rect_f32, inlier_mask)
    #     xcr_i16 = np.rint(ev_x_rect_f32[inlier_mask]).astype(np.int16)
    #     disp_map = np.zeros((self.calib.rect_image_height, self.calib.rect_image_width), dtype=np.float32)
    #     disp_map[ycr_i16, xcr_i16] = ev_disparity_f32
    #     return disp_map

    def compute_disp_map_camera_view(self, events, inlier_mask, ev_disparity_f32):
        x_cam = events["x"][inlier_mask]
        y_cam = events["y"][inlier_mask]
        disp_map = np.zeros((self.calib.camera_height, self.calib.camera_width), dtype=np.float32)
        disp_map[y_cam, x_cam] = ev_disparity_f32
        return disp_map

    def construct_point_cloud(self, xpr_f32, ypr_f32, disp_f32):
        points = np.ones((len(xpr_f32), 4), dtype=np.float32)
        points[:, 0] = xpr_f32 + disp_f32
        points[:, 1] = ypr_f32
        points[:, 2] = -disp_f32
        point_cloud = (self.Q.astype(np.float32) @ points.T).T
        point_cloud = (point_cloud / point_cloud[:, 3:])[:, :3]

        # invert y and z axis
        point_cloud[:, 1] = -point_cloud[:, 1]
        point_cloud[:, 2] = -point_cloud[:, 2]

        return point_cloud
