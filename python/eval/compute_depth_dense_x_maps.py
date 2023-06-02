# coding: UTF-8
import argparse
import copy
import glob
import json
import os
import os.path
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from esl_utilities import utils as ut
import pandas as pd
from pyntcloud import PyntCloud

import warnings

import sys

from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from x_maps_disparity import XMapsDisparity


warnings.filterwarnings("ignore")


@dataclass
class XMapsCalib:
    disp_cam_mapx: np.ndarray
    disp_cam_mapy: np.ndarray
    Q: np.ndarray


@dataclass
class XMapsProjTimeMap:
    projector_time_map_rectified: np.ndarray


def initUndistortRectifyMapInverse(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type):
    W, H = size
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).T.reshape((-1, 1, 2)).astype("float32")
    points = cv2.undistortPoints(coords, cameraMatrix, distCoeffs, None, R, newCameraMatrix)
    maps = points.reshape((H, W, 2))
    return maps[..., 0], maps[..., 1]


def project_and_backproject_punkt(point, cam_K, cam_kc, pro_K, pro_kc, T_to_cam, Z):
    intrinsic = cam_K
    distortion = cam_kc
    points_undistorted = cv2.undistortPoints(point, intrinsic, distortion, P=intrinsic)
    points_undistorted = np.squeeze(points_undistorted, axis=1)
    z = Z
    x = ((points_undistorted[0, 0] - intrinsic[0, 2]) / intrinsic[0, 0]) * z
    y = ((points_undistorted[0, 1] - intrinsic[1, 2]) / intrinsic[1, 1]) * z
    result = np.array([x, y, z]).reshape(3, 1)

    rvec, _ = cv2.Rodrigues(T_to_cam[:3, :3])
    tvec = np.array(T_to_cam[:3, 3])
    intrinsic = pro_K
    distortion = pro_kc
    result, _ = cv2.projectPoints(result, rvec, tvec, intrinsic, distortion)
    return np.squeeze(result, axis=1)


def cost_calculator(
    rho, point, t_event, t_proj, window_size, cam0, cam_dist, proj, proj_dist, T, cost_arr=None, rho_arr=None
):
    scale_factor = 1
    point_2d_proj = project_and_backproject_punkt(point, cam0, cam_dist, proj, proj_dist, T, rho)

    x_cam, y_cam = point[0].astype(np.int32)
    x_proj, y_proj = point_2d_proj[0].astype(np.int32)
    w = int(window_size / 2)

    if (
        (y_proj - w * scale_factor) > 0
        and (y_proj + w * scale_factor) < t_proj.shape[0]
        and (x_proj - w * scale_factor) > 0
        and (x_proj + w * scale_factor) < t_proj.shape[1]
    ):
        projector_patch = t_proj[y_proj - w : y_proj + w + 1, x_proj - w : x_proj + w + 1]
        event_patch = t_event[y_cam - w : y_cam + w + 1, x_cam - w : x_cam + w + 1]
        cost = np.linalg.norm(event_patch - projector_patch)
        if cost_arr is not None:
            cost_arr.append(cost)
            rho_arr.append(rho)
    else:
        cost = 100000
    return cost


def disparity_init(cam_img_rectified, proj_img_rectified):
    disparity = np.zeros(cam_img_rectified.shape)
    min_disp, max_disp = 5, 900
    r, c = np.where(cam_img_rectified > 0)
    for i in range(len(r)):
        num_nonzero = np.nonzero(proj_img_rectified[r[i], c[i] + min_disp : c[i] + max_disp])[0]
        projector_patches = proj_img_rectified[r[i], c[i] + min_disp + num_nonzero]
        if len(num_nonzero) > 1:
            event_patch = cam_img_rectified[r[i], c[i]]
            cost = (projector_patches - event_patch) ** 2
            c_proj = c[i] + min_disp + num_nonzero[np.argmin(cost)]
            if (c_proj - c[i]) < max_disp:
                disparity[r[i], c[i]] = abs(c_proj - c[i])
    return disparity


def disparity_to_depth_rectified(disparity, P1):
    depth = P1[0, 3] / disparity
    depth[disparity == 0] = 0.0
    return depth


def get_projector_time_surface(proj_shape):
    idx = 0
    proj_image = np.zeros((proj_shape[1], proj_shape[0]), np.float32)
    for x in range(proj_shape[0]):
        for y in range(proj_shape[1]):
            proj_image[y, x] = idx / (proj_shape[0] * proj_shape[1])
            idx += 1
    return proj_image


def depth_optimization(depth, cam_image, proj_image, window_size, calib):
    depth_optim = np.zeros(depth.shape, np.float32)
    diff_depth = (depth**2) / calib.P1[0, 3]
    for y in range(window_size, depth.shape[0] - window_size):
        for x in range(window_size, depth.shape[1] - window_size):
            if depth[y, x] > 0:
                point = np.array([[x, y]], np.float32)
                res = minimize_scalar(
                    cost_calculator,
                    bounds=(depth[y, x] - diff_depth[y, x], depth[y, x] + diff_depth[y, x]),
                    method="bounded",
                    args=(
                        point,
                        cam_image,
                        proj_image,
                        window_size,
                        calib.cam_int,
                        calib.cam_dist,
                        calib.proj_int,
                        calib.proj_dist,
                        calib.T,
                    ),
                    options={"disp": False},
                )
                depth_optim[y, x] = res.x
    return depth_optim


def compute_all_depths(cam_imgs_rectified, proj_img_rectified, disparity_dir, start_id):
    for i in range(len(cam_imgs_rectified)):
        start = time.time()
        disparity = disparity_init(cam_imgs_rectified[i], proj_img_rectified)
        np.save(os.path.join(disparity_dir, "scan_" + str(start_id + i).zfill(2) + ".npy"), disparity)
        print("Completed scan: " + str(start_id + i) + "in time " + str(time.time() - start))
    return disparity


def main():
    parser = argparse.ArgumentParser(
        description="Depth estimation of event camera and projector system using point scanning projection\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("-object_dir", type=str, default="", help="Directory containing dat files for camera")
    parser.add_argument("-proj_height", type=int, default=1920, help="projector pixel height")
    parser.add_argument("-proj_width", type=int, default=1080, help="projector pixel width")
    parser.add_argument("-calib", type=str, default="", help="camera extrinsics parameter yaml file")
    parser.add_argument("-w", type=int, default=3, help="Window size")
    parser.add_argument("-num_scans", type=int, default=60, help="Number of scans to average over")
    parser.add_argument("-start_scan", type=int, default=0, help="Scan start id")

    args = parser.parse_args()
    proj_shape = (args.proj_width, args.proj_height)
    rect_shape = (int(args.proj_width * 3), int(args.proj_height * 3))

    x_maps_dir = os.path.join(args.object_dir, "x_maps")
    if not os.path.isdir(x_maps_dir):
        os.mkdir(x_maps_dir)

    depth_dir = os.path.join(x_maps_dir, "depth_init")
    if not os.path.isdir(depth_dir):
        os.mkdir(depth_dir)

    pointcloud_dir = os.path.join(x_maps_dir, "pointcloud_init")
    if not os.path.isdir(pointcloud_dir):
        os.mkdir(pointcloud_dir)

    e3d_setup = ut.loadCalibParams(args.calib, (rect_shape[0], rect_shape[1]), alpha=-1)
    cam_image_names = sorted(glob.glob(args.object_dir + "scans_np/*.npy"))
    if len(cam_image_names) == 0:
        print("No camera files found in " + str(args.object_dir + "scans_np/") + "!")
        print(args.object_dir + "scans_np/")
        exit()
    print("Found {0} scans!".format(len(cam_image_names)))
    print()

    # remap rectified view to image plane view
    disp_mapx, disp_mapy = initUndistortRectifyMapInverse(
        e3d_setup.cam_int, e3d_setup.cam_dist, e3d_setup.R0, e3d_setup.P0, (640, 480), None
    )

    # remap image plane to rectified plane
    img_mapx, img_mapy = cv2.initUndistortRectifyMap(
        e3d_setup.cam_int, e3d_setup.cam_dist, e3d_setup.R0, e3d_setup.P0, (rect_shape[0], rect_shape[1]), cv2.CV_32FC1
    )
    proj_mapx, proj_mapy = cv2.initUndistortRectifyMap(
        e3d_setup.proj_int, np.zeros((1, 5)), e3d_setup.R1, e3d_setup.P1, (rect_shape[0], rect_shape[1]), cv2.CV_32FC1
    )

    proj_image = get_projector_time_surface(proj_shape)
    proj_image_rectified = cv2.remap(proj_image, proj_mapx, proj_mapy, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

    calib = XMapsCalib(disp_cam_mapx=disp_mapx, disp_cam_mapy=disp_mapy, Q=e3d_setup.Q)
    proj_time_map = XMapsProjTimeMap(projector_time_map_rectified=proj_image_rectified)
    x_maps_comp = XMapsDisparity(calib=calib, proj_time_map=proj_time_map, proj_width=args.proj_width)
    x_maps_comp.disp_map_shape = (480, 640)

    # x_maps_comp.init_proj_rect_map(proj_image_rectified, proj_shape[0])

    for i in range(args.start_scan, args.start_scan + args.num_scans):
        print("Processing frame: {0}, camera npy file {1}".format(str(i), cam_image_names[i]))
        cam_image = np.load(cam_image_names[i])
        if np.count_nonzero(cam_image) > 0:
            # cam_image = cv2.medianBlur(cam_image, 3)
            cam_image = (cam_image - np.min(cam_image[cam_image != 0])) / (
                np.max(cam_image[cam_image != 0]) - np.min(cam_image[cam_image != 0])
            )

            # cam_image = (cam_image - np.min(cam_image)) / (np.max(cam_image) - np.min(cam_image))
            cam_image[cam_image < 0] = 0

            # cam_image[cam_image == 0] = 1 / cam_image[0, 0]

            event_y = np.argwhere(cam_image > 0)[:, 0]
            event_x = np.argwhere(cam_image > 0)[:, 1]
            event_t = cam_image[cam_image > 0]
            events = {
                "x": event_x,
                "y": event_y,
                "t": event_t,
            }
            start = time.time()
            point_cloud, disparity = x_maps_comp.compute_event_disparity(
                events,
                compute_point_cloud=True,
                compute_disp_map=True,
                projector_view=False,
                rectified_view=False,
            )

            print("Completed disparity estimation: " + str(i) + " in time " + str(time.time() - start))

            depth_init = disparity_to_depth_rectified(disparity, e3d_setup.P1)

            np.save(os.path.join(depth_dir, "scans" + str(i).zfill(3) + ".npy"), depth_init)

            cloud = PyntCloud(
                pd.DataFrame(
                    # same arguments that you are passing to visualize_pcl
                    data=point_cloud,
                    columns=["x", "y", "z"],
                )
            )

            cloud.to_file(os.path.join(pointcloud_dir, "scans" + str(i).zfill(3) + ".ply"))
        else:
            print("Skip camera npy file {} since it is empty".format(cam_image_names[i]))


if __name__ == "__main__":
    main()
