# coding: UTF-8
import argparse
import glob
import os.path
import sys
import time

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

from esl_utilities import utils as ut


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from x_maps_disparity import XMapsDisparity
from cam_proj_calibration import CamProjCalibrationParams, CamProjMaps
from proj_time_map import ProjectorTimeMap


@dataclass
class XMapsCalib:
    disp_cam_mapx: np.ndarray
    disp_cam_mapy: np.ndarray
    Q: np.ndarray
    projector_width: int
    projector_height: int
    projector_mapx: np.ndarray
    projector_mapy: np.ndarray


def initUndistortRectifyMapInverse(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type):
    W, H = size
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).T.reshape((-1, 1, 2)).astype("float32")
    points = cv2.undistortPoints(coords, cameraMatrix, distCoeffs, None, R, newCameraMatrix)
    maps = points.reshape((H, W, 2))
    return maps[..., 0], maps[..., 1]


def disparity_to_depth_rectified(disparity, P1):
    depth = P1[0, 3] / disparity
    depth[disparity == 0] = 0.0
    return depth


def main():
    parser = argparse.ArgumentParser(
        description="Depth estimation of event camera and projector system using point scanning projection\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("-object_dir", type=str, default="", help="Directory containing dat files for camera")
    parser.add_argument("-proj_height", type=int, default=1920, help="projector pixel height")
    parser.add_argument("-proj_width", type=int, default=1080, help="projector pixel width")
    parser.add_argument("-calib", type=str, default="", help="camera extrinsics parameter yaml file")
    parser.add_argument("-num_scans", type=int, default=60, help="Number of scans to average over")
    parser.add_argument("-start_scan", type=int, default=0, help="Scan start id")

    args = parser.parse_args()
    proj_shape = (args.proj_width, args.proj_height)
    rect_shape = (int(args.proj_width * 3), int(args.proj_height * 3))

    x_maps_dir = os.path.join(args.object_dir, "x_maps")
    os.makedirs(x_maps_dir, exist_ok=True)

    depth_dir = os.path.join(x_maps_dir, "depth_init")
    os.makedirs(depth_dir, exist_ok=True)

    pointcloud_dir = os.path.join(x_maps_dir, "pointcloud_init")
    os.makedirs(pointcloud_dir, exist_ok=True)

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
    proj_mapx, proj_mapy = cv2.initUndistortRectifyMap(
        e3d_setup.proj_int, np.zeros((1, 5)), e3d_setup.R1, e3d_setup.P1, (rect_shape[0], rect_shape[1]), cv2.CV_32FC1
    )

    calib__ = XMapsCalib(
        disp_cam_mapx=disp_mapx,
        disp_cam_mapy=disp_mapy,
        Q=e3d_setup.Q,
        projector_width=args.proj_width,
        projector_height=args.proj_height,
        projector_mapx=proj_mapx,
        projector_mapy=proj_mapy,
    )

    cam_width, cam_height = 640, 480

    calib_params = CamProjCalibrationParams.from_ESL_yaml(
        args.calib, cam_width, cam_height, args.proj_width, args.proj_height
    )

    cam_proj_maps = CamProjMaps(calib_params)

    # by default X-Maps processing we would use scan_upwards=True (as we expect the projector to scan from bottom to top) and BORDER_REPLICATE
    # to compare against ESL results, we use their settings here instead of the generated time map
    proj_time_map = ProjectorTimeMap.from_calib(
        calib_params=calib_params,
        cam_proj_maps=cam_proj_maps,
        scan_upwards=False,
        remap_border_mode=cv2.BORDER_CONSTANT,
    )

    x_maps_comp = XMapsDisparity(
        calib_params=calib_params, calib_maps=cam_proj_maps, proj_time_map=proj_time_map, proj_width=args.proj_width
    )
    x_maps_comp.disp_map_shape = (480, 640)

    for i in range(args.start_scan, args.start_scan + args.num_scans):
        print("Processing frame: {0}, camera npy file {1}".format(str(i), cam_image_names[i]))
        cam_image = np.load(cam_image_names[i])
        if np.count_nonzero(cam_image) > 0:
            cam_image = (cam_image - np.min(cam_image[cam_image != 0])) / (
                np.max(cam_image[cam_image != 0]) - np.min(cam_image[cam_image != 0])
            )

            cam_image[cam_image < 0] = 0

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
                    data=point_cloud,
                    columns=["x", "y", "z"],
                )
            )

            cloud.to_file(os.path.join(pointcloud_dir, "scans" + str(i).zfill(3) + ".ply"))
        else:
            print("Skip camera npy file {} since it is empty".format(cam_image_names[i]))


if __name__ == "__main__":
    main()
