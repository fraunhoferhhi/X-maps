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
    proj_mapx, proj_mapy = cv2.initUndistortRectifyMap(
        e3d_setup.proj_int, np.zeros((1, 5)), e3d_setup.R1, e3d_setup.P1, (rect_shape[0], rect_shape[1]), cv2.CV_32FC1
    )

    proj_image = get_projector_time_surface(proj_shape)
    proj_image_rectified = cv2.remap(proj_image, proj_mapx, proj_mapy, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

    calib = XMapsCalib(disp_cam_mapx=disp_mapx, disp_cam_mapy=disp_mapy, Q=e3d_setup.Q)
    proj_time_map = XMapsProjTimeMap(projector_time_map_rectified=proj_image_rectified)
    x_maps_comp = XMapsDisparity(calib=calib, proj_time_map=proj_time_map, proj_width=args.proj_width)
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
