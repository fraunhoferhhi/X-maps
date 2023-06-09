# Implementation of the baseline MC3D
# from https://github.com/uzh-rpg/ESL/blob/734bf8e88f689db79a0b291b1fb30839c6dd4130/python/mc3d_baseline.py

import os
import time
import glob
import os.path
import argparse
import numpy as np
import cv2

from esl_utilities import utils as ut


def disparity_to_depth(disparity, P1):
    depth = P1 / disparity
    depth[disparity == 0] = 0
    return depth


def initUndistortRectifyMapInverse(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type):
    W, H = size
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).T.reshape((-1, 1, 2)).astype("float32")
    points = cv2.undistortPoints(coords, cameraMatrix, distCoeffs, None, R, newCameraMatrix)
    maps = points.reshape((H, W, 2))
    return maps[..., 0], maps[..., 1]


def remap_events(xy, mapx, mapy):
    # cam0 is the projector
    x, y = xy
    try:
        x_undist = int(mapx[y, x])
        y_undist = int(mapy[y, x])
        return [x_undist, y_undist]
    except:
        return None


def compute_disparity(cam_image, img_mapx, img_mapy, proj_mapx, proj_mapy, proj_shape, rectified_shape):
    proj_img = np.zeros((proj_shape[1], proj_shape[0]), np.float32)
    disparity = np.zeros(cam_image.shape, np.float32)
    nc = int(
        proj_shape[1] / 15
    )  # noise accounting for 15us between start and end of a line to improve the mc3d depth map
    for i in range(0, cam_image.shape[0]):
        for j in range(0, cam_image.shape[1]):
            if cam_image[i, j] > 0:
                remapped_event_c = remap_events([j, i], img_mapx, img_mapy)
                try:
                    [xc_undist, yc_undist] = remapped_event_c
                    if (
                        xc_undist > 0
                        and xc_undist < rectified_shape[1]
                        and yc_undist > 0
                        and yc_undist < rectified_shape[0]
                    ):
                        proj_id = int(proj_shape[0] * proj_shape[1] * cam_image[i, j])
                        proj_x, proj_y = np.unravel_index(proj_id, (proj_shape[0], proj_shape[1]))
                        diff_y = []
                        disp = []
                        proj_px = []
                        y_true = []
                        for y in range(max(proj_y - nc, 0), min(proj_y + nc, proj_shape[1])):
                            remapped_event_ps = remap_events([proj_x, y], proj_mapx, proj_mapy)
                            [xp_undist, yp_undist] = remapped_event_ps
                            diff_y.append(abs(yc_undist - yp_undist))
                            y_true.append(y)
                            disp.append(xp_undist - xc_undist)
                            proj_px.append([xp_undist, yp_undist])
                        if len(diff_y) > 0 and np.min(diff_y) <= 50:
                            idx = np.argmin(diff_y)
                            if disp[idx] > 0:
                                disparity[i, j] = disp[idx]
                                proj_img[y_true[idx], proj_x] = cam_image[i, j]
                except:
                    pass
    return disparity


def main():
    parser = argparse.ArgumentParser(
        description="Depth estimation of cam-pro system using MC3D baseline\n"
        "        |      .       |        .\n"
        "        |      .       |        .\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("-object_dir", type=str, default="", help="Directory containing dat files for camera")
    parser.add_argument("-proj_height", type=int, default=1920, help="projector pixel height")
    parser.add_argument("-proj_width", type=int, default=1080, help="projector pixel width")
    parser.add_argument("-calib", type=str, default="", help="camera extrinsics parameter yaml file")
    parser.add_argument("-cal_disparity", type=int, default=1, help="Calculate disparity or use a saved disparity")
    parser.add_argument("-num_scans", type=int, default=1, help="Number of scans to average over")
    parser.add_argument("-start_scan", type=int, default=0, help="Scan start id")

    args = parser.parse_args()
    print()
    proj_shape = (args.proj_width, args.proj_height)
    rectified_shape = (args.proj_width * 3, args.proj_height * 3)
    e3d_setup = ut.loadCalibParams(args.calib, proj_shape, alpha=-1)
    mc3d_dir = os.path.join(args.object_dir, "mc3d")
    if not os.path.isdir(mc3d_dir):
        os.mkdir(mc3d_dir)

    depth_dir = os.path.join(mc3d_dir, "depth")
    if not os.path.isdir(depth_dir):
        os.mkdir(depth_dir)
    proj_mapx, proj_mapy = initUndistortRectifyMapInverse(
        e3d_setup.proj_int, e3d_setup.proj_dist, e3d_setup.R1, e3d_setup.P1, proj_shape, None
    )
    img_mapx, img_mapy = initUndistortRectifyMapInverse(
        e3d_setup.cam_int, e3d_setup.cam_dist, e3d_setup.R0, e3d_setup.P0, proj_shape, None
    )

    print("Searching camera npy files ...")
    print(args.object_dir)
    cam_image_names = sorted(glob.glob(args.object_dir + "scans_np/*.npy"))
    if len(cam_image_names) == 0:
        print("No camera files found!")
        exit()
    print("Found {0} scans!".format(len(cam_image_names)))

    disparities = []
    for k in range(args.start_scan, args.start_scan + args.num_scans):
        if args.cal_disparity > 0:
            print(cam_image_names[k])
            cam_image = np.load(cam_image_names[k])
            if np.count_nonzero(cam_image) > 0:
                cam_image = cv2.medianBlur(cam_image, 3)

                start = time.time()
                print("Computing depth")
                disparity = compute_disparity(
                    cam_image, img_mapx, img_mapy, proj_mapx, proj_mapy, proj_shape, rectified_shape
                )
                print("Completed frame " + str(k) + "in time " + str(time.time() - start))

                depth = disparity_to_depth(disparity, e3d_setup.P1[0, 3])
                np.save(os.path.join(depth_dir, "scans" + str(k).zfill(3) + ".npy"), depth)
            else:
                print("Skip {}".format(k))

            # plt.imshow(depth, "jet")
            # plt.title("depth")
            # plt.colorbar()
            # plt.show()


if __name__ == "__main__":
    main()
