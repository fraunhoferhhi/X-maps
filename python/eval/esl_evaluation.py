# file adapted from https://github.com/uzh-rpg/ESL/blob/734bf8e88f689db79a0b291b1fb30839c6dd4130/python/evaluation.py

import numpy as np
import cv2
import os
from esl_utilities import utils as ut
import pandas as pd
from pyntcloud import PyntCloud

import argparse
import glob

# Visualization parameters
cmap = "jet"


class evaluation_stats:
    def __init__(self, estimate, groundtruth):
        self.groundtruth = groundtruth
        self.estimate = estimate
        self.margin = 0.01 * np.sum(self.groundtruth[self.groundtruth > 0]) / (np.sum(self.groundtruth > 0))
        self.calculate_metrics()

    def calculate_fillrate(self):
        diff = np.abs(self.groundtruth - self.estimate)
        diff[self.groundtruth == 0] = 0
        self.fillrate = (np.sum(diff < self.margin) - np.sum(self.groundtruth == 0)) / (
            diff.shape[0] * diff.shape[1] - np.sum(self.groundtruth == 0)
        )

    def calculate_rmse(self):
        diff_sq = pow((self.groundtruth - self.estimate), 2)
        valid_values = (self.groundtruth > 0) & (self.estimate > 0)
        if np.sum(valid_values) > 0:
            self.rmse = np.sqrt(np.sum(diff_sq[valid_values]) / (np.sum(valid_values)))
        else:
            self.rmse = 0

    def middlebury_metrics(self):
        diff_abs = np.abs(self.groundtruth - self.estimate)
        diff_abs[self.groundtruth == 0] = 0

        self.perc_1 = 100 * np.sum(diff_abs > 1) / (diff_abs.shape[0] * diff_abs.shape[1])
        self.perc_5 = 100 * np.sum(diff_abs > 5) / (diff_abs.shape[0] * diff_abs.shape[1])
        self.perc_10 = 100 * np.sum(diff_abs > 10) / (diff_abs.shape[0] * diff_abs.shape[1])

    def calculate_metrics(self):
        self.calculate_fillrate()
        self.calculate_rmse()
        self.middlebury_metrics()
        return 0

    def print_metrics(self):
        print("Fill rate: " + str(self.fillrate))
        print("RMSE: " + str(self.rmse))
        print("% Pixels with error greater than 1cm: " + str(self.perc_1))
        print("% Pixels with error greater than 5cm: " + str(self.perc_5))
        print("% Pixels with error greater than 10cm: " + str(self.perc_10))


def clip_depth_img(depth_img: np.ndarray, min_value: float, max_value: float):
    """function to clip frame to min and max arguments"""
    # as 0 depth represents no depth value, zeros must be preserved
    # get a boolean map of all zeros
    if min_value > 0:
        bool_map = depth_img == 0
    # clip values
    np.clip(depth_img, a_min=min_value, a_max=max_value, out=depth_img)
    # reset zeros according to boolean map
    if min_value > 0:
        depth_img[bool_map] = 0
    return depth_img


def construct_point_cloud(xpr_f32, ypr_f32, disp_f32, Q):
    points = np.ones((len(xpr_f32), 4), dtype=np.float32)
    # points = np.stack((xpr_dispf, ypr_dispf, 1)).T
    points[:, 0] = xpr_f32
    points[:, 1] = ypr_f32
    points[:, 2] = -disp_f32
    point_cloud = (Q.astype(np.float32) @ points.T).T
    point_cloud = (point_cloud / point_cloud[:, 3:])[:, :3]

    # invert y and z axis
    point_cloud[:, 1] = -point_cloud[:, 1]
    point_cloud[:, 2] = -point_cloud[:, 2]

    return point_cloud


def get_pointcloud(frame, e3d_setup, min_depth, max_depth):
    frame[frame < min_depth] = 0
    frame[frame > max_depth] = 0

    event_y = np.argwhere(frame > 0)[:, 0]
    event_x = np.argwhere(frame > 0)[:, 1]
    event_depth = frame[frame > 0]

    coords = np.stack((event_x, event_y)).T.astype(np.float32)

    xypr_f32_filt = cv2.undistortPoints(coords, e3d_setup.cam_int, e3d_setup.cam_dist, R=e3d_setup.R0, P=e3d_setup.P0)

    xpr_f32 = xypr_f32_filt[:, 0, 0]
    ypr_f32 = xypr_f32_filt[:, 0, 1]
    disp = e3d_setup.P1[0, 3] / event_depth

    point_cloud = construct_point_cloud(xpr_f32, ypr_f32, disp, e3d_setup.Q)
    # still some t offset
    # point_cloud += e3d_setup.T[:3, 3]
    # print(e3d_setup.T[:3, 3])
    cloud = PyntCloud(
        pd.DataFrame(
            # same arguments that you are passing to visualize_pcl
            data=point_cloud,
            columns=["x", "y", "z"],
        )
    )
    return cloud


def load_and_filter(filename, gt, min_depth, max_depth):
    result = np.load(filename)
    result[result >= max_depth] = 0
    result[result <= min_depth] = 0
    result[gt == 0] = 0
    return result


def round_rmse(val):
    return str(round(val, 2))


def round_fr(val):
    return str(round(val, 2))


def print_result(method, res):
    print("{} & {} &".format(round_fr(res[0]), round_rmse(res[1])))


def print_tableLine(method, results):
    print("{}".format(method), end="")
    for res in results:
        print(" & {} & {} ".format(round_fr(res[0]), round_rmse(res[1])), end="")
    print("\\\\")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation of event camera and projector system using line scanning projection\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("-object_dir", type=str, default="", help="Directory containing depth files")
    parser.add_argument("-proj_height", type=int, default=1920, help="projector pixel height")
    parser.add_argument("-proj_width", type=int, default=1080, help="projector pixel width")
    parser.add_argument("-calib", type=str, default="", help="camera extrinsics parameter yaml file")
    parser.add_argument("-out_dir", type=str, default="", help="output dir")
    parser.add_argument("-max_depth", type=int, default="120", help="output dir")
    parser.add_argument("-min_depth", type=int, default="20", help="output dir")

    args = parser.parse_args()

    scenes = [
        "seq9",  # Dacid
        "seq8",  # Heart
        "seq1",  # Book-Duck
        "seq2",  # Plant
        "seq3",  # City of Lights
        "seq7",  # Cycle
        "seq6",  # Room
        "seq5",  # Desk-chair
        "seq4",  # Desk books
    ]

    all_mc3d = []
    all_mc3d_1s = []
    all_esl = []
    all_x_maps = []

    min_depth = args.min_depth
    max_depth = args.max_depth
    print("Max depth {}".format(max_depth))
    print("Mean depth ", end="")
    for seq_name in scenes:
        seq_dir = os.path.join(args.object_dir, seq_name)
        gt_files = sorted(glob.glob(os.path.join(seq_dir, "esl/depth_optim_filtered/*.npy")))
        esl_files = sorted(glob.glob(os.path.join(seq_dir, "esl/depth_init/*.npy")))
        x_maps_files = sorted(glob.glob(os.path.join(seq_dir, "x_maps/depth_init/*.npy")))
        mc3d_files = sorted(glob.glob(os.path.join(seq_dir, "mc3d/depth/*.npy")))

        mc3d_combined, thresh, avg_depth = ut.combine_mc3d(mc3d_files, len(mc3d_files), min_depth, max_depth)
        gt_combined, thresh, avg_depth = ut.combine_mc3d(gt_files, len(gt_files), min_depth, max_depth)

        print(" & \\multicolumn{{2}}{{c}}{{{}}}".format(round(avg_depth, 1)), end="")

        if len(gt_files) != len(x_maps_files) or len(gt_files) != len(mc3d_files):
            print("Something is wrong with the data. Frames are missing")
            print(f"gt_files: {len(gt_files)}\t x_maps_files: {len(x_maps_files)}\t mc3d_files: {len(mc3d_files)}")
            exit()

        # if args.calib is not "" and args.out_dir is not "":
        #     if not os.path.isdir(args.out_dir):
        #         os.mkdir(args.out_dir)
        #     proj_shape = (args.proj_width, args.proj_height)
        #     rect_shape = (int(args.proj_width * 3), int(args.proj_height * 3))
        #     e3d_setup = ut.loadCalibParams(args.calib, (rect_shape[0], rect_shape[1]), alpha=-1)

        #     esl = np.load(gt_files[0])

        #     cloud_test = get_pointcloud(esl, e3d_setup, min_depth, max_depth)
        #     cloud_gt = get_pointcloud(gt_combined, e3d_setup, min_depth, max_depth)

        #     cloud_test.to_file(os.path.join(args.out_dir, "eval_poincloud.ply"))
        #     cloud_gt.to_file(os.path.join(args.out_dir, "gt_poincloud.ply"))

        # plt.imshow(gt_combined, "jet")
        # plt.show()

        avg_mc3d = []
        avg_mc3d_1s = []
        avg_esl = []
        avg_x_maps = []

        for idx in range(len(gt_files)):
            esl_gt_file = load_and_filter(gt_files[idx], gt_combined, min_depth, max_depth)
            esl_init_file = load_and_filter(esl_files[idx], gt_combined, min_depth, max_depth)
            mc3d_file = load_and_filter(mc3d_files[idx], gt_combined, min_depth, max_depth)
            x_maps_file = load_and_filter(x_maps_files[idx], gt_combined, min_depth, max_depth)
            stats_mc3d = evaluation_stats(mc3d_file, esl_gt_file)
            avg_mc3d.append([stats_mc3d.fillrate, stats_mc3d.rmse])

            stats_mc3d_1_sec = evaluation_stats(mc3d_combined, esl_gt_file)
            avg_mc3d_1s.append([stats_mc3d_1_sec.fillrate, stats_mc3d_1_sec.rmse])

            stats_esl = evaluation_stats(esl_init_file, esl_gt_file)
            avg_esl.append([stats_esl.fillrate, stats_esl.rmse])

            stats_x_maps = evaluation_stats(x_maps_file, esl_gt_file)
            avg_x_maps.append([stats_x_maps.fillrate, stats_x_maps.rmse])

        avg_mc3d = np.mean(np.array(avg_mc3d), axis=0)
        avg_mc3d_1s = np.mean(np.array(avg_mc3d_1s), axis=0)
        avg_esl = np.mean(np.array(avg_esl), axis=0)
        avg_x_maps = np.mean(np.array(avg_x_maps), axis=0)

        all_mc3d.append(avg_mc3d)
        all_mc3d_1s.append(avg_mc3d_1s)
        all_esl.append(avg_esl)
        all_x_maps.append(avg_x_maps)

        # print("Result (MC3D, MC3D (1 sec), ESL (init), X-Maps (ours)")
        # print_result("MC3D", avg_mc3d)
        # print_result("MC3D (1 sec)", avg_mc3d_1s)
        # print_result("ESL (init)", avg_esl)
        # print_result("X-Maps (ours)", avg_x_maps)

    print("")
    print_tableLine("MC3D", all_mc3d)
    print_tableLine("MC3D (1 sec)", all_mc3d_1s)
    print_tableLine("ESL (init)", all_esl)
    print_tableLine("X-Maps (ours)", all_x_maps)


if __name__ == "__main__":
    main()
