# file from https://github.com/uzh-rpg/ESL/blob/734bf8e88f689db79a0b291b1fb30839c6dd4130/python/utils/utilities.py

import yaml
import numpy as np
import cv2
from matplotlib import pyplot as plt

try:
    import pylops
except ImportError:
    pylops = None


class calib:
    def __init__(self, cam, cam_dist, proj, proj_dist, R, R0, R1, P0, P1, Q, T):
        self.cam_int = cam
        self.cam_dist = cam_dist
        self.proj_int = proj
        self.proj_dist = proj_dist
        self.R = R
        self.R0 = R0
        self.R1 = R1
        self.P0 = P0
        self.P1 = P1
        self.Q = Q
        self.T = T


class utils:
    def __init__(self):
        pass

    @staticmethod
    def writePly(fn, verts, colors):
        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        ply_header = """ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """
        with open(fn, "wb") as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode("utf-8"))
            np.savetxt(f, verts, fmt="%f %f %f %d %d %d ")
        print("Finished writing pointcloud to: {0} ".format(fn))

    @staticmethod
    def loadStereoCameraParamKalibr(yaml_file):
        print("Reading calibration from file: {0}".format(yaml_file))
        with open(yaml_file, "r") as f:
            param_data = yaml.load(f)
            i = param_data["cam1"]["intrinsics"]
            d0 = param_data["cam1"]["distortion_coeffs"]
            P0 = np.zeros((3, 3), np.float32)
            P0[0, 0] = i[0]
            P0[1, 1] = i[1]
            P0[0, 2] = i[2]
            P0[1, 2] = i[3]

            i = param_data["cam0"]["intrinsics"]
            d1 = param_data["cam0"]["distortion_coeffs"]
            P1 = np.zeros((3, 3), np.float32)
            P1[0, 0] = i[0]
            P1[1, 1] = i[1]
            P1[0, 2] = i[2]
            P1[1, 2] = i[3]

            i = param_data["cam1"]["T_cn_cnm1"]
            T_1_0 = np.zeros((4, 4))  # T cam right (0) to left (1) i.e VGA to 3m1.1 i.e 0333 to 1126
            for t in range(4):
                T_1_0[t,] = i[t]
            # print(T_1_0)
            return P0, np.array(d0), P1, np.array(d1), T_1_0

    @staticmethod
    def rectifyImage(cam0, dist0, cam1, dist1, cam_0_1_R, cam_0_1_t, cam0_image=None, cam1_image=None):
        rectify_w = 960
        rectify_h = 640
        R1, R0, P1, P0, Q, _, _ = cv2.stereoRectify(
            cam1,
            dist1,
            cam0,
            dist0,
            (rectify_w, rectify_h),
            cam_0_1_R,
            cam_0_1_t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=1,
        )
        # cv2.imshow("cam0_image", cam0_image)
        # cv2.imshow("cam1_image", cam1_image)
        # cv2.waitKey(0)
        cam0MapX, cam0MapY = cv2.initUndistortRectifyMap(cam0, dist0, R0, P0, (rectify_w, rectify_h), cv2.CV_32FC1)
        cam1MapX, cam1MapY = cv2.initUndistortRectifyMap(cam1, dist1, R1, P1, (rectify_w, rectify_h), cv2.CV_32FC1)
        if cam0_image is not None:
            cam0_image_rectified = cv2.remap(cam0_image, cam0MapX, cam0MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            cam1_image_rectified = cv2.remap(cam1_image, cam1MapX, cam1MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        else:
            cam0_image_rectified = None
            cam1_image_rectified = None

        # cv2.line(cam0_image_rectified, (0, 450), (cam1_image.shape[1], 450), (255, 255, 255))
        # cv2.line(cam1_image_rectified, (0, 450), (cam1_image.shape[1], 450), (255, 255, 255))

        # cv2.line(cam0_image_rectified, (0, 600), (cam1_image.shape[1], 600), (255, 255, 255))
        # cv2.line(cam1_image_rectified, (0, 600), (cam1_image.shape[1], 600), (255, 255, 255))

        # cv2.line(cam0_image_rectified, (0, 750), (cam1_image.shape[1], 750), (255, 255, 255))
        # cv2.line(cam1_image_rectified, (0, 750), (cam1_image.shape[1], 750), (255, 255, 255))

        # cv2.imshow("cam0_image_rectified", cam0_image_rectified)
        # cv2.imshow("cam1_image_rectified", cam1_image_rectified)
        # cv2.waitKey(0)
        return R0, R1, P0, P1, cam0_image_rectified, cam1_image_rectified, Q

    @staticmethod
    def loadCalibParams(calib_file, shape, alpha):
        print("Reading calibration from file: {0}".format(calib_file))
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        cam_int = fs.getNode("cam_K").mat()
        cam_dist = fs.getNode("cam_kc").mat()
        proj_int = fs.getNode("proj_K").mat()
        proj_dist = fs.getNode("proj_kc").mat()
        cam_proj_rmat = fs.getNode("R").mat()
        cam_proj_tvec = fs.getNode("T").mat()
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            proj_int,
            proj_dist,
            cam_int,
            cam_dist,
            shape,
            cam_proj_rmat,
            cam_proj_tvec,
            # flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=alpha,
        )
        T = np.zeros((4, 4), np.float32)
        T[:3, :3] = cam_proj_rmat
        T[:3, 3] = cam_proj_tvec[:, 0]
        # T_inv = np.zeros((4, 4), np.float32)
        # T_inv[:3, :3] = cam_proj_rmat.T
        # T_inv[:3, 3] = -np.dot(cam_proj_rmat.T, cam_proj_tvec[:, 0])
        return calib(cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, R1, R2, P1, P2, Q, T)

    @staticmethod
    def combine_mc3d(depth_files, num_scans, min_d, max_d):
        cmap = "jet"
        depth_combined = np.zeros((480, 640), np.float32)
        count = np.zeros((480, 640), np.float32)
        depths_combined = []
        for i in range(0, num_scans):
            try:
                d = np.load(depth_files[i]).astype(np.float32)
                d[d >= max_d] = 0
                d[d <= min_d] = 0
                depth_combined += d
                count += d > 0
                d[d >= max_d] = 0
                d[d <= min_d] = 0
            except:
                pass

        depth_combined /= count
        depth_combined[count == 0] = 0
        depth_combined = cv2.medianBlur(depth_combined, 3)
        avg_depth = np.sum(depth_combined[depth_combined > 0]) / (np.sum(depth_combined > 0))
        thresh = 0.01 * (avg_depth)
        return depth_combined, thresh, avg_depth

    @staticmethod
    def disparityToPointcloud(disparity, Q, v_min=None, v_max=None):
        disparity[disparity > v_max] = 0
        disparity[disparity < v_min] = 0
        points = cv2.reprojectImageTo3D(disparity, Q)
        if v_max is None:
            v_max = np.max(disparity)
        if v_min is None:
            v_min = np.min(disparity)
        norm = plt.Normalize(vmin=v_min, vmax=v_max)
        image = plt.cm.jet(norm(disparity))
        image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        out_colors = np.array(image[:, :, :3])
        out_colors = out_colors[np.isfinite(points)]
        points = points[np.isfinite(points)]
        return points, out_colors

    @staticmethod
    def denoise_tv(y, mu=0.3):
        if pylops is None:
            raise ImportError("The denoise_tv function requires the 'pylops' package")

        nx, ny = y.shape
        Iop = pylops.Identity(nx * ny)
        # Dop = pylops.FirstDerivative(nx, edge=True, kind='backward')
        Dop = [
            pylops.FirstDerivative(ny * nx, dims=(ny, nx), dir=0, edge=False, kind="backward", dtype=np.float32),
            pylops.FirstDerivative(ny * nx, dims=(ny, nx), dir=1, edge=False, kind="backward", dtype=np.float32),
        ]
        # mu = mu
        lamda = [0.1, 0.1]
        niter = 20
        niterinner = 10

        xinv, niter = pylops.optimization.sparsity.SplitBregman(
            Iop,
            Dop,
            y.flatten(),
            niter,
            niterinner,
            mu=mu,
            epsRL1s=lamda,
            tol=1e-4,
            tau=1.0,
            show=False,
            **dict(iter_lim=5, damp=1e-4)
        )
        return xinv.reshape(y.shape)
