import numba
import numpy as np

from epipolar_disparity import (
    rectify_cam_coords,
    compute_disparity,
    construct_point_cloud,
)


def dump_frame_data(events, inlier_mask, xcr_f32, ycr_f32, disp_f32, csv_name="/ESL_data/static/seq1/frame.csv"):
    import pandas as pd

    df = pd.DataFrame(
        [
            events["x"][inlier_mask].T,
            events["y"][inlier_mask].T,
            events["t"][inlier_mask].T,
            xcr_f32[inlier_mask].T,
            ycr_f32[inlier_mask].T,
            disp_f32.T,
        ],
    ).T
    df.columns = ["x", "y", "t", "x_r", "y_r", "disp"]

    df.to_csv(csv_name, index=False)


def compute_disparity(xcr_f32, ycr_f32, t, proj_x_map, T_PX_SCALE, X_OFFSET):
    # TODO perf are xcr_f32 and ycr_f32 ever used as floats?
    # otherwise directly look up the int16's

    # rectified rounded event coordinates
    xcr_i16 = np.rint(xcr_f32).astype(np.int16)
    ycr_i16 = np.rint(ycr_f32).astype(np.int16)

    # events may not be completely sorted by time if
    # they were processed in a filter that averages over coordinates
    min_t = t.min()
    max_t = t.max()

    # event time normalized to [0, 1]
    event_norm_t = (t - min_t) / (max_t - min_t)

    # event time in the scale of the X-map
    t_scaled = np.rint(event_norm_t * T_PX_SCALE).astype(np.int16)

    y_inlier_mask = (ycr_f32 >= 0) & (ycr_f32 < proj_x_map.shape[0] - 1)

    # TODO use cv2.remap to retrieve with interpolation from proj_x_map
    # TODO subpixel + 0.5
    x_proj = proj_x_map[ycr_i16[y_inlier_mask], t_scaled[y_inlier_mask]]

    # TODO check x_proj lies within defined pixels in proj_x_map

    disp = x_proj - xcr_i16[y_inlier_mask] - X_OFFSET

    disp_inlier_mask = disp >= 0
    y_inlier_mask[y_inlier_mask] = disp_inlier_mask

    # # PAPER VIS
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd

    # sns.set_theme()
    # # plt.scatter(x=t, y=ycr_f32, c=xcr_f32, cmap="plasma")
    # df = pd.DataFrame({"t": t_scaled[y_inlier_mask], "y": ycr_f32[y_inlier_mask], "x": xcr_f32[y_inlier_mask]})
    # df.plot.scatter(x="t", y="y", c="x", cmap="plasma")
    # plt.gca().invert_yaxis()
    # plt.show()

    return disp[disp_inlier_mask], y_inlier_mask


@numba.njit
def optimize_proj_x_map(proj_time_map, T_MAP_SIZE, T_PX_SCALE, X_OFFSET, proj_width):
    # TODO perf precompute

    proj_x_map = np.zeros((proj_time_map.shape[0], T_MAP_SIZE), dtype=np.int16)

    # don't allow more than two rows difference
    max_t_diff = 2 / proj_width
    # max_t_diff = np.inf

    # proj_x_map_def = np.zeros((proj_time_map.shape[0], T_MAP_SIZE), dtype=np.uint8)
    t_diffs = np.zeros((proj_time_map.shape[0], T_MAP_SIZE), dtype=np.float32)

    for y in range(proj_x_map.shape[0]):
        for t_coord in range(proj_x_map.shape[1]):
            # compute optimal x for each t

            t = t_coord / T_PX_SCALE

            # TODO 0-value is not defined - but also is the first pixel -- proj map has no offset
            if t == 0:
                continue

            min_t_diff = np.inf
            min_t_diff_x = -1

            for x in range(proj_time_map.shape[1]):
                t_map = proj_time_map[y, x]
                if t_map == 0:
                    continue

                t_diff = np.abs(t - t_map)
                if t_diff < min_t_diff:
                    min_t_diff = t_diff
                    min_t_diff_x = x

            if min_t_diff_x != -1:
                if min_t_diff <= max_t_diff:
                    proj_x_map[y, t_coord] = min_t_diff_x + X_OFFSET
                    # proj_x_map_def[y, t_coord] = True
                    t_diffs[y, t_coord] = min_t_diff

    # return proj_x_map, proj_x_map_def
    return proj_x_map, t_diffs
    # return proj_x_map


class XMapsDisparity:
    def __init__(self, calib, proj_time_map, proj_width):
        self.calib = calib
        self.init_proj_x_map(proj_time_map.projector_time_map_rectified, proj_width)
        self.disp_map_shape = proj_time_map.projector_time_map_rectified.shape

    def init_proj_x_map(self, proj_time_map, proj_width):
        # to mask where x = 0
        # TODO reduce to 1 if it works
        self.X_OFFSET = 4242

        # use 16 bit for indices
        assert proj_time_map.shape[0] <= 2**15 - 1
        assert proj_time_map.shape[1] + self.X_OFFSET <= 2**15 - 1

        # T_PX_SCALE = proj_time_map.shape[1] - 1
        self.T_MAP_SIZE = 1080
        self.T_PX_SCALE = self.T_MAP_SIZE - 1

        # xy_xsf, xy_ysf = init_direct_disparity(proj_time_map.shape)

        self.proj_x_map, t_diffs = optimize_proj_x_map(
            proj_time_map, self.T_MAP_SIZE, self.T_PX_SCALE, self.X_OFFSET, proj_width
        )

    def compute_event_disparity(
        self,
        events,
        proj_window_name=None,
        compute_point_cloud=False,
        compute_disp_map=True,
        projector_view=True,
        rectified_view=True,
        PRECOMP_RECT=False,
    ):
        # events = mean_first_last_event_per_xy(events)
        # events = first_event_per_yt(events)
        # events = first_event_per_yt(events)
        # events = events[events["p"] == 1]

        # for each event
        # get rectified coordinates
        xcr_f32, ycr_f32 = rectify_cam_coords(self.calib.disp_cam_mapx, self.calib.disp_cam_mapy, events)

        # at time t and rectified y, access yt map
        disp_f32, inlier_mask = compute_disparity(
            xcr_f32, ycr_f32, events["t"], self.proj_x_map, self.T_PX_SCALE, self.X_OFFSET
        )

        # if proj_window_name is not None:
        #     pm_display = self.proj_x_map.copy()
        #     pm_display -= self.X_OFFSET
        #     pm_display[self.proj_x_map == 0] = 0
        #     cv2.imshow(proj_window_name, utils.img_to_viridis(pm_display))
        # cv2.imshow(cam_window_name, utils.img_to_viridis(show_cam_map))

        point_cloud, disp_map = None, None

        if compute_point_cloud:
            point_cloud = construct_point_cloud(
                xcr_f32[inlier_mask] + disp_f32, ycr_f32[inlier_mask], disp_f32, self.calib.Q
            )

        if compute_disp_map:
            disp_map = np.zeros(self.disp_map_shape, dtype=np.float32)

            # ypr_dispf = np.rint(ypr_f32[inlier_mask]).astype(np.int16)
            # xpr_dispf = np.rint(xpr_f32[inlier_mask]).astype(np.int16)

            # TODO: choice between ypr and ycr
            # disp_map[ypr_dispf, xpr_dispf] = disp[inlier_mask]

            # TODO + 0.5?
            if rectified_view:
                ycr_i16 = np.rint(ycr_f32[inlier_mask]).astype(np.int16)

            # TODO should one of the disparities actually be negative, assuption: no, since then the baseline also must be negative and it would cancel out ? (N.G.)
            # TODO instead of using the rounded rectified coordinates, be could also use the input event coordinates
            if projector_view:
                if not rectified_view:
                    print("Projector view not implemented. Return rectified view")
                    return point_cloud, None

                xpr_i16 = np.rint(xcr_f32[inlier_mask] + disp_f32).astype(np.int16)
                disp_map[ycr_i16, xpr_i16] = disp_f32
            else:
                if rectified_view:
                    xcr_i16 = np.rint(xcr_f32[inlier_mask]).astype(np.int16)
                    disp_map[ycr_i16, xcr_i16] = disp_f32
                else:
                    x_cam = events["x"][inlier_mask]
                    y_cam = events["y"][inlier_mask]
                    disp_map[y_cam, x_cam] = disp_f32

        # dump_frame_data(
        #     events,
        #     inlier_mask,
        #     xcr_f32,
        #     ycr_f32,
        #     disp_f32,
        # )

        return point_cloud, disp_map
