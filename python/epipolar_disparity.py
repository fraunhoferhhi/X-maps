from funcy import print_durations

import numpy as np
import cv2

from numba import njit

# TODO don't round yp?

# TODO distinguish between discrete and continuous values
#      just because it's a float doesn't mean it's continuous

PRINT_DURATIONS = False


def benchmark(func):
    if not PRINT_DURATIONS:
        return func
    return print_durations(func)


@benchmark
@njit
def compute_xp(events, proj_width):
    # TODO use actual frame length instead of max - min event time?
    event_norm_t = (events["t"] - events["t"].min()) / (events["t"].max() - events["t"].min())

    # projector x from event time, rounded to nearest int
    xp_i16 = np.rint(event_norm_t * (proj_width - 1)).astype(np.int16)

    # TODO unused
    # TODO make f32
    xp_f64 = event_norm_t * (proj_width - 1)

    # TODO check rounding down instead?
    # xp = (event_norm_t * proj_width - 0.5).astype(np.int16)

    return xp_i16


@njit
def intersect(a, b, c, xp_i16):
    # intersect camera epipolar line with projector vertical line
    # TODO actually the line is not vertical
    yp_from_intersection_f32 = (-c - a * xp_i16) / b
    # yp = (-c - a * xp_noround) / b

    # yp_i16 = np.rint(yp_from_intersection_f32).astype(np.int16)

    return yp_from_intersection_f32


@benchmark
def intersect_cam_epipolar_line_with_proj_x(events, fund_mat, disp_cam_mapx_undist, disp_cam_mapy_undist, xp_i16):
    # TODO perf: remove np.stack
    # TODO query with +0.5 pixel centers as float32

    xpr_f32 = disp_cam_mapx_undist[events["y"], events["x"]]
    ypr_f32 = disp_cam_mapy_undist[events["y"], events["x"]]

    query_points_f32 = np.stack((xpr_f32, ypr_f32)).T.astype(np.float32)

    # query_points_f = query_points_i32.astype(np.float32) + 0.5

    # cv2.computeCorrespondEpilines(np.array([xcr, ycr]).T, 1, fund_mat)
    # TODO perf query points are always from the same coordinates
    # -> could precompute

    # epipolar lines of the camera points
    a, b, c = np.squeeze(cv2.computeCorrespondEpilines(query_points_f32, 1, fund_mat)).T

    return intersect(a, b, c, xp_i16)


@benchmark
def compute_disparity(
    yp_f32,
    xp_i16,
    xcr_f32,
    disp_proj_mapx,
    disp_proj_mapy,
    rect_shape,
    projector_K,
    projector_D,
    R2,
    P2,
    DISP_MAX,
    PRECOMP_RECT,
):
    if PRECOMP_RECT:
        yp_i16 = np.rint(yp_f32).astype(np.int16)
        yp_inlier_mask = (yp_i16 >= 0) & (yp_i16 < disp_proj_mapy.shape[0])

        yp_i16_filt = yp_i16[yp_inlier_mask]
        xp_i16_filt = xp_i16[yp_inlier_mask]

        xpr_f32 = disp_proj_mapx[yp_i16_filt, xp_i16_filt]
        ypr_f32 = disp_proj_mapy[yp_i16_filt, xp_i16_filt]
    else:
        yp_inlier_mask = (yp_f32 >= 0) & (yp_f32 < disp_proj_mapy.shape[0])
        yp_f32_filt = yp_f32[yp_inlier_mask]
        xp_i16_filt = xp_i16[yp_inlier_mask]

        xyp_f32_filt = np.stack((xp_i16_filt, yp_f32_filt)).T.astype(np.float32)
        xypr_f32_filt = cv2.undistortPoints(xyp_f32_filt, projector_K, projector_D, R=R2, P=P2)

        xpr_f32 = xypr_f32_filt[:, 0, 0]
        ypr_f32 = xypr_f32_filt[:, 0, 1]

    # assert np.allclose(ypr, ycr[yp_inlier_mask])
    # TODO minimize np.abs(ypr - ycr[yp_inlier_mask]).sum()

    disp_f32 = xpr_f32 - xcr_f32[yp_inlier_mask]

    y_inlier_mask = (ypr_f32 >= 0) & (ypr_f32 < rect_shape[0] - 0.5)
    x_inlier_mask = (xpr_f32 >= 0) & (xpr_f32 < rect_shape[1] - 0.5)

    disp_inlier_mask = (disp_f32 >= 0) & (disp_f32 <= DISP_MAX)

    inlier_mask = y_inlier_mask & x_inlier_mask & disp_inlier_mask

    yp_inlier_mask[yp_inlier_mask] = inlier_mask

    return (
        disp_f32[inlier_mask],
        xpr_f32[inlier_mask],
        ypr_f32[inlier_mask],
        yp_inlier_mask,
    )


@benchmark
def compute_epipolar_disparity(
    events,
    proj_width,
    disp_cam_mapx,
    disp_cam_mapy,
    disp_cam_mapx_undist,
    disp_cam_mapy_undist,
    disp_proj_mapx,
    disp_proj_mapy,
    fund_mat,
    rect_shape,
    P2,
    Q,
    R2,
    projector_K,
    projector_D,
    DISP_MAX=900,
    PRECOMP_RECT=False,
):
    # events = mean_first_last_event_per_xy(events)
    # events = first_event_per_xy(events)
    events = events[events["p"] == 1]

    # projector x from event time, rounded to nearest int
    xp_i16 = compute_xp(events, proj_width)

    events = first_event_per_yt(events, xp_i16)

    # TODO fixme is just a simple mask
    # redo after events are filtered
    xp_i16 = compute_xp(events, proj_width)

    yp_f32 = intersect_cam_epipolar_line_with_proj_x(
        events, fund_mat, disp_cam_mapx_undist, disp_cam_mapy_undist, xp_i16
    )

    xcr_f32, ycr_f32 = rectify_cam_coords(disp_cam_mapx, disp_cam_mapy, events)

    disp_f32, xpr_f32, ypr_f32, inlier_mask = compute_disparity(
        yp_f32,
        xp_i16,
        xcr_f32,
        disp_proj_mapx,
        disp_proj_mapy,
        rect_shape,
        projector_K,
        projector_D,
        R2,
        P2,
        DISP_MAX,
        PRECOMP_RECT,
    )

    # import pandas as pd

    # df = pd.DataFrame(
    #     {"xpr": xpr_f32, "ypr": ypr_f32, "disp": disp_f32, "xcr": xcr_f32[inlier_mask], "ycr": ycr_f32[inlier_mask]}
    # )

    point_cloud = construct_point_cloud(xpr_f32, ypr_f32, disp_f32, Q)

    # TODO remove dispf == 0 events before division
    # depth = P2[0, 3] / dispf
    # depth[dispf == 0] = 0.0

    # point_cloud = points * depth[:, np.newaxis]

    # disp_map = np.zeros(rect_shape, dtype=np.float32)

    # ypr_dispf = np.rint(ypr_f32[inlier_mask]).astype(np.int16)
    # xpr_dispf = np.rint(xpr_f32[inlier_mask]).astype(np.int16)

    # TODO: choice between ypr and ycr
    # disp_map[ypr_dispf, xpr_dispf] = disp[inlier_mask]

    return point_cloud
