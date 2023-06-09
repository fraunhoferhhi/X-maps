import numba
import numpy as np

from x_map import compute_x_map_from_time_map

from epipolar_disparity import (
    rectify_cam_coords,
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

    # spurious events create from scene movement or noise may lie outside the projector X-map
    # these events will be filtered out
    y_inlier_mask = (ycr_f32 >= 0) & (ycr_f32 < proj_x_map.shape[0] - 1)

    # TODO use cv2.remap to retrieve with interpolation from proj_x_map
    # TODO subpixel + 0.5
    x_proj = proj_x_map[ycr_i16[y_inlier_mask], t_scaled[y_inlier_mask]]

    disp = x_proj - xcr_i16[y_inlier_mask] - X_OFFSET

    disp_inlier_mask = disp >= 0
    y_inlier_mask[y_inlier_mask] = disp_inlier_mask

    return disp[disp_inlier_mask], y_inlier_mask


class XMapsDisparity:
    def __init__(self, calib_params, calib_maps, proj_time_map, proj_width):
        self.calib_params = calib_params
        self.cam_proj_maps = calib_maps
        self.init_proj_x_map(proj_time_map.projector_time_map_rectified, proj_width)
        self.disp_map_shape = proj_time_map.projector_time_map_rectified.shape

    def init_proj_x_map(self, proj_time_map, proj_width):
        """Setup the projector X-map for disparity lookup"""

        # we want to differentiate between x=0 and x undefined
        # so we add an offset to the x values -> x=0 starts at x'=X_OFFSET, x' < X_OFFSET means x is undefined
        self.X_OFFSET = 4242

        # using 16 bit for indices, make sure we don't overflow
        assert proj_time_map.shape[0] <= 2**15 - 1
        assert proj_time_map.shape[1] + self.X_OFFSET <= 2**15 - 1

        # the time axis can be freely discretized
        # we choose the projector width as the number of time steps
        # which should allow different scan lines to map to different time columns
        self.X_MAP_WIDTH = proj_width
        self.T_PX_SCALE = self.X_MAP_WIDTH - 1

        self.proj_x_map, t_diffs = compute_x_map_from_time_map(
            time_map=proj_time_map,
            x_map_width=self.X_MAP_WIDTH,
            t_px_scale=self.T_PX_SCALE,
            X_OFFSET=self.X_OFFSET,
            num_scanlines=proj_width,
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
        xcr_f32, ycr_f32 = rectify_cam_coords(
            self.cam_proj_maps.disp_cam_mapx, self.cam_proj_maps.disp_cam_mapy, events
        )

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
                xcr_f32[inlier_mask] + disp_f32, ycr_f32[inlier_mask], disp_f32, self.cam_proj_maps.Q
            )

        if compute_disp_map:
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
                disp_map = np.zeros(self.disp_map_shape, dtype=np.float32)
                disp_map[ycr_i16, xpr_i16] = disp_f32
            else:
                if rectified_view:
                    xcr_i16 = np.rint(xcr_f32[inlier_mask]).astype(np.int16)
                    disp_map = np.zeros(self.disp_map_shape, dtype=np.float32)
                    disp_map[ycr_i16, xcr_i16] = disp_f32
                else:
                    x_cam = events["x"][inlier_mask]
                    y_cam = events["y"][inlier_mask]
                    disp_map = np.zeros(
                        (self.calib_params.camera_height, self.calib_params.camera_width), dtype=np.float32
                    )
                    disp_map[y_cam, x_cam] = disp_f32

        # dump_frame_data(
        #     events,
        #     inlier_mask,
        #     xcr_f32,
        #     ycr_f32,
        #     disp_f32,
        # )

        return point_cloud, disp_map
