import numba
import numpy as np

from dataclasses import dataclass, field, InitVar

from x_map import compute_x_map_from_time_map
from cam_proj_calibration import CamProjCalibrationParams, CamProjMaps

from epipolar_disparity import (
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


@dataclass
class XMapsDisparity:
    calib_params: CamProjCalibrationParams
    cam_proj_maps: CamProjMaps

    proj_time_map_rect: InitVar[np.ndarray]

    proj_x_map: np.ndarray = field(init=False)
    disp_map_shape: tuple = field(init=False)

    def __post_init__(self, proj_time_map_rect):
        """Setup the projector X-map for disparity lookup"""

        # we want to differentiate between x=0 and x undefined
        # so we add an offset to the x values -> x=0 starts at x'=X_OFFSET, x' < X_OFFSET means x is undefined
        self.X_OFFSET = 4242

        # using 16 bit for indices, make sure we don't overflow
        assert proj_time_map_rect.shape[0] <= 2**15 - 1
        assert proj_time_map_rect.shape[1] + self.X_OFFSET <= 2**15 - 1

        # the time axis can be freely discretized
        # we choose the projector width as the number of time steps
        # which should allow different scan lines to map to different time columns
        self.X_MAP_WIDTH = self.calib_params.projector_width
        self.T_PX_SCALE = self.X_MAP_WIDTH - 1

        self.proj_x_map, t_diffs = compute_x_map_from_time_map(
            time_map=proj_time_map_rect,
            x_map_width=self.X_MAP_WIDTH,
            t_px_scale=self.T_PX_SCALE,
            X_OFFSET=self.X_OFFSET,
            num_scanlines=self.calib_params.projector_width,
        )

        self.disp_map_shape = proj_time_map_rect.shape

    def compute_event_disparity(
        self,
        events,
        ev_x_rect_f32,
        ev_y_rect_f32,
        compute_point_cloud=False,
        compute_disp_map=True,
        projector_view=True,
        rectified_view=True,
    ):
        # at time t and rectified y, access X-map
        # note: ev_disparity_f32 may be shorter original events list
        # because some events may lie outside the projector X-map.
        # events[inlier_mask] can be used to trim the original events list
        ev_disparity_f32, inlier_mask = compute_disparity(
            ev_x_rect_f32, ev_y_rect_f32, events["t"], self.proj_x_map, self.T_PX_SCALE, self.X_OFFSET
        )

        point_cloud, disp_map = None, None

        if compute_point_cloud:
            point_cloud = construct_point_cloud(
                ev_x_rect_f32[inlier_mask] + ev_disparity_f32,
                ev_y_rect_f32[inlier_mask],
                ev_disparity_f32,
                self.cam_proj_maps.Q,
            )

        if compute_disp_map:
            # ypr_dispf = np.rint(ypr_f32[inlier_mask]).astype(np.int16)
            # xpr_dispf = np.rint(xpr_f32[inlier_mask]).astype(np.int16)

            # TODO: choice between ypr and ycr
            # disp_map[ypr_dispf, xpr_dispf] = disp[inlier_mask]

            # TODO + 0.5?
            if rectified_view:
                ycr_i16 = np.rint(ev_y_rect_f32[inlier_mask]).astype(np.int16)

            # TODO should one of the disparities actually be negative, assuption: no, since then the baseline also must be negative and it would cancel out ? (N.G.)
            # TODO instead of using the rounded rectified coordinates, be could also use the input event coordinates
            if projector_view:
                if not rectified_view:
                    print("Projector view not implemented. Return rectified view")
                    return point_cloud, None

                xpr_i16 = np.rint(ev_x_rect_f32[inlier_mask] + ev_disparity_f32).astype(np.int16)
                disp_map = np.zeros(self.disp_map_shape, dtype=np.float32)
                disp_map[ycr_i16, xpr_i16] = ev_disparity_f32
            else:
                if rectified_view:
                    xcr_i16 = np.rint(ev_x_rect_f32[inlier_mask]).astype(np.int16)
                    disp_map = np.zeros(self.disp_map_shape, dtype=np.float32)
                    disp_map[ycr_i16, xcr_i16] = ev_disparity_f32
                else:
                    x_cam = events["x"][inlier_mask]
                    y_cam = events["y"][inlier_mask]
                    disp_map = np.zeros(
                        (self.calib_params.camera_height, self.calib_params.camera_width), dtype=np.float32
                    )
                    disp_map[y_cam, x_cam] = ev_disparity_f32

        # dump_frame_data(
        #     events,
        #     inlier_mask,
        #     xcr_f32,
        #     ycr_f32,
        #     disp_f32,
        # )

        return point_cloud, disp_map
