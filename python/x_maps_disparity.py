import numpy as np

from dataclasses import dataclass, field, InitVar

from x_map import compute_x_map_from_time_map
from cam_proj_calibration import CamProjCalibrationParams, CamProjMaps


def compute_disparity(xcr_i16, ycr_i16, t, proj_x_map, T_PX_SCALE, X_OFFSET):
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
    y_inlier_mask = (ycr_i16 >= 0) & (ycr_i16 < proj_x_map.shape[0] - 1)

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

    def compute_event_disparity(
        self,
        events,
        ev_x_rect_i16,
        ev_y_rect_i16,
    ):
        # at time t and rectified y, access X-map
        # note: ev_disparity_f32 may be shorter original events list
        # because some events may lie outside the projector X-map.
        # events[inlier_mask] can be used to trim the original events list
        ev_disparity_f32, inlier_mask = compute_disparity(
            ev_x_rect_i16, ev_y_rect_i16, events["t"], self.proj_x_map, self.T_PX_SCALE, self.X_OFFSET
        )
        return ev_disparity_f32, inlier_mask
