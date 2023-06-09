from typing import Any, Callable

from metavision_sdk_core import PolarityFilterAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm

from trigger_finder import RobustTriggerFinder
from stats_printer import StatsPrinter, SingleTimer
from cam_proj_calibration import CamProjMaps, CamProjCalibrationParams
from x_maps_disparity import XMapsDisparity
from proj_time_map import ProjectorTimeMap
from disp_to_depth import DisparityToDepth
from timing_watchdog import TimingWatchdog
from event_buf_pool import EventBufPool
from frame_event_filter import FrameEventFilterProcessor

from dataclasses import dataclass, field


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


@dataclass
class DepthReprojectionPipe:
    params: "RuntimeParams"
    stats_printer: StatsPrinter
    frame_callback: Callable

    pos_filter = PolarityFilterAlgorithm(1)

    # TODO revisit: does this have an effect on latency?
    act_filter: ActivityNoiseFilterAlgorithm = field(init=False)

    pos_events_buf = PolarityFilterAlgorithm.get_empty_output_buffer()
    # act_events_buf = None

    calib_maps: CamProjMaps = field(init=False)

    trigger_finder: RobustTriggerFinder = field(init=False)

    ev_filter_proc = FrameEventFilterProcessor()

    x_maps_disp: XMapsDisparity = field(init=False)
    disp_to_depth: DisparityToDepth = field(init=False)

    watchdog: TimingWatchdog = field(init=False)

    pool = EventBufPool()

    def __post_init__(self):
        self.act_filter = ActivityNoiseFilterAlgorithm(
            self.params.camera_width, self.params.camera_height, int(1e6 / self.params.projector_fps)
        )

        with SingleTimer("Setting up calibration"):
            calib_params = CamProjCalibrationParams.from_yaml(
                self.params.calib,
                self.params.camera_width,
                self.params.camera_height,
                self.params.projector_width,
                self.params.projector_height,
            )
            self.calib_maps = CamProjMaps(calib_params)

        with SingleTimer("Setting up projector time map"):
            if self.params.projector_time_map is not None:
                proj_time_map = ProjectorTimeMap.from_file(self.params.projector_time_map)
            else:
                proj_time_map = ProjectorTimeMap.from_calib(calib_params, self.calib_maps)

        with SingleTimer("Setting up projector X-map"):
            self.x_maps_disp = XMapsDisparity(
                calib_params=calib_params,
                cam_proj_maps=self.calib_maps,
                proj_time_map_rect=proj_time_map.projector_time_map_rectified,
            )

        with SingleTimer("Setting up disparity to depth"):
            self.disp_to_depth = DisparityToDepth(
                stats=self.stats_printer,
                calib_params=calib_params,
                calib_maps=self.calib_maps,
                z_near=self.params.z_near,
                z_far=self.params.z_far,
            )

        self.trigger_finder = RobustTriggerFinder(
            projector_fps=self.params.projector_fps,
            stats=self.stats_printer,
            pool=self.pool,
            frame_callback=self.process_ev_frame,
        )

        self.watchdog = TimingWatchdog(stats_printer=self.stats_printer, projector_fps=self.params.projector_fps)

    def process_events(self, evs):
        if self.watchdog.is_processing_behind(evs) and self.params.should_drop_frames:
            self.trigger_finder.drop_frame()

        self.pos_filter.process_events(evs, self.pos_events_buf)

        act_out_buf = self.pool.get_buf()
        self.act_filter.process_events(self.pos_events_buf, act_out_buf)

        self.trigger_finder.process_events(act_out_buf)

    def process_ev_frame(self, evs):
        """Callback from the trigger finder, evs contain the events of the current frame"""
        # generate_frame(evs, frame)
        # window.show_async(frame)

        with self.stats_printer.measure_time("ev rect"):
            # get rectified event coordinates
            ev_x_rect_i16, ev_y_rect_i16 = self.calib_maps.rectify_cam_coords_i16(evs)

        with self.stats_printer.measure_time("frame ev filter"):
            filtered_evs = self.ev_filter_proc.filter_events(evs, ev_x_rect_i16)
            self.stats_printer.add_metric("frame evs filtered out [%]", 100 - len(filtered_evs) / len(evs) * 100)

            # redo the rectification, because we don't know which events were filtered out
            # TODO perf y coords aren't used in the filtering, are computed twice
            if len(filtered_evs) < len(evs):
                ev_x_rect_i16, ev_y_rect_i16 = self.calib_maps.rectify_cam_coords_i16(filtered_evs)

            evs = filtered_evs

        with self.stats_printer.measure_time("x-maps disp"):
            ev_disparity_f32, inlier_mask = self.x_maps_disp.compute_event_disparity(
                events=evs,
                ev_x_rect_i16=ev_x_rect_i16,
                ev_y_rect_i16=ev_y_rect_i16,
            )

        if self.params.camera_perspective:
            with self.stats_printer.measure_time("disp map"):
                disp_map = self.calib_maps.compute_disp_map_camera_view(
                    events=evs, inlier_mask=inlier_mask, ev_disparity_f32=ev_disparity_f32
                )
        else:
            with self.stats_printer.measure_time("disp map"):
                disp_map = self.calib_maps.compute_disp_map_projector_view(
                    ev_x_rect_i16=ev_x_rect_i16,
                    ev_y_rect_i16=ev_y_rect_i16,
                    inlier_mask=inlier_mask,
                    ev_disparity_f32=ev_disparity_f32,
                )
            with self.stats_printer.measure_time("remap disp"):
                disp_map = self.disp_to_depth.remap_rectified_disp_map_to_proj(disp_map)

        with self.stats_printer.measure_time("disp2rgb"):
            depth_map = self.disp_to_depth.colorize_depth_from_disp(disp_map)

        self.frame_callback(depth_map)

    def select_next_frame_event_filter(self):
        new_filter = self.ev_filter_proc.select_next_filter()
        self.stats_printer.log(f"Selected event filter: {new_filter}")

    def reset(self):
        self.watchdog.reset()
        self.trigger_finder.reset()
