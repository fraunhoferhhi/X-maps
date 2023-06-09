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

from dataclasses import dataclass


@dataclass
class DepthReprojectionPipe:
    params: "RuntimeParams"

    stats_printer: StatsPrinter

    frame_callback: Callable

    _pos_filter = PolarityFilterAlgorithm(1)

    # TODO revisit: does this have an effect on latency?
    _act_filter: ActivityNoiseFilterAlgorithm

    _pos_events_buf: Any
    # act_events_buf = None

    _trigger_finder: RobustTriggerFinder

    _x_maps_disp: XMapsDisparity
    _disp_to_depth: DisparityToDepth

    _watchdog: TimingWatchdog

    _pool: EventBufPool

    @staticmethod
    def create(params: "RuntimeParams", stats_printer: StatsPrinter, frame_callback: Callable):
        pool = EventBufPool()

        with SingleTimer("Setting up calibration"):
            calib_params = CamProjCalibrationParams.from_yaml(
                params.calib, params.camera_width, params.camera_height, params.projector_width, params.projector_height
            )
            calib_maps = CamProjMaps(calib_params)

        with SingleTimer("Setting up projector time map"):
            if params.projector_time_map is not None:
                proj_time_map = ProjectorTimeMap.from_file(params.projector_time_map)
            else:
                proj_time_map = ProjectorTimeMap.from_calib(calib_params, calib_maps)

        with SingleTimer("Setting up projector X-map"):
            x_maps_disp = XMapsDisparity(
                calib_params=calib_params,
                cam_proj_maps=calib_maps,
                proj_time_map_rect=proj_time_map.projector_time_map_rectified,
            )

        with SingleTimer("Setting up disparity to depth"):
            disp_to_depth = DisparityToDepth(stats_printer, calib_maps, params.z_near, params.z_far)

        trigger_finder = RobustTriggerFinder(projector_fps=params.projector_fps, stats=stats_printer, pool=pool)

        pipe = DepthReprojectionPipe(
            params=params,
            stats_printer=stats_printer,
            frame_callback=frame_callback,
            _act_filter=ActivityNoiseFilterAlgorithm(
                params.camera_width, params.camera_height, int(1e6 / params.projector_fps)
            ),
            _pos_events_buf=PolarityFilterAlgorithm.get_empty_output_buffer(),
            _trigger_finder=trigger_finder,
            _x_maps_disp=x_maps_disp,
            _disp_to_depth=disp_to_depth,
            _watchdog=TimingWatchdog(stats_printer=stats_printer, projector_fps=params.projector_fps),
            _pool=pool,
        )

        trigger_finder.register_callback(pipe.process_ev_frame)

        return pipe

    def process_events(self, evs):
        if self._watchdog.is_processing_behind(evs) and self.params.should_drop_frames:
            self._trigger_finder.drop_frame()

        self._pos_filter.process_events(evs, self._pos_events_buf)

        act_out_buf = self._pool.get_buf()
        self._act_filter.process_events(self._pos_events_buf, act_out_buf)

        self._trigger_finder.process_events(act_out_buf)

    def process_ev_frame(self, evs):
        """Callback from the trigger finder, evs contain the events of the current frame"""
        # generate_frame(evs, frame)
        # window.show_async(frame)

        with self.stats_printer.measure_time("x-maps disp"):
            point_cloud, disp_map = self._x_maps_disp.compute_event_disparity(
                evs,
                projector_view=not self.params.camera_perspective,
                rectified_view=not self.params.camera_perspective,
            )

        if not self.params.camera_perspective:
            with self.stats_printer.measure_time("remap disp"):
                disp_map = self._disp_to_depth.remap_rectified_disp_map_to_proj(disp_map)

        with self.stats_printer.measure_time("disp2rgb"):
            depth_map = self._disp_to_depth.colorize_depth_from_disp(disp_map)

        self.frame_callback(depth_map)

    def reset(self):
        self._watchdog.reset()
        self._trigger_finder.reset()
