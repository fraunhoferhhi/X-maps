from metavision_sdk_core import PolarityFilterAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_ui import BaseWindow, MTWindow, UIAction, UIKeyEvent

from trigger_finder import RobustTriggerFinder
from stats_printer import StatsPrinter, SingleTimer
from cam_proj_calibration import CamProjCalibration
from x_maps_disparity import XMapsDisparity
from proj_time_map import ProjectorTimeMap
from disp_to_depth import DisparityToDepth

import time
from dataclasses import dataclass


@dataclass
class DepthReprojectionPipe:
    camera_width: int
    camera_height: int

    projector_width: int
    projector_height: int

    projector_fps: int

    should_drop_frames: bool

    first_event_time_us: int = -1
    start_time: int = -1

    pos_filter = PolarityFilterAlgorithm(1)

    # TODO revisit: does this have an effect on latency?
    act_filter = None

    pos_events_buf = None
    act_events_buf = None

    trigger_finder = None

    x_maps_disp: XMapsDisparity = None
    disp_to_depth: DisparityToDepth = None
    stats_printer: StatsPrinter = StatsPrinter()

    @property
    def activity_time_ths(self):
        return int(1e6 / self.projector_fps)

    def should_close(self):
        return self.window.should_close()

    def on_frame_evs(self, evs):
        """Callback from the trigger finder, evs contain the events of the current frame"""
        # generate_frame(evs, frame)
        # window.show_async(frame)

        with self.stats_printer.measure_time("x-maps disp"):
            point_cloud, disp_map = self.x_maps_disp.compute_event_disparity(evs)

        with self.stats_printer.measure_time("disp2depth"):
            depth_map = self.disp_to_depth.compute_depth_map(disp_map)

        self.window.show_async(depth_map)
        self.stats_printer.count("frames shown")

    def setup(self, cli_params):
        self.act_filter = ActivityNoiseFilterAlgorithm(self.camera_width, self.camera_height, self.activity_time_ths)

        self.pos_events_buf = PolarityFilterAlgorithm.get_empty_output_buffer()
        self.act_events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

        self.trigger_finder = RobustTriggerFinder(
            projector_fps=self.projector_fps, stats=self.stats_printer, callback=self.on_frame_evs
        )

        with SingleTimer("Setting up calibration"):
            calib_obj = CamProjCalibration(
                cli_params["calib"], self.camera_width, self.camera_height, self.projector_width, self.projector_height
            )

        with SingleTimer("Setting up projector time map"):
            proj_time_map = ProjectorTimeMap(calib_obj, cli_params["projector_time_map"])

        with SingleTimer("Setting up projector X-map"):
            self.x_maps_disp = XMapsDisparity(calib_obj, proj_time_map, self.projector_width)

        with SingleTimer("Setting up disparity to depth"):
            self.disp_to_depth = DisparityToDepth(
                self.stats_printer, calib_obj, cli_params["z_near"], cli_params["z_far"]
            )

        self.window = MTWindow(
            title="X Maps Depth",
            width=self.projector_width,
            height=self.projector_height,
            mode=BaseWindow.RenderMode.BGR,
            open_directly=True,
        )
        self.window.set_keyboard_callback(self.keyboard_cb)

    def keyboard_cb(self, key, scancode, action, mods):
        if action != UIAction.RELEASE:
            return
        if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
            self.window.set_close_flag()

    def process_events(self, evs):
        if self.first_event_time_us == -1:
            self.first_event_time_us = evs["t"][0]
            self.start_time = time.perf_counter_ns()

        ev_time_diff_ns = (evs["t"][0] - self.first_event_time_us) * 1000
        proc_time_diff_ns = time.perf_counter_ns() - self.start_time
        proc_behind = proc_time_diff_ns - ev_time_diff_ns

        self.stats_printer.add_time_measure_ns("(cpu t - ev[0] t)", proc_behind)

        frames_behind_i = int(proc_behind / (1000 * 1000 * 1000 / self.projector_fps))
        self.stats_printer.add_metric("frames behind", frames_behind_i)
        if frames_behind_i > 0 and self.should_drop_frames:
            self.trigger_finder.drop_frame()

        self.stats_printer.print_stats_if_needed()
        self.stats_printer.count("processed evs", len(evs))

        self.pos_filter.process_events(evs, self.pos_events_buf)
        self.act_filter.process_events(self.pos_events_buf, self.act_events_buf)

        self.trigger_finder.process_events(self.act_events_buf)

        self.stats_printer.print_stats_if_needed()
