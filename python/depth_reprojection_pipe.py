from typing import List

from metavision_sdk_core import PolarityFilterAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_ui import BaseWindow, MTWindow, UIAction, UIKeyEvent

from trigger_finder import RobustTriggerFinder
from stats_printer import StatsPrinter, SingleTimer
from cam_proj_calibration import CamProjMaps, CamProjCalibrationParams
from x_maps_disparity import XMapsDisparity
from proj_time_map import ProjectorTimeMap
from disp_to_depth import DisparityToDepth
from timing_watchdog import TimingWatchdog

from dataclasses import dataclass, field

USE_FAKE_WINDOW = False


class FakeWindow:
    def should_close(self):
        return False

    def show_async(self, img):
        pass

    def set_keyboard_callback(self, cb):
        pass


@dataclass
class Pool:
    bufs: List["EventCDBuffer"] = field(default_factory=list)

    def get_buf(self):
        if not self.bufs:
            return PolarityFilterAlgorithm.get_empty_output_buffer()
        return self.bufs.pop()

    def return_buf(self, buf):
        self.bufs.append(buf)


@dataclass
class RuntimeParams:
    camera_width: int
    camera_height: int

    projector_width: int
    projector_height: int

    projector_fps: int

    z_near: float
    z_far: float

    calib: str

    projector_time_map: str

    no_frame_dropping: bool

    camera_perspective: bool

    @property
    def should_drop_frames(self):
        return not self.no_frame_dropping


@dataclass
class DepthReprojectionPipe:
    params: RuntimeParams

    _pool = Pool()

    pos_filter = PolarityFilterAlgorithm(1)

    # TODO revisit: does this have an effect on latency?
    act_filter = None

    pos_events_buf = None
    # act_events_buf = None

    trigger_finder = None

    x_maps_disp: XMapsDisparity = None
    disp_to_depth: DisparityToDepth = None
    stats_printer: StatsPrinter = StatsPrinter()

    watchdog: TimingWatchdog = None

    @property
    def camera_width(self):
        return self.params.camera_width

    @property
    def camera_height(self):
        return self.params.camera_height

    @property
    def projector_width(self):
        return self.params.projector_width

    @property
    def projector_height(self):
        return self.params.projector_height

    @property
    def projector_fps(self):
        return self.params.projector_fps

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
            point_cloud, disp_map = self.x_maps_disp.compute_event_disparity(
                evs,
                projector_view=not self.params.camera_perspective,
                rectified_view=not self.params.camera_perspective,
            )

        if not self.params.camera_perspective:
            with self.stats_printer.measure_time("remap disp"):
                disp_map = self.disp_to_depth.remap_rectified_disp_map_to_proj(disp_map)

        with self.stats_printer.measure_time("disp2rgb"):
            depth_map = self.disp_to_depth.colorize_depth_from_disp(disp_map)

        self.window.show_async(depth_map)
        self.stats_printer.count("frames shown")

    def __enter__(self):
        self.act_filter = ActivityNoiseFilterAlgorithm(self.camera_width, self.camera_height, self.activity_time_ths)

        self.pos_events_buf = PolarityFilterAlgorithm.get_empty_output_buffer()
        # self.act_events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

        self.watchdog = TimingWatchdog(stats_printer=self.stats_printer, projector_fps=self.projector_fps)

        self.trigger_finder = RobustTriggerFinder(
            projector_fps=self.projector_fps, stats=self.stats_printer, callback=self.on_frame_evs, pool=self._pool
        )

        with SingleTimer("Setting up calibration"):
            calib_params = CamProjCalibrationParams.from_yaml(
                self.params.calib, self.camera_width, self.camera_height, self.projector_width, self.projector_height
            )
            calib_maps = CamProjMaps(calib_params)

        with SingleTimer("Setting up projector time map"):
            if self.params.projector_time_map is not None:
                proj_time_map = ProjectorTimeMap.from_file(self.params.projector_time_map)
            else:
                proj_time_map = ProjectorTimeMap.from_calib(calib_params, calib_maps)

        with SingleTimer("Setting up projector X-map"):
            self.x_maps_disp = XMapsDisparity(calib_params, calib_maps, proj_time_map, self.projector_width)

        with SingleTimer("Setting up disparity to depth"):
            self.disp_to_depth = DisparityToDepth(self.stats_printer, calib_maps, self.params.z_near, self.params.z_far)

        if USE_FAKE_WINDOW:
            self.window = FakeWindow()
        else:
            self.window = MTWindow(
                title="X Maps Depth",
                width=self.camera_width if self.params.camera_perspective else self.projector_width,
                height=self.camera_height if self.params.camera_perspective else self.projector_height,
                mode=BaseWindow.RenderMode.BGR,
                open_directly=True,
            )
            print(
                """
Available keyboard shortcuts:
- S:     Toggle printing statistics
- Q/Esc: Quit the application"""
            )

        self.window.set_keyboard_callback(self.keyboard_cb)

        return self

    def __exit__(self, *exc_info):
        self.stats_printer.print_stats()
        return False

    def keyboard_cb(self, key, scancode, action, mods):
        if action != UIAction.RELEASE:
            return
        if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
            self.window.set_close_flag()
        if key == UIKeyEvent.KEY_S:
            self.stats_printer.toggle_silence()

    def process_events(self, evs):
        if self.watchdog.is_processing_behind(evs) and self.params.should_drop_frames:
            self.trigger_finder.drop_frame()

        self.stats_printer.print_stats_if_needed()
        self.stats_printer.count("processed evs", len(evs))

        self.pos_filter.process_events(evs, self.pos_events_buf)

        act_out_buf = self._pool.get_buf()
        self.act_filter.process_events(self.pos_events_buf, act_out_buf)

        self.trigger_finder.process_events(act_out_buf)

        self.stats_printer.print_stats_if_needed()

    def reset(self):
        self.watchdog.reset()
        self.trigger_finder.reset()
