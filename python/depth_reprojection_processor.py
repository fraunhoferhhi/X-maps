from typing import Any, Callable, Optional

from metavision_sdk_ui import BaseWindow, MTWindow, UIAction, UIKeyEvent

from depth_reprojection_pipe import DepthReprojectionPipe
from stats_printer import StatsPrinter

from dataclasses import dataclass, field

USE_FAKE_WINDOW = False


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


class FakeWindow:
    def should_close(self):
        return False

    def show_async(self, img):
        pass

    def set_keyboard_callback(self, cb):
        pass


@dataclass
class DepthReprojectionProcessor:
    params: RuntimeParams

    stats_printer: StatsPrinter = StatsPrinter()

    _pipe: Optional[DepthReprojectionPipe] = None
    _window: BaseWindow = field(init=False)

    def should_close(self):
        return self._window.should_close()

    def show_async(self, depth_map):
        self._window.show_async(depth_map)
        self.stats_printer.count("frames shown")

    def __enter__(self):
        self._pipe = DepthReprojectionPipe.create(
            params=self.params, stats_printer=self.stats_printer, frame_callback=self.show_async
        )

        if USE_FAKE_WINDOW:
            self._window = FakeWindow()
        else:
            self._window = MTWindow(
                title="X Maps Depth",
                width=self.params.camera_width if self.params.camera_perspective else self.params.projector_width,
                height=self.params.camera_height if self.params.camera_perspective else self.params.projector_height,
                mode=BaseWindow.RenderMode.BGR,
                open_directly=True,
            )
            print(
                """
Available keyboard shortcuts:
- S:     Toggle printing statistics
- Q/Esc: Quit the application"""
            )

        self._window.set_keyboard_callback(self.keyboard_cb)

        return self

    def __exit__(self, *exc_info):
        self.stats_printer.print_stats()
        return False

    def keyboard_cb(self, key, scancode, action, mods):
        if action != UIAction.RELEASE:
            return
        if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
            self._window.set_close_flag()
        if key == UIKeyEvent.KEY_S:
            self.stats_printer.toggle_silence()

    def process_events(self, evs):
        self.stats_printer.print_stats_if_needed()
        self.stats_printer.count("processed evs", len(evs))
        self._pipe.process_events(evs)
        self.stats_printer.print_stats_if_needed()

    def reset(self):
        self._pipe.reset()
