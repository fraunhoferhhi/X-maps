import numpy as np
from metavision_sdk_base import EventCD
from dataclasses import dataclass
from typing import Callable
from stats_printer import StatsPrinter


MIN_EVENTS_PER_FRAME = 1000


@dataclass
class RobustTriggerFinder:
    projector_fps: int
    stats: StatsPrinter
    callback: Callable[[np.ndarray], None]

    frame_paused_thresh_us = 40
    should_drop = False
    ev_buf = np.array([], dtype=EventCD)

    @property
    def frame_len_ms(self):
        return 1e3 / self.projector_fps

    def reset_buffer(self):
        self.ev_buf = np.array([], dtype=EventCD)

    def drop_frame(self):
        self.should_drop = True

    def process_events(self, evs):
        # TODO perf use a ring buffer
        # e.g. np-rw-buffer
        # TODO investigate: EventIterator from Metavision just uses a deque in MetaEventBufferProducer
        with self.stats.measure_time("append events"):
            self.ev_buf = np.append(self.ev_buf, evs.numpy())

        if self.should_drop:
            # drop one frame worth of events
            buf_start_t = self.ev_buf["t"][0]
            evs_next_frame_or_later = self.ev_buf["t"] >= buf_start_t + self.frame_len_ms * 1000
            if not evs_next_frame_or_later.any():
                # we don't have a frame worth's events to drop yet, return and request more
                return

            next_frame_first_event = np.argmax(evs_next_frame_or_later)
            self.ev_buf = self.ev_buf[next_frame_first_event:]
            self.stats.count("frames dropped")
            self.should_drop = False

        if len(self.ev_buf) < 2:
            return

        # if we have fewer than one frame worth of events, don't bother
        if self.ev_buf["t"][-1] - self.ev_buf["t"][0] < 1e6 / self.projector_fps:
            return

        self.stats.add_metric("evs in buf", len(self.ev_buf))

        ev_time = self.find_trigger() / 1000
        if ev_time > 0:
            self.stats.count("trig ✅")
        else:
            self.stats.count("trig ❌")

    def find_trigger(self):
        """function to get the frame starts and ends from the current buffer."""

        # find the indicies of the events which have a relative long time difference to their respective next event in the buffer

        # perf: is this recomputed a lot? - no, not at all, usually there's a whole frame worth of events here, so it is not
        with self.stats.measure_time("find pauses"):
            frame_paused_ev_idx = np.nonzero(np.diff(self.ev_buf["t"]) >= self.frame_paused_thresh_us)[0]

        # offline: avg 4
        # online: avg 2
        # self.stats.add_metric("frame pauses", len(frame_paused_ev_idx))

        for prev_idx, next_idx in zip(frame_paused_ev_idx[:-1], frame_paused_ev_idx[1:]):
            # time between event pauses
            diff_t_event_pauses = self.ev_buf["t"][next_idx] - self.ev_buf["t"][prev_idx]

            # if this time is greater than half of the frametime, it
            # is assumend that the two events are the first and last events of the frame

            if diff_t_event_pauses > 1e6 / self.projector_fps / 2:
                if diff_t_event_pauses <= 1e6 / self.projector_fps and next_idx - prev_idx > MIN_EVENTS_PER_FRAME:
                    # TODO reexamine the +2 and -2
                    # The interval is trimmed a a little to ensure that the events are in the frame.
                    self.callback(self.ev_buf[prev_idx + 2 : next_idx - 2])

                    start_time = self.ev_buf["t"][prev_idx + 2]
                    end_time = self.ev_buf["t"][next_idx - 2]

                    self.stats.add_metric("frame len [ms]", (end_time - start_time) / 1000)

                    self.ev_buf = self.ev_buf[next_idx - 2 :]
                    return start_time
                else:
                    # trigger not found correctly, drop these events
                    self.ev_buf = self.ev_buf[next_idx:]
                    return -1

        return -1
