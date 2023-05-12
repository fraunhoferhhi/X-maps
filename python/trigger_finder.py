# TODO find trigger stat time is so long because the callback is part of it, and is measured with it
#      --> measure trigger finding by itself

# TODO perf --> disp2depth is the slow one: particularly the 255 masking in the color map :o

# TODO perf: next biggest perf issue: the trigger concat --> use a ring buffer

import numpy as np
import time
from metavision_sdk_base import EventCD


class SimpleTriggerFinder:
    def __init__(self, projector_fps=60, frame_paused_thresh_us=40, stats=None, callback=None):
        # when there is a time diff between consecutive events of more than this, we assume
        # that the projector is currently not projecting a frame
        self.frame_paused_thresh_us = frame_paused_thresh_us

        self.projector_fps = projector_fps

        self.ev_buf = np.array([], dtype=EventCD)

        self.stats = stats

        self.callback = callback

    def process_events(self, evs):
        # TODO perf use a ring buffer
        # e.g. np-rw-buffer
        # TODO investigate: EventIterator from Metavision just uses a deque in MetaEventBufferProducer
        self.ev_buf = np.append(self.ev_buf, evs.numpy())

        if len(self.ev_buf) < 2:
            return

        # if we have fewer than one frame worth of events, don't bother
        if self.ev_buf["t"][-1] - self.ev_buf["t"][0] < 1e6 / self.projector_fps:
            return

        self.stats.add_metric("#events in buf [k]", len(self.ev_buf) / 1000)
        self.stats.add_metric("event t range in buf [ms]", (self.ev_buf["t"][-1] - self.ev_buf["t"][0]) / 1000)

        with self.stats.measure_time("find trigger"):
            found = self.find_trigger()

        if found:
            self.stats.count_occurrence("trigger found")
        else:
            self.stats.count_occurrence("no trigger found")

    def find_trigger(self):
        """function to get the frame starts and ends from the current buffer."""

        # find the indicies of the events which have a relative long time difference to their respective next event in the buffer

        # TODO perf: this is recomputed a lot
        frame_paused_ev_idx = np.nonzero(np.diff(self.ev_buf["t"]) >= self.frame_paused_thresh_us)[0]

        for prev_idx, next_idx in zip(frame_paused_ev_idx[:-1], frame_paused_ev_idx[1:]):
            # time between event pauses
            diff_t_event_pauses = self.ev_buf["t"][next_idx] - self.ev_buf["t"][prev_idx]

            # if this time is greater than half of the frametime, it
            # is assumend that the two events are the first and last events of the frame

            if diff_t_event_pauses > 1e6 / self.projector_fps / 2:
                # TODO reexamine the +2 and -2
                # The interval is trimmed a a little to ensure that the events are in the frame.
                self.callback(self.ev_buf[prev_idx + 2 : next_idx - 2])

                self.stats.add_metric(
                    "frame len [ms]", (self.ev_buf["t"][next_idx - 2] - self.ev_buf["t"][prev_idx + 2]) / 1000
                )

                self.ev_buf = self.ev_buf[next_idx - 2 :]
                return True

        return False


MIN_EVENTS_PER_FRAME = 1000


class RobustTriggerFinder:
    def __init__(self, projector_fps=60, frame_paused_thresh_us=40, stats=None, callback=None):
        # when there is a time diff between consecutive events of more than this, we assume
        # that the projector is currently not projecting a frame
        self.frame_paused_thresh_us = frame_paused_thresh_us

        self.projector_fps = projector_fps

        self.ev_buf = np.array([], dtype=EventCD)

        self.stats = stats

        self.callback = callback

        self.frame_len_ms = 1e3 / self.projector_fps

        self.should_drop = False

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

        with self.stats.measure_time("find trigger"):
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
