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

        self.last_frame_start_cpu_time_ms = -1
        self.last_frame_start_event_time_ms = -1

        self.acc_cpu_ev_time_diff_ms = -1

        self.should_drop = False

    def reset_buffer(self):
        self.ev_buf = np.array([], dtype=EventCD)

    def drop_frame(self):
        print("Scheduling drop")
        self.should_drop = True

    def process_events(self, evs):
        # TODO perf use a ring buffer
        # e.g. np-rw-buffer
        # TODO investigate: EventIterator from Metavision just uses a deque in MetaEventBufferProducer
        self.ev_buf = np.append(self.ev_buf, evs.numpy())

        # # if time_since_last_frame_ms > self.frame_len_ms and not self.have_dropped:
        # #     if len(self.ev_buf) < 1:
        # #         return

        # #     print(f"time since last frame {time_since_last_frame_ms}")

        #     # # drop one frame worth of events
        #     # buf_start_t = self.ev_buf["t"][0]
        #     # evs_next_frame_or_later = self.ev_buf["t"] >= buf_start_t + self.frame_len_ms * 1000
        #     # if not evs_next_frame_or_later.any():
        #     #     # we don't have a frame worth's events to drop yet, return and request more
        #     #     return

        #     # next_frame_first_event = np.argmax(evs_next_frame_or_later)
        #     # print(f"Dropping {next_frame_first_event} events!")
        #     # self.ev_buf = self.ev_buf[next_frame_first_event:]
        #     self.ev_buf = np.array([], dtype=EventCD)
        #     self.have_dropped = True

        # if time_since_last_frame_ms > self.frame_len_ms and not self.have_dropped:
        #     if len(self.ev_buf) < 1:
        #         return

        #     print(f"time since last frame {time_since_last_frame_ms}")

        if self.should_drop:
            # drop one frame worth of events
            buf_start_t = self.ev_buf["t"][0]
            evs_next_frame_or_later = self.ev_buf["t"] >= buf_start_t + self.frame_len_ms * 1000
            if not evs_next_frame_or_later.any():
                # we don't have a frame worth's events to drop yet, return and request more
                return

            next_frame_first_event = np.argmax(evs_next_frame_or_later)
            # print(f"Dropping {next_frame_first_event} events!")
            self.ev_buf = self.ev_buf[next_frame_first_event:]
            self.should_drop = False

        if len(self.ev_buf) < 2:
            return

        # if we have fewer than one frame worth of events, don't bother
        if self.ev_buf["t"][-1] - self.ev_buf["t"][0] < 1e6 / self.projector_fps:
            return

        self.stats.add_metric("buf #ev [k]", len(self.ev_buf) / 1000)
        self.stats.add_metric("buf ev t [ms]", (self.ev_buf["t"][-1] - self.ev_buf["t"][0]) / 1000)

        # ignore camera time, we don't know how many events are still waiting to be processed

        # if we haven't generated a frame in the last 32 ms, drop

        # if self.last_frame_start_cpu_time_ms != -1 and self.last_frame_start_event_time_ms != -1:
        #     cpu_time_diff_ms = time.perf_counter() * 1000 - self.last_frame_start_cpu_time_ms
        #     ev_time_diff_ms = self.ev_buf["t"][-1] / 1000 - self.last_frame_start_event_time_ms
        #     cpu_ev_diff = cpu_time_diff_ms - ev_time_diff_ms
        #     if cpu_ev_diff > 0:
        #         self.acc_cpu_ev_time_diff_ms += cpu_ev_diff
        #     self.stats.add_metric("cpu diff [ms]", cpu_time_diff_ms)
        #     self.stats.add_metric("ev diff [ms]", ev_time_diff_ms)

        with self.stats.measure_time("find trigger"):
            ev_time = self.find_trigger() / 1000
        if ev_time > 0:
            self.last_frame_start_event_time_ms = ev_time
            self.stats.count_occurrence("trigger ✅")
            self.last_frame_start_cpu_time_ms = time.perf_counter() * 1000
        # else:
        #     self.stats.count_occurrence("trigger ❌")

        # if self.acc_cpu_ev_time_diff_ms > 1e3 / self.projector_fps:
        #     self.stats.count_occurrence("acc > 1 frame")

        # self.stats.add_metric("acc time diff [ms]", self.acc_cpu_ev_time_diff_ms)

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
                if diff_t_event_pauses <= 1e6 / self.projector_fps and next_idx - prev_idx > MIN_EVENTS_PER_FRAME:
                    # TODO reexamine the +2 and -2
                    # The interval is trimmed a a little to ensure that the events are in the frame.
                    self.callback(self.ev_buf[prev_idx + 2 : next_idx - 2])

                    start_time = self.ev_buf["t"][prev_idx + 2]
                    end_time = self.ev_buf["t"][next_idx - 2]

                    self.stats.add_metric("framelen [ms]", (end_time - start_time) / 1000)

                    self.ev_buf = self.ev_buf[next_idx - 2 :]
                    return start_time
                else:
                    # trigger not found correctly, drop these events
                    self.ev_buf = self.ev_buf[next_idx:]
                    return -1

        return -1
