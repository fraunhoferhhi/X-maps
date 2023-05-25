import numpy as np
from metavision_sdk_base import EventCD, EventCDBuffer
from dataclasses import dataclass, field
from typing import Callable, List
from stats_printer import StatsPrinter


MIN_EVENTS_PER_FRAME = 1000


@dataclass
class EventBufferList:
    # TODO we're reinventing the wheel here a little bit
    # replace with some Metavision ring buffer thing?

    _bufs: List[np.ndarray] = field(default_factory=list)

    def append(self, evs: EventCDBuffer):
        np_copy = np.copy(evs.numpy())
        if len(np_copy) > 0:
            self._bufs.append(np_copy)

    def clear(self):
        self._bufs.clear()

    def empty(self):
        return not self._bufs

    def first_ev_time(self):
        if not self._bufs:
            return -1
        return self._bufs[0]["t"][0]

    def last_ev_time(self):
        if not self._bufs:
            return -1
        return self._bufs[-1]["t"][-1]

    def time_span_us(self):
        first_time = self.first_ev_time()
        last_time = self.last_ev_time()

        if first_time < 0 or last_time < 0:
            return -1

        return last_time - first_time

    def num_events(self):
        return sum(len(buf) for buf in self._bufs)

    def drop(self, drop_len_ms):
        drop_until_us = self.first_ev_time() + drop_len_ms * 1000

        have_dropped = False

        while not self.empty() and self.first_ev_time() < drop_until_us:
            self._bufs.pop(0)
            have_dropped = True

        return have_dropped

    def pop_all(self):
        ret = np.concatenate(self._bufs)
        self._bufs.clear()
        return ret

    def push(self, evs):
        # logic intends that trigger finder will push the remaining events here, so it should be empty
        assert self.empty()
        if len(evs):
            self._bufs.append(evs)


@dataclass
class RobustTriggerFinder:
    projector_fps: int
    stats: StatsPrinter
    callback: Callable[[np.ndarray], None]

    frame_paused_thresh_us = 40
    should_drop = False

    _ev_buf: EventBufferList = EventBufferList()

    @property
    def frame_len_ms(self):
        return 1e3 / self.projector_fps

    def reset_buffer(self):
        self._ev_buf.clear()

    def drop_frame(self):
        self.should_drop = True

    def process_events(self, evs):
        self._ev_buf.append(evs)

        if self.should_drop:
            if self._ev_buf.drop(self.frame_len_ms):
                self.stats.count("frames dropped")
                self.should_drop = False
            else:
                # we don't have a frame worth's events to drop yet, return and request more
                return

        if self._ev_buf.empty():
            return

        # if we have fewer than one frame worth of events, don't bother
        if self._ev_buf.time_span_us() < 1e6 / self.projector_fps:
            return

        self.stats.add_metric("evs in buf", self._ev_buf.num_events())

        ev_time = self.find_trigger() / 1000
        if ev_time > 0:
            self.stats.count("trig ✅")
        else:
            self.stats.count("trig ❌")

    def find_trigger(self):
        """function to get the frame starts and ends from the current buffer."""

        evs = self._ev_buf.pop_all()

        # find the indicies of the events which have a relative long time difference to their respective next event in the buffer

        # perf: is this recomputed a lot? - no, not at all, usually there's a whole frame worth of events here, so it is not
        with self.stats.measure_time("find pauses"):
            frame_paused_ev_idx = np.nonzero(np.diff(evs["t"]) >= self.frame_paused_thresh_us)[0]

        # offline: avg 4
        # online: avg 2
        # self.stats.add_metric("frame pauses", len(frame_paused_ev_idx))

        for prev_idx, next_idx in zip(frame_paused_ev_idx[:-1], frame_paused_ev_idx[1:]):
            # time between event pauses
            diff_t_event_pauses = evs["t"][next_idx] - evs["t"][prev_idx]

            # if this time is greater than half of the frametime, it
            # is assumend that the two events are the first and last events of the frame

            if diff_t_event_pauses > 1e6 / self.projector_fps / 2:
                if diff_t_event_pauses <= 1e6 / self.projector_fps and next_idx - prev_idx > MIN_EVENTS_PER_FRAME:
                    # TODO reexamine the +2 and -2
                    # The interval is trimmed a a little to ensure that the events are in the frame.
                    self.callback(evs[prev_idx + 2 : next_idx - 2])

                    start_time = evs["t"][prev_idx + 2]
                    end_time = evs["t"][next_idx - 2]

                    self.stats.add_metric("frame len [ms]", (end_time - start_time) / 1000)

                    self._ev_buf.push(evs[next_idx - 2 :])
                    return start_time
                else:
                    # trigger not found correctly, drop these events
                    self._ev_buf.push(evs[next_idx:])
                    return -1

        return -1
