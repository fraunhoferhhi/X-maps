import numpy as np
from dataclasses import dataclass
from collections import deque


class FrameEventFilter:
    def filter_events(self, events):
        raise NotImplementedError()


class NoFilter(FrameEventFilter):
    def filter_events(self, events):
        return events

    def __str__(self):
        return "NoFilter"


class LastEventPerXYFilter(FrameEventFilter):
    def filter_events(self, events):
        events = events[events["p"] == 1]

        # TODO perf not optimized

        event_map = np.zeros((events["y"].max() + 1, events["x"].max() + 1), dtype=np.int32)
        event_mask = np.zeros_like(event_map, dtype=bool)
        event_map[events["y"], events["x"]] = events["t"]
        event_mask[events["y"], events["x"]] = True

        event_coords = np.indices(event_mask.shape, dtype=np.int16)

        num_events = event_mask.sum()
        filtered_events = np.zeros(num_events, dtype=events.dtype)
        filtered_events["t"] = event_map[event_mask]
        filtered_events["x"] = event_coords[1][event_mask]
        filtered_events["y"] = event_coords[0][event_mask]
        filtered_events["p"] = True

        return filtered_events

    def __str__(self):
        return "LastEventPerXYFilter"


class FirstEventPerXYFilter(FrameEventFilter):
    def filter_events(self, events):
        events = events[events["p"] == 1]

        # TODO perf not optimized:
        event_map = np.zeros((events["y"].max() + 1, events["x"].max() + 1), dtype=np.int32)
        event_mask = np.zeros_like(event_map, dtype=bool)
        event_map[events["y"][::-1], events["x"][::-1]] = events["t"][::-1]
        event_mask[events["y"][::-1], events["x"][::-1]] = True

        event_coords = np.indices(event_mask.shape, dtype=np.int16)

        num_events = event_mask.sum()
        filtered_events = np.zeros(num_events, dtype=events.dtype)
        filtered_events["t"] = event_map[event_mask]
        filtered_events["x"] = event_coords[1][event_mask]
        filtered_events["y"] = event_coords[0][event_mask]
        filtered_events["p"] = True

        return filtered_events

    def __str__(self):
        return "FirstEventPerXYFilter"


class FirstEventPerYTFilter(FrameEventFilter):
    def filter_events(self, events, xp_i16):
        events = events[events["p"] == 1]

        # TODO perf not optimized:
        event_map_x = np.zeros((events["y"].max() + 1, xp_i16.max() + 1), dtype=np.int32)
        event_map_t = np.zeros_like(event_map_x, dtype=np.int32)
        event_mask = np.zeros_like(event_map_x, dtype=np.bool)

        event_map_x[events["y"][::-1], xp_i16[::-1]] = events["x"][::-1]

        # TODO note this is using t, but xp_i16 was calculated and rounded from t
        # perhaps recompute t from xp_i16?, from coords of event_map_t?
        event_map_t[events["y"][::-1], xp_i16[::-1]] = events["t"][::-1]
        event_mask[events["y"][::-1], xp_i16[::-1]] = True

        event_coords = np.indices(event_mask.shape, dtype=np.int16)

        num_events = event_mask.sum()
        filtered_events = np.zeros(num_events, dtype=events.dtype)
        filtered_events["t"] = event_map_t[event_mask]
        # filtered_events["t"] = event_coords[1][event_mask] * SOMETHING
        filtered_events["x"] = event_map_x[event_mask]
        filtered_events["y"] = event_coords[0][event_mask]
        filtered_events["p"] = True

        return filtered_events

    def __str__(self):
        return "FirstEventPerYTFilter"


class MeanFirstLastEventPerXYFilter(FrameEventFilter):
    def filter_events(self, events):
        events = events[events["p"] == 1]

        # TODO perf not optimized
        first_event_map = np.zeros((events["y"].max() + 1, events["x"].max() + 1), dtype=np.int32)
        event_mask = np.zeros_like(first_event_map, dtype=bool)
        first_event_map[events["y"], events["x"]] = events["t"]
        event_mask[events["y"], events["x"]] = True

        last_event_map = np.zeros((events["y"].max() + 1, events["x"].max() + 1), dtype=np.int32)
        last_event_map[events["y"][::-1], events["x"][::-1]] = events["t"][::-1]

        event_coords = np.indices(event_mask.shape, dtype=np.int16)

        num_events = event_mask.sum()

        filtered_events = np.zeros(num_events, dtype=events.dtype)
        filtered_events["t"] = (first_event_map[event_mask] + last_event_map[event_mask]) // 2
        filtered_events["x"] = event_coords[1][event_mask]
        filtered_events["y"] = event_coords[0][event_mask]
        filtered_events["p"] = True

        # import pandas as pd

        # row_100 = first_event_map[100, 265:465]
        # r100s = pd.Series(row_100)

        # row_200 = first_event_map[200, 265:465]
        # r200s = pd.Series(row_200)
        # r200s.plot()

        return filtered_events

    def __str__(self):
        return "MeanFirstLastEventPerXYFilter"


@dataclass
class FrameEventFilterProcessor:
    filters = deque(
        (
            NoFilter(),
            # FirstEventPerYTFilter(),
            FirstEventPerXYFilter(),
            LastEventPerXYFilter(),
            MeanFirstLastEventPerXYFilter(),
        )
    )

    def selected_filter(self):
        return self.filters[0]

    def filter_events(self, evs):
        return self.selected_filter().filter_events(evs)

    def select_next_filter(self):
        self.filters.rotate(-1)
        return self.selected_filter()
