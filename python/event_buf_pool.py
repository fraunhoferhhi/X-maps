from typing import List
from dataclasses import dataclass, field
from metavision_sdk_core import PolarityFilterAlgorithm
from metavision_sdk_base import EventCDBuffer


@dataclass
class EventBufPool:
    bufs: List[EventCDBuffer] = field(default_factory=list)

    def get_buf(self):
        if not self.bufs:
            return PolarityFilterAlgorithm.get_empty_output_buffer()
        return self.bufs.pop()

    def return_buf(self, buf):
        self.bufs.append(buf)
