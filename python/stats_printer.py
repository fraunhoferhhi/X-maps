from time import time
from collections import defaultdict
from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional


def human_readable_time(elapsed_ns):
    if abs(elapsed_ns) > 1e9:
        return f"{elapsed_ns / 1e9:.2f} s"
    elif abs(elapsed_ns) > 1e6:
        return f"{elapsed_ns / 1e6:.2f} ms"
    elif abs(elapsed_ns) > 1e3:
        return f"{elapsed_ns / 1e3:.2f} us"
    else:
        return f"{elapsed_ns:.0f} ns"


def human_readable_qty(qty):
    if abs(qty) > 1e9:
        return f"{qty / 1e9:.2f} G"
    elif abs(qty) > 1e6:
        return f"{qty / 1e6:.2f} M"
    elif abs(qty) > 1e3:
        return f"{qty / 1e3:.2f} k"
    else:
        return f"{qty}"
    
def human_readable_ev_qty_per_second(qty, elapsed_ns):
    return f"{human_readable_qty(qty * 1e9 / elapsed_ns)}evps"

class StatsPrinter:
    """Utility class to print statistics about the execution of the code.

    Use `count_occurrence`, `add_metric` and `measure_time` to add statistics.

    Call `print_stats` regularly to print the statistics. It will only print
    every `print_every_ms` milliseconds, otherwise return early.
    """

    def __init__(self, print_every_ms=1000):
        self.print_every_ms = print_every_ms

        self.last_print_time = time.perf_counter_ns()
        self.prev_line_len = 0

        self.occurences = defaultdict(int)

        self.measures = defaultdict(float)
        self.measure_counter = defaultdict(int)

        self.timers = dict()
        self.time_measures = defaultdict(float)
        self.time_measure_counter = defaultdict(int)
        
        self.processed_events = 0
        self.global_processed_events = 0
        
        self.global_timer = SingleTimer("Total runtime")
        
    def __del__(self):
        print("")
        self.global_timer.stop()
        print(f"Total events processed: {human_readable_qty(self.global_processed_events)}")
        print(f"Event throughput: {human_readable_ev_qty_per_second(self.global_processed_events, self.global_timer.elapsed_ns())}")
        
    def count_processed_events(self, num_events):
        if not self.global_timer.is_running():
            self.global_timer.start()
        self.processed_events += num_events

    def count_occurrence(self, key):
        """Count occurrences of a certain type (e.g. "trigger found")"""
        self.occurences[key] += 1

    def add_metric(self, key, val):
        """Mesure a certain metric (e.g. "frame len [ms]")"""
        self.measures[key] += val
        self.measure_counter[key] += 1

    def measure_time(self, key):
        """Time the execution of the code in the context manager
        Usage: with stats.measure_time("my key"):
                     do_something()
        """
        return StatsTimer(stats_printer=self, key=key)
    
    def add_time_measure_ns(self, key, elapsed_ns):
        self.time_measures[key] += elapsed_ns
        self.time_measure_counter[key] += 1
    
    def print_stats(self):
        # print stats once a second
        elapsed_ns = time.perf_counter_ns() - self.last_print_time
        if elapsed_ns < self.print_every_ms * 1e6:
            return

        print("\r", end="")
        print(" " * self.prev_line_len, end="")
        print("\r", end="")

        line = f"{human_readable_time(elapsed_ns)}: "

        line += f"evs: {human_readable_qty(self.processed_events)} @ {human_readable_ev_qty_per_second(self.processed_events, elapsed_ns)} | "

        for k, v in sorted(self.time_measures.items()):
            avg_time_ns = v / self.time_measure_counter[k]
            line += f"{k} ({self.time_measure_counter[k]}): {human_readable_time(avg_time_ns)} | "

        for k, v in sorted(self.measures.items()):
            line += f"{k} ({self.measure_counter[k]}): {v / self.measure_counter[k]:.2f} | "

        for k, v in sorted(self.occurences.items()):
            line += f"{k}: {v} | "
            self.occurences[k] = 0

        print(line, end="")
        self.prev_line_len = len(line)

        self.global_processed_events += self.processed_events
        self.processed_events = 0

        self.measures.clear()
        self.measure_counter.clear()

        self.time_measures.clear()
        self.time_measure_counter.clear()

        self.last_print_time = time.perf_counter_ns()


@dataclass
class StatsTimer:
    """Timer to be used in a context manager to measure the time of a code block.
    Time is measured for multiple runs and averaged.
    Use this from StatsPrinter.measure_time(your_key)"""

    stats_printer: StatsPrinter
    key: str
    start_time: int = field(init=False)

    def __enter__(self):
        self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, *exc_info):
        elapsed_time_ns = time.perf_counter_ns() - self.start_time
        self.stats_printer.add_time_measure_ns(self.key, elapsed_time_ns)
        return False


@dataclass
class SingleTimer:
    """Timer to be used in a context manager to measure the time of a code block.
    Prints the time after the block is finished."""

    message: str
    start_time: int = field(init=False)
    started: bool = False
    
    def start(self):
        if self.started:
            raise Exception("Timer already started")
        self.start_time = time.perf_counter_ns()
        self.started = True
        return self
        
    def stop(self):
        if not self.started:
            raise Exception("Timer not started")
        self.elapsed_time_ns = time.perf_counter_ns() - self.start_time
        print(f"{self.message}: {human_readable_time(self.elapsed_time_ns)}")
        self.started = False
        
    def is_running(self):
        return self.started

    def elapsed_ns(self):
        if self.started:
            return time.perf_counter_ns() - self.start_time
        else:
            return self.elapsed_time_ns

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()
        return False

