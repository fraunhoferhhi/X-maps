from time import time
from collections import defaultdict
from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional


def human_readable_time(elapsed_ns):
    if elapsed_ns > 1e9:
        return f"{elapsed_ns / 1e9:.2f} s"
    elif elapsed_ns > 1e6:
        return f"{elapsed_ns / 1e6:.2f} ms"
    elif elapsed_ns > 1e3:
        return f"{elapsed_ns / 1e3:.2f} us"
    else:
        return f"{elapsed_ns} ns"


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

        self.stats = defaultdict(int)

        self.measures = defaultdict(float)
        self.measure_counter = defaultdict(int)

        self.timers = dict()
        self.time_measures = defaultdict(float)
        self.time_measure_counter = defaultdict(int)

    def count_occurrence(self, key):
        """Count occurrences of a certain type (e.g. "trigger found")"""
        self.stats[key] += 1

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

    def print_stats(self):
        if time.perf_counter_ns() - self.last_print_time >= self.print_every_ms * 1e6:
            print("\r", end="")
            print(" " * self.prev_line_len, end="")
            print("\r", end="")

            line = ""

            for k, v in sorted(self.time_measures.items()):
                avg_time_ns = v / self.time_measure_counter[k]
                line += f"{k} ({self.time_measure_counter[k]}): {human_readable_time(avg_time_ns)} | "

            for k, v in sorted(self.measures.items()):
                line += f"{k} ({self.measure_counter[k]}): {v / self.measure_counter[k]:.2f} | "

            for k, v in sorted(self.stats.items()):
                line += f"{k}: {v} | "
                self.stats[k] = 0

            print(line, end="")
            self.prev_line_len = len(line)

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
        elapsed_time = time.perf_counter_ns() - self.start_time

        self.stats_printer.time_measures[self.key] += elapsed_time
        self.stats_printer.time_measure_counter[self.key] += 1

        return False


@dataclass
class SingleTimer:
    """Timer to be used in a context manager to measure the time of a code block.
    Prints the time after the block is finished."""

    message: str
    start_time: int = field(init=False)

    def __enter__(self):
        self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, *exc_info):
        elapsed_time = time.perf_counter_ns() - self.start_time

        print(f"{self.message}: {human_readable_time(elapsed_time)}")

        return False
