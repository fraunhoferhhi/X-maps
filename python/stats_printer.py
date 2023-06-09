from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict
import shutil
import time


def max_cols():
    return shutil.get_terminal_size(fallback=(160, 40)).columns


def human_readable_time(elapsed_ns):
    if abs(elapsed_ns) > 60e9:
        return f"{elapsed_ns / 60e9:6.2f} min"
    elif abs(elapsed_ns) > 1e9:
        return f"{elapsed_ns / 1e9:6.2f} s  "
    elif abs(elapsed_ns) > 1e6:
        return f"{elapsed_ns / 1e6:6.2f} ms "
    elif abs(elapsed_ns) > 1e3:
        return f"{elapsed_ns / 1e3:6.2f} us "
    else:
        return f"{elapsed_ns:6.2f} ns "


def human_readable_qty(qty):
    if abs(qty) > 1e9:
        return f"{qty / 1e9:6.2f}G"
    elif abs(qty) > 1e6:
        return f"{qty / 1e6:6.2f}M"
    elif abs(qty) > 1e3:
        return f"{qty / 1e3:6.2f}k"
    else:
        if type(qty) == int:
            return f"{qty:6d} "
        else:
            return f"{qty:6.2f} "


def human_readable_qty_per_second(qty, elapsed_ns):
    return f"{human_readable_qty(qty * 1e9 / elapsed_ns)}"


@dataclass
class Occurences:
    occurences: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def count(self, name: str, qty: int = 1):
        self.occurences[name] += qty

    def print_total(self):
        lines = 1
        str = "total count "
        for name, occ in sorted(self.occurences.items()):
            append = f"#{name}: {human_readable_qty(occ)} | "
            if len(str + append) > max_cols():
                print(str)
                str = "           "
                lines += 1
            str += append

        print(str)
        return lines

    def print_avg(self, elapsed_ns):
        str = "avg per sec "
        lines = 1
        for name, occ in sorted(self.occurences.items()):
            append = f"#{name}: {human_readable_qty_per_second(occ, elapsed_ns)} | "
            if len(str + append) > max_cols():
                print(str)
                str = "             "
                lines += 1
            str += append

        print(str)
        return lines

    def reset(self):
        for name in self.occurences.keys():
            self.occurences[name] = 0

    def __getitem__(self, name):
        return self.occurences[name]


@dataclass
class Quantities:
    qties: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    qty_counter: Occurences = field(default_factory=Occurences)
    fmt: Callable[[float], str] = field(default_factory=lambda: human_readable_qty)

    def add(self, name: str, qty: float):
        self.qties[name] += qty
        self.qty_counter.count(name)

    def print_avg(self):
        str = "avg "
        lines = 1
        for name, qty in sorted(self.qties.items()):
            count = max(self.qty_counter[name], 1)
            append = f"{name}: {self.fmt(qty / count)} | "

            if len(str + append) > max_cols():
                print(str)
                str = "    "
                lines += 1

            str += append
        print(str)
        return lines

    def reset(self):
        for name in self.qties.keys():
            self.qties[name] = 0
        self.qty_counter.reset()


@dataclass
class TimeMeasures(Quantities):
    fmt: Callable[[float], str] = field(default_factory=lambda: human_readable_time)


@dataclass
class Stats:
    occurences: Occurences = field(default_factory=Occurences)
    qties: Quantities = field(default_factory=Quantities)
    time_measures: TimeMeasures = field(default_factory=TimeMeasures)
    timers: Dict[str, "StatsTimer"] = field(default_factory=dict)
    start_time_ns: int = field(default_factory=lambda: time.perf_counter_ns())

    def count(self, name: str, qty: int):
        self.occurences.count(name, qty)

    def add(self, name: str, qty: float):
        self.qties.add(name, qty)

    def add_time_measure_ns(self, name: str, elapsed_ns: float):
        self.time_measures.add(name, elapsed_ns)

    def elapsed_ns(self):
        return time.perf_counter_ns() - self.start_time_ns

    def print_total_occurrences(self):
        return self.occurences.print_total()

    def print_avg_occurrences(self, elapsed_ns):
        return self.occurences.print_avg(elapsed_ns)

    def print_avg_qties(self):
        return self.qties.print_avg()

    def print_avg_time_measures(self):
        return self.time_measures.print_avg()

    def reset(self):
        self.occurences.reset()
        self.qties.reset()
        self.time_measures.reset()
        self.start_time_ns = time.perf_counter_ns()


@dataclass
class StatsPrinter:
    """Utility class to print statistics about the execution of the code.

    Use `count_occurrence`, `add_metric` and `measure_time` to add statistics.

    Call `print_stats` regularly to print the statistics. It will only print
    every `print_every_ms` milliseconds, otherwise return early.
    """

    print_every_ms: int = 1000

    have_printed: bool = False

    should_print: bool = True

    printed_lines: int = 0

    local_stats: Stats = Stats()
    global_stats: Stats = Stats()

    def count(self, key, qty=1):
        """Count occurrences of a certain type (e.g. "trigger found")"""
        self.local_stats.count(key, qty)
        self.global_stats.count(key, qty)

    def add_metric(self, key, val):
        """Mesure a certain metric (e.g. "frame len [ms]")"""
        self.local_stats.add(key, val)
        self.global_stats.add(key, val)

    def measure_time(self, key):
        """Time the execution of the code in the context manager
        Usage: with stats.measure_time("my key"):
                     do_something()
        """
        return StatsTimer(stats_printer=self, key=key)

    def add_time_measure_ns(self, key, elapsed_ns):
        self.local_stats.add_time_measure_ns(key, elapsed_ns)
        self.global_stats.add_time_measure_ns(key, elapsed_ns)

    def print_stats_if_needed(self):
        if self.local_stats.elapsed_ns() >= self.print_every_ms * 1e6:
            self.print_stats()

    def clear_printed_lines(self):
        if self.have_printed and self.printed_lines > 0:
            # Move cursor up by 11 lines
            print(f"\033[{self.printed_lines}A", end="")
            # Clear the screen from cursor to end
            print("\033[J", end="")

        self.printed_lines = 0

    def log(self, msg):
        self.clear_printed_lines()
        print(msg)

    def toggle_silence(self):
        self.should_print = not self.should_print

    def start_time_ns(self) -> int:
        return self.global_stats.start_time_ns

    def reset(self):
        self.local_stats.reset()
        self.global_stats.reset()

    def print_stats(self):
        if not self.should_print:
            return

        self.clear_printed_lines()

        red = "\033[31m"
        green = "\033[32m"
        blue = "\033[34m"
        magenta = "\033[35m"
        reset_color = "\033[0m"

        local_avg_color = green
        global_avg_color = blue
        global_tot_color = red

        print(
            f"{local_avg_color}Local stats over  {human_readable_time(self.local_stats.elapsed_ns())} {reset_color}- ",
            end="",
        )
        print(f"{global_avg_color}global stats over {human_readable_time(self.global_stats.elapsed_ns())}")
        self.printed_lines += 1

        print()
        self.printed_lines += 1

        print(f"{local_avg_color}", end="")
        self.printed_lines += self.local_stats.print_avg_occurrences(self.local_stats.elapsed_ns())
        print(f"{global_avg_color}", end="")
        self.printed_lines += self.global_stats.print_avg_occurrences(self.global_stats.elapsed_ns())
        print(f"{global_tot_color}", end="")
        self.printed_lines += self.global_stats.print_total_occurrences()

        print()
        self.printed_lines += 1

        print(f"{local_avg_color}", end="")
        self.printed_lines += self.local_stats.print_avg_qties()
        print(f"{global_avg_color}", end="")
        self.printed_lines += self.global_stats.print_avg_qties()

        print()
        self.printed_lines += 1

        print(f"{local_avg_color}", end="")
        self.printed_lines += self.local_stats.print_avg_time_measures()
        print(f"{global_avg_color}", end="")
        self.printed_lines += self.global_stats.print_avg_time_measures()

        print(reset_color, end="")

        self.local_stats.reset()

        self.last_print_time = time.perf_counter_ns()

        self.have_printed = True


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
