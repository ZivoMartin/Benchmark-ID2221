"""Best-effort process-tree RSS memory sampler."""

import os
import threading

import psutil


class MemorySampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0
        self.baseline = 0
        self.delta = 0

        self.stop = threading.Event()
        self.thread = None
        self.parent = psutil.Process(os.getpid())

    def _rss(self):
        total = 0

        processes = [self.parent]
        try:
            processes.extend(self.parent.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        for proc in processes:
            try:
                total += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return total

    def _sample(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self._rss())
            self.stop.wait(self.interval)

    def __enter__(self):
        self.stop.clear()

        self.baseline = self._rss()
        self.peak = self.baseline

        self.thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )
        self.thread.start()

        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self._rss())

        self.stop.set()
        self.thread.join()

        self.delta = max(0, self.peak - self.baseline)
