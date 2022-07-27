import numpy as np
import time

class tracker():
    """
    Creates an array, tracks next unfilled entry & runtime, holds tracked name for plotting
    """

    def __init__(self, entry_type='R2X', track_runtime=False):
        self.metric = entry_type
        self.track_runtime = track_runtime

    def begin(self):
        """ Must run to track runtime """
        self.start = time.time()

    def first_entry(self, tFac):
        self.array = np.full((1, 1), 1 - tFac.R2X)
        if self.track_runtime:
            self.time_array = np.full((1, 1), time.time() - self.start)

    def update(self, tFac):
        self.array = np.append(self.array, 1 - tFac.R2X)
        if self.track_runtime:
            self.time_array = np.append(self.time_array, time.time() - self.start)

    def plot_iteration(self, ax):
        ax.plot(range(1, self.array.size + 1), self.array)
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((0, self.array.size))
        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.metric)

    def plot_runtime(self, ax):
        assert self.track_runtime
        self.time_array
        ax.plot(self.time_array, self.array)
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((0, np.max(self.time_array) * 1.2))
        ax.set_xlabel('Runtime')
        ax.set_ylabel(self.metric)
