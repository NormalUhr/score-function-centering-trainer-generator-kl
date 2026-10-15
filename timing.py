"""Wall-time printing, based on https://stackoverflow.com/a/69156219."""
from time import perf_counter


class context:
    """Print the wall time of the `with` block under `label`."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = perf_counter()

    def __exit__(self, type, value, traceback):
        elapsed_time = perf_counter() - self.start
        print(f'[timing] {self.label}: {elapsed_time:.2f}s')
