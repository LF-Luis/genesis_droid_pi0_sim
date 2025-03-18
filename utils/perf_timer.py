import time
from contextlib import contextmanager


@contextmanager
def perf_timer(block_name):
    # Performance profiling, just timer for now
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"> > > '{block_name}' took {end - start:.6f} seconds")
