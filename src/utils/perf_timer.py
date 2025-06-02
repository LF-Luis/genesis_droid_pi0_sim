import time
from contextlib import contextmanager


@contextmanager
def perf_timer(block_name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{block_name} | {end - start:.6f} seconds")


def start_timer():
    return time.perf_counter()

def end_timer(start_time, log):
    end_time = time.perf_counter()
    print(f"{log} | {end_time - start_time:.6f} seconds")
