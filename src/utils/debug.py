import sys
import inspect

import torch
import IPython
import numpy as np


def enter_interactive(exit_at_end=False):
    """
    Type `exit` to continue from this point in the code.
    """
    print("="*42)
    print("Entering interactive mode.")
    print("-"*42)
    if exit_at_end:
        print("Type `exit` to exit the simulation.")
    else:
        print("Type `exit` to continue from this point in the code.")

    # Captures caller's local variables and inject them into the interactive namespace
    caller_frame = inspect.currentframe().f_back
    if caller_frame:
        print(f"Injecting caller's local vars...")
        print("="*42)
        caller_locals = caller_frame.f_locals.copy()
        interactive_namespace = caller_locals.copy()
        IPython.embed(user_ns=interactive_namespace)
    else:
        print(f"Not injecting caller's local vars...")
        print("="*42)
        IPython.embed()

    if exit_at_end:
        sys.exit()

def inspect_structure(obj):
    """
    For APIs lacking good documentation.
    """

    print(f"Type: {type(obj)}")

    try:
        print(f"Length: {len(obj)}")
    except TypeError:
        print("Not iterable")
        return

    if isinstance(obj, np.ndarray):
        print(f"dtype: {obj.dtype}")
        print(f"shape: {obj.shape}")
        if obj.size > 0:
            print(f"element type: {type(obj.flat[0])}")

    elif torch and isinstance(obj, torch.Tensor):
        print(f"dtype: {obj.dtype}")
        print(f"shape: {obj.shape}")
        if obj.numel() > 0:
            print(f"element type: {type(obj.flatten()[0].item())}")

    elif isinstance(obj, list):
        if len(obj) > 0:
            print(f"element type: {type(obj[0])}")
            try:
                print(f"nested type: {type(obj[0][0])}")
            except Exception:
                pass
        else:
            print("empty list")
    else:
        print("unsupported type")
