import functools
import sys
import time
import traceback

from utils.slack_util import post_message_to_slack


def sample_decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value

    return wrapper_decorator


def slack_alert(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        signature = build_signature(args, kwargs)
        try:
            v = func(*args, **kwargs)
            post_message_to_slack(f"Your task finished fine: {func.__name__}({signature})")
            print("Sent success message to slack")
            return v
        except KeyboardInterrupt as e:
            print(traceback.format_exc())  # do not send any message when we kill it purposefully
            raise e
        except:
            t, v, tb = sys.exc_info()
            traceback_msg = traceback.format_exc()
            print(traceback_msg)
            post_message_to_slack(f"Something went wrong with our task: {func.__name__}({signature})\n{traceback_msg}")
            print("Sent failure notification to slack")

            raise t(v).with_traceback(tb)

    return wrapper_decorator


def debug(func):
    """Print the function signature and return value. This should only be used when there is no debugger available!"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        signature = build_signature(args, kwargs)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")  # 4
        return value

    return wrapper_debug


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def build_signature(args, kwargs):
    args_repr = [repr(a) for a in args]  # 1
    kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
    signature = ", ".join(args_repr + kwargs_repr)  # 3
    return signature
