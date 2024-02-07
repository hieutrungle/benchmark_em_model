import signal

DEBUG = False


# Custom exception. In general, it's a good practice.
class TimeoutError(Exception):
    def __init__(self, value="Timed Out"):
        self.value = value

    def __str__(self):
        return repr(self.value)


# This is the decorator itself.
# Read about it here: https://pythonbasics.org/decorators/
# Basically, it's a function that receives another function and a time parameter, i.e. the number of seconds.
# Then, it wraps it so that it raises an error if the function does not
# return a result before `seconds_before_timeout` seconds
def timeout(seconds_before_timeout):
    def decorate(f):
        if DEBUG:
            print("[DEBUG]: Defining decorator handler and the new function")
        if DEBUG:
            print(f"[DEBUG]: Received the following function >> `{f.__name__}`")

        def handler(signum, frame):
            raise TimeoutError()

        def new_f(*args, **kwargs):
            # Verbatim from Python Docs
            # > The signal.signal() function allows defining custom handlers to be executed
            #   when a signal is received.
            if DEBUG:
                print(
                    f"[DEBUG]: in case of ALARM for function `{f.__name__}`, I'll handle it with the... `handler`"
                )
            old = signal.signal(signal.SIGALRM, handler)

            # See https://docs.python.org/3/library/signal.html#signal.alarm
            if DEBUG:
                print(
                    f"[DEBUG]: setting an alarm for {seconds_before_timeout} seconds."
                )
            signal.alarm(seconds_before_timeout)
            try:
                if DEBUG:
                    print(f"[DEBUG]: executing `{f.__name__}`...")
                result = f(*args, **kwargs)
            finally:
                # reinstall the old signal handler
                signal.signal(signal.SIGALRM, old)
                # Cancel alarm.
                # See: https://docs.python.org/3/library/signal.html#signal.alarm
                signal.alarm(0)
            return result

        new_f.__name__ = f.__name__
        return new_f

    return decorate
