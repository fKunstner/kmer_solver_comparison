import contextlib
import inspect

import tqdm


@contextlib.contextmanager
def redirect_to_tqdm():
    old_print = print

    def new_print(*args, **kwargs):
        tqdm.tqdm.write(" ".join([str(x) for x in args]), **kwargs)

    try:
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


def progressbar(*args, **kwargs):
    with redirect_to_tqdm():
        for x in tqdm.tqdm(*args, **kwargs):
            yield x
