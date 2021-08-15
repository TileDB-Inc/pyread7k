import collections
import functools
import itertools as it
from typing import Iterable, Iterator, Tuple, TypeVar

T = TypeVar("T")


def window(seq: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    """Return a sliding window of width n over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    iterator = iter(seq)
    q = collections.deque(it.islice(iterator, n), maxlen=n)
    if len(q) == n:
        yield tuple(q)
    for elem in iterator:
        q.append(elem)
        yield tuple(q)


def cached_property(func):
    """
    Fix functools.cached_property to preserve docstrings and name.
    Note that it does not properly preserve type hints!
    """
    return functools.update_wrapper(functools.cached_property(func), func)
