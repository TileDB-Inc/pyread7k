import collections
import csv
import functools
import itertools as it
from typing import Iterable, Iterator, Tuple, TypeVar

from .records import FileCatalog


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


def export_catalog(filename: str, file_catalog: FileCatalog):
    """ Write the catalog to a file in csv format """

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([f"file={filename}"])
        writer.writerow(["idx", "record_id", "file_offset", "size"])
        for idx, (type_id, offset, size) in enumerate(
            zip(
                file_catalog.record_types,
                file_catalog.offsets,
                file_catalog.sizes,
            )
        ):
            writer.writerow(str(n) for n in [idx, type_id, offset, size])
