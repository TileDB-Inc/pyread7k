import collections
import csv
import functools
import io
import itertools as it
from typing import Iterable, Iterator, Tuple, TypeVar

from . import records
from .records import FileCatalog, FileHeader
from .records import record as _record

__all__ = [
    "read_file_header",
    "read_file_catalog",
    "get_record_offsets",
    "get_record_count",
    "gen_records",
    "read_records",
    "export_catalog",
]


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


def read_file_header(source: io.RawIOBase) -> FileHeader:
    """ Read the file header 7200 record """
    return _record(7200).read(source)


def read_file_catalog(source: io.RawIOBase, file_header: FileHeader) -> FileCatalog:
    """ Read the file catalog 7300 record """
    source.seek(file_header.catalog_offset)
    file_catalog: FileCatalog = _record(7300).read(source)
    return file_catalog


def get_record_offsets(type_id: int, file_catalog: FileCatalog) -> tuple:
    """ Get offsets to all records of given type_id from the catalog """

    cat_zip = zip(file_catalog.offsets, file_catalog.record_types)

    return tuple(offset for offset, _type_id in cat_zip if _type_id == type_id)


def get_record_count(type_id: int, file_catalog: FileCatalog) -> int:
    """ Count number of records of given type in the catalog """
    return len(get_record_offsets(type_id, file_catalog))


def gen_records(
    type_id: int,
    source: io.RawIOBase,
    file_catalog: FileCatalog,
    *,
    first_idx=0,
    count=None,
):
    """ Generator reading records of the given type from the file """
    start_offset = source.tell()
    cat_offsets = get_record_offsets(type_id, file_catalog)
    if first_idx > 0:
        cat_offsets = cat_offsets[first_idx:]

    for idx, offset in enumerate(cat_offsets):
        if count is not None and idx >= count:
            break
        source.seek(offset)
        data = _record(type_id).read(source)
        source.seek(start_offset)  # reset source
        yield data


def read_records(
    type_id: int,
    source: io.RawIOBase,
    file_catalog: FileCatalog,
    *,
    first_idx=0,
    count=None,
) -> records.BaseRecord:
    """ Read all records of the given type from the file """

    return tuple(
        gen_records(type_id, source, file_catalog, first_idx=first_idx, count=count)
    )


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
