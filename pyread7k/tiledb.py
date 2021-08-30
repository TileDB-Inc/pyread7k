"""
Store 7K records as a TileDB array
"""
import sys
from datetime import datetime
from io import BytesIO
from itertools import chain, islice
from typing import (
    Any,
    BinaryIO,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import tiledb

import pyread7k

T = TypeVar("T")


def chunked(iterable: Iterable[T], n: int) -> Iterable[List[T]]:
    """"Break iterable into lists of length n"""
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, n)), [])


def iter_record_bytes(reader: pyread7k.S7KFileReader) -> Iterator[bytes]:
    """Generate unparsed records as bytes for every record of this reader"""
    # include the FileCatalog offset and size too
    offsets = chain(reader.file_catalog.offsets, [reader.file_header.catalog_offset])
    sizes = chain(reader.file_catalog.sizes, [reader.file_header.catalog_size])
    seek = reader._fhandle.seek
    read = reader._fhandle.read
    for offset, size in zip(offsets, sizes):
        seek(offset)
        yield read(size)


def to_tiledb(
    array_uri: str,
    filename: str,
    mode: str = "ingest",
    time_tile: int = 5000,
    tile_capacity: int = 500,
    batch_size: int = 5000,
    config: Mapping[str, Any] = {
        "sm.consolidation.steps": 3,
        "sm.consolidation.mode": "fragment_meta",
    },
) -> None:
    """Create TileDB array at given URI from a dataset

    :param array_uri: URI for new TileDB array
    :param filename: Path to the s7k file to ingest
    :param mode: Creation mode, one of 'ingest' (default), 'schema_only', 'append'
    :param time_tile: TileDB tile size for time
    :param tile_capacity: TileDB sparse fragment capacity
    :param batch_size: Ingest in batches of this size
    """
    reader = pyread7k.S7KFileReader(filename)
    file_catalog = reader.file_catalog
    # include the FileCatalog offset, record_type and time too
    record_types = (*file_catalog.record_types, 7300)
    offsets = (*file_catalog.offsets, reader.file_header.catalog_offset)
    times = (*file_catalog.times, file_catalog.frame.time)

    with tiledb.scope_ctx(config):
        if mode in ("ingest", "schema_only"):
            schema = tiledb.ArraySchema(
                sparse=True,
                domain=tiledb.Domain(
                    tiledb.Dim(
                        name="offset",
                        domain=(min(offsets), max(offsets)),
                        tile=1_000_000,
                        dtype=np.dtype("uint64"),
                    ),
                    tiledb.Dim(
                        name="record_type_id",
                        domain=(min(record_types), max(record_types)),
                        tile=1,
                        dtype=np.dtype("uint16"),
                    ),
                    tiledb.Dim(
                        name="time",
                        domain=(np.datetime64(min(times)), np.datetime64(max(times))),
                        tile=time_tile,
                        dtype=np.dtype("datetime64[us]"),
                    ),
                ),
                attrs=[
                    tiledb.Attr(
                        name="bytes",
                        dtype=np.dtype("bytes"),
                        filters=[tiledb.ZstdFilter()],
                    ),
                ],
                capacity=tile_capacity,
                coords_filters=[tiledb.ZstdFilter()],
            )
            tiledb.Array.create(array_uri, schema)
            if mode == "schema_only":
                return

        with tiledb.open(array_uri, mode="w") as arr:
            tuples = zip(offsets, record_types, times, iter_record_bytes(reader))
            for chunk in chunked(tuples, batch_size):
                os, rs, ts, bs = zip(*chunk)
                # XXX: writing bytes directly trims off trailing NUL chars
                # wrapping it in an numpy array of object dtype preserves all the bytes
                arr[os, rs, ts] = np.array(bs, dtype=np.dtype("O"))

        tiledb.consolidate(array_uri)


class S7KTileDBReader(pyread7k.S7KReader):
    """Reader class for s7k TileDB arrays"""

    def __init__(self, arr: tiledb.Array):
        self._arr = arr

    def read_record_bytes(self, record_type: int) -> Optional[bytes]:
        offset = self.get_first_offset(record_type, 0, sys.maxsize)
        return self._get_stream_for_read(offset).read() if offset is not None else None

    def iter_timedelta_record_bytes(
        self, *record_types: int
    ) -> Iterator[Tuple[Optional[float], bytes]]:
        query = self._arr.query(dims=("time",), return_incomplete=True)
        for rdict in query.multi_index[:, list(record_types)]:
            time_deltas = (rdict["time"][1:] - rdict["time"][:-1]).tolist()
            for td, record_bytes in zip(time_deltas, rdict["bytes"][:-1]):
                yield td.total_seconds(), record_bytes
            yield None, rdict["bytes"][-1]

    def read_records_during_ping(
        self,
        record_type: int,
        ping_start: datetime,
        ping_end: datetime,
        offset_hint: int,
    ) -> List[pyread7k.records.BaseRecord]:
        records = []
        read = pyread7k.record(record_type).read
        query = self._arr.query(dims=(), return_incomplete=True)
        time_slice = slice(np.datetime64(ping_start), np.datetime64(ping_end))
        for rdict in query.multi_index[:, record_type, time_slice]:
            for record_bytes in rdict["bytes"]:
                records.append(read(BytesIO(record_bytes)))
        return records

    def _iter_offset_records(
        self, record_type: int
    ) -> Iterator[Tuple[int, pyread7k.records.BaseRecord]]:
        read = pyread7k.record(record_type).read
        query = self._arr.query(dims=("offset",), return_incomplete=True)
        for rdict in query.multi_index[:, record_type]:
            for offset, record_bytes in zip(rdict["offset"], rdict["bytes"]):
                yield offset, read(BytesIO(record_bytes))

    def _get_stream_for_read(self, offset: int) -> BinaryIO:
        query = self._arr.query(dims=())
        record_bytes = query.multi_index[offset]["bytes"]
        assert len(record_bytes) == 1
        return BytesIO(record_bytes[0])


if __name__ == "__main__":
    import shutil

    inpath, outpath = sys.argv[1:]
    shutil.rmtree(outpath, ignore_errors=True)
    to_tiledb(outpath, inpath)
