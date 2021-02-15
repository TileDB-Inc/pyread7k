import csv
import io

from ._datarecord import DataParts as _DataParts
from ._datarecord import record as _record

__all__ = [
    "read_file_header",
    "read_file_catalog",
    "get_record_offsets",
    "get_record_count",
    "records",
    "read_records",
    "export_catalog",
]


def read_file_header(source: io.RawIOBase) -> _DataParts:
    return _record(7200).read(source)


def read_file_catalog(source: io.RawIOBase, file_header: _DataParts) -> _DataParts:
    source.seek(file_header.od["catalog_offset"])
    file_catalog: _DataParts = _record(7300).read(source)
    return file_catalog


def get_record_offsets(type_id: int, file_catalog: _DataParts) -> tuple:

    cat_zip = zip(file_catalog.rd["offset"], file_catalog.rd["record_type_id"])

    return tuple(offset for offset, _type_id in cat_zip if _type_id == type_id)


def get_record_count(type_id: int, file_catalog: _DataParts) -> int:
    return len(get_record_offsets(type_id, file_catalog))


def records(
    type_id: int,
    source: io.RawIOBase,
    file_catalog: _DataParts,
    *,
    first_idx=0,
    count=None,
):

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
    file_catalog: _DataParts,
    *,
    first_idx=0,
    count=None,
) -> _DataParts:

    return tuple(
        records(type_id, source, file_catalog, first_idx=first_idx, count=count)
    )


def export_catalog(filename: str, file_catalog: _DataParts):

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([f"file={filename}"])
        writer.writerow(["idx", "record_id", "file_offset", "size"])
        for idx, (type_id, offset, size) in enumerate(
            zip(
                file_catalog.rd["record_type_id"],
                file_catalog.rd["offset"],
                file_catalog.rd["size"],
            )
        ):
            writer.writerow(str(n) for n in [idx, type_id, offset, size])
