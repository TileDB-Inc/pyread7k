"""
Tools for reading structured binary data
"""
from collections import defaultdict, namedtuple
from io import FileIO
from struct import Struct
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

_ElementTypes = namedtuple(
    "_ElementTypes",
    [
        # These are the canonical names of low-level element types:
        "c8",
        "i8",
        "u8",  # 3 int types of 1 byte
        "i16",
        "u16",  # 2 int types of 2 bytes
        "i32",
        "u32",  # 2 int types of 4 bytes
        "i64",
        "u64",  # 2 int types of 8 bytes
        "f32",  # 1 float type of 4 bytes
        "f64",  # 1 float type of 4 bytes
    ],
)
elemT = _ElementTypes(**dict(zip(_ElementTypes._fields, _ElementTypes._fields)))

_elemD_Type = Tuple[Optional[str], Tuple[str, int]]


def elemD_(name: Optional[str], fmt: str, count: int = 1) -> _elemD_Type:
    return (name, (fmt, count))


class DataBlock:
    """
    Reads fixed-size blocks of structured binary data, according to a specified format
    """

    _byte_order_fmt = "<"
    _fmt_mapping = {
        elemT.c8: "c",
        elemT.i8: "b",
        elemT.u8: "B",
        elemT.i16: "h",
        elemT.u16: "H",
        elemT.i32: "i",
        elemT.u32: "I",
        elemT.i64: "q",
        elemT.u64: "Q",
        elemT.f32: "f",
        elemT.f64: "d",
    }

    def __init__(self, elements: Sequence[_elemD_Type]):
        self._names, self._sizes = zip(*elements)
        bom = self._byte_order_fmt
        struct_fmt = bom
        np_tuples: List[Union[Tuple[str, str], Tuple[str, str, int]]] = []
        for field_name, (type_name, count) in zip(self.names, self._sizes):
            fmt = self._fmt_mapping[type_name]
            if count == 1:
                struct_fmt += fmt
                np_tuples.append((field_name, bom + fmt))
            else:
                struct_fmt += str(count) + fmt
                np_tuples.append((field_name, bom + fmt, count))
        self._struct = Struct(struct_fmt)
        self._dtype = np.dtype(np_tuples)

    @property
    def names(self) -> Iterable[str]:
        return (
            f"__reserved{i}__" if name is None else name
            for i, name in enumerate(self._names)
        )

    @property
    def size(self) -> int:
        return self._struct.size

    def read(self, source: BinaryIO, count: int = 1) -> Dict[str, Any]:
        dict_read: Dict[str, Any] = defaultdict(list)
        for _ in range(count):
            unpacked = self._struct.unpack(source.read(self.size))
            stop = 0
            for name, (_, count) in zip(self._names, self._sizes):
                start = stop
                stop += count
                if isinstance(name, str):
                    dict_read[name].extend(unpacked[start:stop])
        return {k: (v[0] if len(v) == 1 else v) for k, v in dict_read.items()}

    def read_dense(self, source: BinaryIO, count: int = 1) -> np.ndarray:
        if isinstance(source, FileIO):
            return np.fromfile(source, dtype=self._dtype, count=count)
        else:
            return np.frombuffer(
                source.read(self._dtype.itemsize * count),
                dtype=self._dtype,
                count=count,
            )
