"""
Tools for reading structured binary data
"""
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from io import FileIO
from struct import Struct
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


class ElementType(Enum):
    c8 = "c"
    i8 = "b"
    u8 = "B"
    i16 = "h"
    u16 = "H"
    i32 = "i"
    u32 = "I"
    i64 = "q"
    u64 = "Q"
    f32 = "f"
    f64 = "d"


@dataclass
class Element:
    type: ElementType
    name: Optional[str] = None
    count: int = 1


class DataBlock:
    """
    Reads fixed-size blocks of structured binary data, according to a specified format
    """

    def __init__(self, *elements: Element):
        self._elements = elements
        bom = "<"
        struct_fmt = bom
        np_tuples: List[Union[Tuple[str, str], Tuple[str, str, int]]] = []
        for element, name in zip(elements, self.names):
            fmt = element.type.value
            if element.count == 1:
                struct_fmt += fmt
                np_tuples.append((name, bom + fmt))
            else:
                struct_fmt += str(element.count) + fmt
                np_tuples.append((name, bom + fmt, element.count))
        self._struct = Struct(struct_fmt)
        self._dtype = np.dtype(np_tuples)

    @property
    def names(self) -> Iterable[str]:
        return (
            f"__reserved{i}__" if element.name is None else element.name
            for i, element in enumerate(self._elements)
        )

    @property
    def size(self) -> int:
        return self._struct.size

    def read(self, source: BinaryIO) -> Dict[str, Any]:
        unpacked_dict: Dict[str, Any] = {}
        unpacked = self._struct.unpack(source.read(self.size))
        stop = 0
        for element in self._elements:
            start = stop
            stop += element.count
            if element.name is not None:
                if element.count == 1:
                    unpacked_dict[element.name] = unpacked[start]
                else:
                    unpacked_dict[element.name] = unpacked[start:stop]
        return unpacked_dict

    def read_multiple(self, source: BinaryIO, count: int) -> Dict[str, List[Any]]:
        unpacked_dict: Dict[str, List[Any]] = defaultdict(list)
        for _ in range(count):
            for k, v in self.read(source).items():
                unpacked_dict[k].append(v)
        return unpacked_dict

    def read_dense(self, source: BinaryIO, count: int = 1) -> np.ndarray:
        if isinstance(source, FileIO):
            return np.fromfile(source, dtype=self._dtype, count=count)
        else:
            return np.frombuffer(
                source.read(self._dtype.itemsize * count),
                dtype=self._dtype,
                count=count,
            )
