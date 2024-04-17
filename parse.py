import argparse
import bisect
import io
import itertools
import json
import math
import os
import struct
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple


@dataclass
class Q:
    """Represents a quantity with a unit."""

    value: float | int = 0
    unit: str = ""

    @property
    def v(self):
        """Alias for q.value"""
        return self.value

    @v.setter
    def set_v(self, v: float | int):
        self.value = v

    @property
    def is_dimensionless(self):
        """Is the quantity dimensionless"""
        return not self.unit

    def __repr__(self) -> str:
        return f"Q({self.v} {self.unit or '(none)'})"


def unpack_stream(fmt, stream, peek=False):
    """Unpacks data from a stream."""
    size = struct.calcsize(fmt)
    if peek:
        buf = stream.peek(size)
    else:
        buf = stream.read(size)
    if not buf or len(buf) < size:
        return None
    return struct.unpack(fmt, buf)


def decode_nulterm_bytes(b: bytes, encoding="utf-8"):
    """Decodes a null-terminated string in a bytes sequence."""
    idx = b.find(b"\x00")
    if idx == -1:
        idx = len(b)
    return b[:idx].decode(encoding)


def read_nulterm_utf8(stream: io.BufferedReader):
    """Reads a null-terminated utf8 string from a stream.."""
    sequence = bytearray()
    while True:
        result = unpack_stream(">B", stream)
        if not result:
            return None
        byte = result[0]
        if byte == 0:
            break
        sequence.append(byte)
    return sequence.decode("utf-8")


@dataclass
class BoxHeader:
    """Represents an MP4 box header."""

    size: int
    type: bytes
    start: int
    hdr_end: int

    @property
    def box_end(self):
        return self.start + self.size


def read_box_header(reader: io.BufferedRandom):
    """Reads the box header at the current reader position."""
    box_start = reader.tell()

    result = unpack_stream(">I", reader)
    if not result:
        return None
    (size,) = result
    is_largesize = size == 1
    is_box_to_eof = size == 0

    result = unpack_stream(">4s", reader)
    if not result:
        return None
    (box_type,) = result

    if is_box_to_eof:
        file_length = os.fstat(reader.fileno()).st_size
        size = file_length - reader.tell()
    elif is_largesize:
        result = unpack_stream(">Q", reader)
        if not result:
            return None
        (size,) = result

    hdr_end = reader.tell()
    return BoxHeader(size, box_type, box_start, hdr_end)


@dataclass(kw_only=True)
class Box:
    header: BoxHeader
    meta_end: int

    @property
    def meta_start(self):
        return self.header.hdr_end

    @property
    def size(self):
        return self.header.size

    @property
    def type(self):
        return self.header.type

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        reader.seek(hdr.hdr_end)
        return cls(header=hdr, meta_end=hdr.hdr_end)


@dataclass(kw_only=True)
class FullBox(Box):
    version: int
    flags: int

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = Box.read_from(reader, hdr)
        result = unpack_stream(">4B", reader)
        if not result:
            return None
        return cls(
            header=box.header,
            meta_end=reader.tell(),
            version=result[0],
            flags=(result[1] << 16) | (result[2] << 8) | result[3],
        )


@dataclass(kw_only=True)
class FileTypeBox(Box):
    major_brand: int
    minor_version: int
    compatible_brands: List[int]

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = Box.read_from(reader, hdr)
        result = unpack_stream(">4sI", reader)
        if not result:
            return None

        major_brand, minor_version = result
        compatible_brands: List[int] = []
        while reader.tell() < box.header.box_end:
            result = unpack_stream(">4s", reader)
            if not result:
                return None
            compatible_brands.append(result[0])

        return cls(
            header=box.header,
            meta_end=reader.tell(),
            major_brand=major_brand,
            minor_version=minor_version,
            compatible_brands=compatible_brands,
        )


@dataclass(kw_only=True)
class HandlerReferenceBox(FullBox):
    handler_type: bytes
    name: str

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = FullBox.read_from(reader, hdr)
        result = unpack_stream(">I4s3I", reader)
        if not result:
            return None
        _, handler_type, *_ = result
        name = read_nulterm_utf8(reader)
        if name is None:
            return None
        return cls(
            header=box.header,
            meta_end=reader.tell(),
            version=box.version,
            flags=box.flags,
            handler_type=handler_type,
            name=name,
        )


@dataclass(kw_only=True)
class SampleEntryBox(Box):
    data_reference_index: int

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = Box.read_from(reader, hdr)
        # parse reserved
        result = unpack_stream(">6B", reader)
        if not result:
            return None

        result = unpack_stream(">H", reader)
        if not result:
            return None
        ref_idx = result[0]
        return cls(
            header=box.header,
            meta_end=reader.tell(),
            data_reference_index=ref_idx,
        )


@dataclass(kw_only=True)
class SampleDescriptionBox(FullBox):
    entry_count: int

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = FullBox.read_from(reader, hdr)
        result = unpack_stream(">I", reader)
        if not result:
            return None
        count = result[0]
        return cls(
            header=box.header,
            meta_end=reader.tell(),
            version=box.version,
            flags=box.flags,
            entry_count=count,
        )


@dataclass(kw_only=True)
class TimeToEntryBox(FullBox):
    """Also known as the time-to-sample box"""

    entry_count: int
    entry_sample_count: List[int]
    entry_sample_delta: List[int]

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = FullBox.read_from(reader, hdr)
        result = unpack_stream(">I", reader)
        if not result:
            return None

        count = result[0]
        sample_count: List[int] = []
        sample_delta: List[int] = []
        for _ in range(count):
            result = unpack_stream(">2I", reader)
            if not result:
                return None
            sample_count.append(result[0])
            sample_delta.append(result[1])

        return cls(
            header=box.header,
            meta_end=reader.tell(),
            version=box.version,
            flags=box.flags,
            entry_count=count,
            entry_sample_count=sample_count,
            entry_sample_delta=sample_delta,
        )

    def iter_sample_times(self):
        ts = 0
        for count, delta in zip(self.entry_sample_count, self.entry_sample_delta):
            while count:
                yield ts
                ts += delta


@dataclass(kw_only=True)
class SampleToChunkBox(FullBox):
    entry_count: int
    # chunks are 1-indexed here
    entry_first_chunk: List[int]
    entry_samples_per_chunk: List[int]
    entry_sample_description_index: List[int]

    def get_samples_per_chunk(self, chunk_index: int):
        """chunk_index should be 0-indexed despite entry_first_chunk containing 1-indexed indices."""
        # this assumes that entry_first_chunk is sorted
        idx = bisect.bisect_right(self.entry_first_chunk, chunk_index + 1)
        if idx == 0:
            return None
        return self.entry_samples_per_chunk[idx - 1]

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = FullBox.read_from(reader, hdr)
        result = unpack_stream(">I", reader)
        if not result:
            return None

        count = result[0]
        first_chunk: List[int] = []
        samples_per_chunk: List[int] = []
        sample_description_index: List[int] = []
        for _ in range(count):
            result = unpack_stream(">3I", reader)
            if not result:
                return None
            first_chunk.append(result[0])
            samples_per_chunk.append(result[1])
            sample_description_index.append(result[2])

        return cls(
            header=box.header,
            meta_end=reader.tell(),
            version=box.version,
            flags=box.flags,
            entry_count=count,
            entry_first_chunk=first_chunk,
            entry_samples_per_chunk=samples_per_chunk,
            entry_sample_description_index=sample_description_index,
        )


@dataclass(kw_only=True)
class SampleSizeBox(FullBox):
    sample_size: int
    sample_count: int
    sample_entry_size: List[int] | None = None

    def get_sample_size(self, sample_index: int):
        if self.sample_entry_size:
            return self.sample_entry_size[sample_index]
        return self.sample_size

    def iter_sample_sizes(self) -> Iterable[int]:
        if self.sample_entry_size:
            return iter(self.sample_entry_size)
        return itertools.repeat(self.sample_size, self.sample_count)

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = FullBox.read_from(reader, hdr)
        result = unpack_stream(">2I", reader)
        if not result:
            return None

        size = result[0]
        count = result[1]
        entry_sizes = None

        if size == 0:
            entry_sizes = []
            for _ in range(count):
                result = unpack_stream(">I", reader)
                if not result:
                    return None
                entry_sizes.append(result[0])

        return cls(
            header=box.header,
            meta_end=reader.tell(),
            version=box.version,
            flags=box.flags,
            sample_size=size,
            sample_count=count,
            sample_entry_size=entry_sizes,
        )


@dataclass(kw_only=True)
class AbstractChunkOffsetBox(FullBox):
    entry_count: int
    entry_chunk_offset: List[int]
    _chunk_offset_size: str = field(default="", init=False)

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = FullBox.read_from(reader, hdr)
        result = unpack_stream(">I", reader)
        if not result:
            return None

        count = result[0]
        chunk_offsets = []
        for _ in range(count):
            result = unpack_stream(f">{cls._chunk_offset_size}", reader)
            if not result:
                return None
            chunk_offsets.append(result[0])

        return cls(
            header=box.header,
            meta_end=reader.tell(),
            version=box.version,
            flags=box.flags,
            entry_count=count,
            entry_chunk_offset=chunk_offsets,
        )


@dataclass(kw_only=True)
class ChunkOffsetBox(AbstractChunkOffsetBox):
    _chunk_offset_size = "I"


@dataclass(kw_only=True)
class ChunkLargeOffsetBox(AbstractChunkOffsetBox):
    _chunk_offset_size = "Q"


@dataclass(kw_only=True)
class MediaHeaderBox(FullBox):
    creation_time: int
    modification_time: int
    timescale: int
    # in the scale of the timescale
    duration: int

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = FullBox.read_from(reader, hdr)
        if box.version == 1:
            result = unpack_stream(">QQIQ", reader)
        else:
            result = unpack_stream(">IIII", reader)
        if not result:
            return None

        creation, modification, timescale, duration = result
        # skip the language code (16 bits) and pre_defined (16 bits)
        unpack_stream(">2H", reader)

        return cls(
            header=box.header,
            meta_end=reader.tell(),
            version=box.version,
            flags=box.flags,
            creation_time=creation,
            modification_time=modification,
            timescale=timescale,
            duration=duration,
        )


box_parsers: Dict[bytes, "Box"] = {
    b"ftyp": FileTypeBox,
    b"hdlr": HandlerReferenceBox,
    b"gmhd": Box,
    b"gmin": Box,
    b"gpmd": SampleEntryBox,
    b"stbl": Box,
    b"stsd": SampleDescriptionBox,
    b"stts": TimeToEntryBox,
    b"stsc": SampleToChunkBox,
    b"stsz": SampleSizeBox,
    b"stco": ChunkOffsetBox,
    b"co64": ChunkLargeOffsetBox,
    b"mdhd": MediaHeaderBox,
}


@staticmethod
def parse_box(reader: io.BufferedReader):
    """Parses an MP4 Box."""
    hdr = read_box_header(reader)
    if not hdr:
        return None

    cls = box_parsers.get(hdr.type, None)
    if not cls:
        # print(f"[WARN] Cannot handle box of type {hdr.type}")
        return Box.read_from(reader, hdr)

    return cls.read_from(reader, hdr)


def iterboxes(reader: io.BufferedRandom, container: Box = None) -> Iterable[Box]:
    """Iterates over MP4 boxes."""
    limit = math.inf
    if container:
        limit = container.header.box_end
        reader.seek(container.meta_end)
    while reader.tell() < limit:
        box = parse_box(reader)
        if not box:
            break
        yield box
        reader.seek(box.header.box_end)


def find_box_by_type(box_type: str | bytes, iterable: Iterable[Box]):
    if isinstance(box_type, str):
        box_type = bytes(box_type, "ascii")
    for box in iterable:
        if box.type == box_type:
            return box
    return None


@dataclass
class Mp4Sample:
    """Represents an MP4 track sample."""

    offset: int
    size: int
    decoding_time: int


def iter_track_samples(
    *,
    stsz: SampleSizeBox,
    cob: AbstractChunkOffsetBox,
    stsc: SampleToChunkBox,
    stts: TimeToEntryBox,
):
    """Iterate over the mp4 track's samples."""
    sample_sizes_iter = stsz.iter_sample_sizes()
    sample_times_iter = stts.iter_sample_times()
    for idx, chunk_offset in enumerate(cob.entry_chunk_offset):
        samples_per_chunk = stsc.get_samples_per_chunk(idx)
        sample_offset_in_chunk = 0
        for _ in range(samples_per_chunk):
            sample_size = next(sample_sizes_iter)
            yield Mp4Sample(
                offset=chunk_offset + sample_offset_in_chunk,
                size=sample_size,
                decoding_time=next(sample_times_iter),
            )
            sample_offset_in_chunk += sample_size


def iter_sample_file_offset(
    *, stsz: SampleSizeBox, cob: AbstractChunkOffsetBox, stsc: SampleToChunkBox
):
    """Iterates over the track sample's position + size in an MP4 file.

    Samples are defined by the stsz, cob, and stsc boxes.
    """
    sample_sizes_iter = stsz.iter_sample_sizes()
    for idx, chunk_offset in enumerate(cob.entry_chunk_offset):
        samples_per_chunk = stsc.get_samples_per_chunk(idx)
        sample_offset_in_chunk = 0
        for _ in range(samples_per_chunk):
            sample_size = next(sample_sizes_iter)
            yield chunk_offset + sample_offset_in_chunk, sample_size
            sample_offset_in_chunk += sample_size


gpmf_type_to_struct_fmt = {
    ord(b"b"): "b",
    ord(b"B"): "B",
    ord(b"c"): "s",
    ord(b"d"): "d",
    ord(b"f"): "f",
    ord(b"F"): "4s",
    ord(b"G"): "16s",
    ord(b"j"): "q",
    ord(b"J"): "Q",
    ord(b"l"): "i",
    ord(b"L"): "I",
    ord(b"q"): "I",  # Q number
    ord(b"Q"): "Q",  # Q number
    ord(b"s"): "h",
    ord(b"S"): "H",
    ord(b"U"): "16s",
}


@dataclass
class Scaler:
    """Helper class to scale values."""

    # scale is a list of divisors
    scale: List[float] = field(default_factory=lambda: [1])

    @classmethod
    def from_scal_gpmf(cls, fp: io.BufferedRandom, scal: Optional["GpmfEntry"]):
        """Construct a Scaler from a GPMF SCAL entry."""
        if not scal:
            return cls()
        return cls(list(scal.iter_samples(fp, apply_modifiers=False)))

    def try_scale(self, it: Any):
        """Tries to scale a given input.

        If `it' is not a list of ints or floats, the input is passed through as-is.
        """
        if not any(isinstance(it, typ) for typ in (tuple, list)):
            return it

        output = []
        for value in it:
            if any(isinstance(value, typ) for typ in (int, float)):
                output.append(value / self.scale[0])
            elif any(isinstance(value, typ) for typ in [tuple, list]):
                output.append(list(val / scal for val, scal in zip(value, self.scale)))
            else:
                output.append(value)
        return output


@dataclass(kw_only=True)
class GpmfEntry:
    """Represents a GPMF entry/frame."""

    fourcc: bytes
    type: bytes
    # size of each gpmf entry sample
    sample_size: int
    # number of gpmf entry samples
    num_samples: int
    # offset in the stream where the data starts
    data_start_offset: int
    level: int = 0
    # modifier properties as gpmf entries
    scale: Optional["GpmfEntry"] = None
    display_units: Optional["GpmfEntry"] = None
    standard_units: Optional["GpmfEntry"] = None

    @property
    def data_size(self):
        return self.sample_size * self.num_samples

    @property
    def data_size_aligned(self):
        """The data size, aligned to 32 bits"""
        return self.data_size + ((4 - self.data_size % 4) % 4)

    @property
    def data_end_offset_aligned(self):
        """The 32-bit aligned end offset of the entry data"""
        return self.data_start_offset + self.data_size_aligned

    @property
    def is_nested(self):
        return self.type == b"\x00"

    @property
    def is_complex_type(self):
        return self.type == b"?"

    @property
    def units(self):
        """Prefers display units, then standard units"""
        return self.display_units or self.standard_units

    def get_unit(self, index: int, reader: io.BufferedRandom, encoding="utf-8"):
        """Gets a unit for index I of an N-tuple sample."""
        if not self.units:
            return ""
        try:
            return decode_nulterm_bytes(
                list(self.units.iter_samples(reader))[index], encoding=encoding
            )
        except IndexError:
            return ""

    def _get_struct_fmt(self):
        fmt = gpmf_type_to_struct_fmt[self.type[0]]
        fmt_size = struct.calcsize(fmt)
        # Handles cases such as strings and MTRX.
        # A type of "f" (32-bit float) with size=36 and samples=1 means
        # that the sample format should be "9f" (corresponds to 36 bytes).
        # this makes sense for MTRX, which is a 3x3 matrix.
        if fmt_size < self.sample_size:
            fmt = f"{self.sample_size // fmt_size}{fmt}"
        return f">{fmt}"

    def get_data_buffer(self, reader: io.BufferedRandom) -> bytes:
        """Gets the raw data buffer for this GPMF entry/frame."""
        reader.seek(self.data_start_offset)
        return reader.read(self.data_size)

    def iter_raw_samples(self, reader: io.BufferedRandom) -> Generator[Any, None, None]:
        """Iterates over the samples in a GPMF entry/frame.

        No type processing or scaling is done; raw bytes are yielded.
        """
        if self.is_nested or self.is_complex_type:
            return None

        offset = self.data_start_offset
        for _ in range(self.num_samples):
            reader.seek(offset)
            result = reader.read(self.sample_size)
            yield result
            offset += self.sample_size

    def iter_samples(
        self,
        reader: io.BufferedRandom,
        apply_modifiers=True,
        collapse_single_result=True,
    ) -> Generator[Any, None, None]:
        """Iterates over the samples in a GPMF entry/frame."""
        if self.is_nested or self.is_complex_type:
            return None

        scaler = Scaler.from_scal_gpmf(reader, self.scale)

        struct_fmt = self._get_struct_fmt()
        for raw_sample in self.iter_raw_samples(reader):
            result = struct.unpack(struct_fmt, raw_sample)
            if not result:
                return None
            if apply_modifiers:
                result = list(scaler.try_scale(result))
            # handles complex types that need to return multiple values
            if collapse_single_result and len(result) == 1:
                result = result[0]
            yield result


@dataclass(kw_only=True)
class ComplexGpmfEntry(GpmfEntry):
    """A GPMF entry with a complex type."""

    complex_type: bytes

    def _get_struct_fmt(self):
        "".join(gpmf_type_to_struct_fmt[type_] for type_ in self.complex_type)


def read_gpmf_entry(reader: io.BufferedRandom, pos: int, level=0):
    """Reads a GPMF entry at a given position."""
    reader.seek(pos)
    result = unpack_stream(">4scBH", reader)
    if not result:
        return None

    fourcc, data_type, sample_size, num_samples = result
    return GpmfEntry(
        level=level,
        fourcc=fourcc,
        type=data_type,
        sample_size=sample_size,
        num_samples=num_samples,
        data_start_offset=reader.tell(),
    )


def unpack_type_specifier(struct_specifier: bytes):
    """Unpacks the packed GPMF complex type specifier."""
    specifier_iter = iter(struct_specifier)
    unpacked = bytearray()

    # states: no_type, has_type, in_array
    state = "no_type"
    cur_type = None
    while True:
        if state == "no_type":
            ch = next(specifier_iter, None)
            if not ch:
                break
            if ch not in gpmf_type_to_struct_fmt:
                raise RuntimeError("Encountered invalid type")
            unpacked.append(ch)
            state = "has_type"
        elif state == "has_type":
            ch = next(specifier_iter, None)
            if not ch:
                break
            if ch == ord(b"["):
                state = "in_array"
            elif ch in gpmf_type_to_struct_fmt:
                unpacked.append(ch)
            else:
                raise RuntimeError("Encountered invalid type")
        elif state == "in_array":
            count = 0
            while True:
                ch = next(specifier_iter)
                if ch == ord(b"]"):
                    break
                count = count * 10 + int(ch)
            # -1 because we've already added the first char to unpacked
            unpacked.extend(cur_type * (count - 1))
            state = "no_type"
    return bytes(unpacked)


def parse_gpmf(
    reader: io.BufferedRandom, start_offset: int
) -> Generator[GpmfEntry | None, None, None]:
    """Iterates over the GPMF entries in a file, starting at start_offset."""

    def recurse_parser(start: int, level=0) -> Generator[GpmfEntry, None, None]:
        entry = read_gpmf_entry(reader, start, level=level)
        if not entry:
            return None

        yield entry
        if entry.is_nested:
            next_level = level + 1
            type_modifier = None
            reader.seek(entry.data_start_offset)
            while reader.tell() < entry.data_end_offset_aligned:
                for child in recurse_parser(reader.tell(), level=next_level):
                    if child.level == next_level:
                        if child.fourcc == b"TYPE":
                            type_modifier = unpack_type_specifier(
                                child.get_data_buffer(reader)
                            )
                        elif child.is_complex_type:
                            child = ComplexGpmfEntry(
                                complex_type=type_modifier, **asdict(child)
                            )
                    yield child
                    reader.seek(child.data_end_offset_aligned)

    yield from recurse_parser(start_offset)


def iter_gopro_meta_samples(fp: io.BufferedRandom, debug=False):
    """Iterate over the track samples in the GoPro META track."""
    moov = find_box_by_type("moov", iterboxes(fp))
    if not moov:
        raise RuntimeError("no moov box")

    for trak in iterboxes(fp, moov):
        if trak.type != b"trak":
            continue

        mdia = find_box_by_type("mdia", iterboxes(fp, trak))
        if not mdia:
            continue

        hdlr: HandlerReferenceBox = find_box_by_type("hdlr", iterboxes(fp, mdia))
        if not hdlr:
            continue

        if hdlr.handler_type != b"meta" or "GoPro MET" not in hdlr.name:
            continue

        mdhd: MediaHeaderBox = find_box_by_type("mdhd", iterboxes(fp, mdia))
        if not mdhd:
            continue

        minf = find_box_by_type("minf", iterboxes(fp, mdia))
        if not minf:
            continue

        gmhd = find_box_by_type("gmhd", iterboxes(fp, minf))
        if not gmhd:
            continue

        # find gmin (media info) and gpmd (type for gpmf data)
        # for box in iterboxes(fp, gmhd):
        #   print("found:", box.type, box)

        stbl = find_box_by_type("stbl", iterboxes(fp, minf))
        if not stbl:
            continue

        stsd: SampleDescriptionBox = find_box_by_type("stsd", iterboxes(fp, stbl))
        if not stsd:
            continue

        if stsd.entry_count != 1:
            continue

        gpmd = list(iterboxes(fp, stsd))[0]
        if gpmd.type != b"gpmd":
            continue

        stts: TimeToEntryBox = find_box_by_type("stts", iterboxes(fp, stbl))
        stsc: SampleToChunkBox = find_box_by_type("stsc", iterboxes(fp, stbl))
        stsz: SampleSizeBox = find_box_by_type("stsz", iterboxes(fp, stbl))
        stco: ChunkOffsetBox = find_box_by_type("stco", iterboxes(fp, stbl))
        co64: ChunkLargeOffsetBox = find_box_by_type("co64", iterboxes(fp, stbl))
        cob: AbstractChunkOffsetBox = co64 or stco
        if not (stts and stsc and stsz and cob):
            continue

        if debug:
            print("Found GPMF track")
            print(f"Timescale: {mdhd.timescale}")
            print(f"Duration: {mdhd.duration}")
            print(f"There are {cob.entry_count} chunks")
            print(f"There are {stsz.sample_count} samples")

        for sample in iter_track_samples(stsz=stsz, cob=cob, stsc=stsc, stts=stts):
            yield sample


StreamType = Tuple[GpmfEntry, List[GpmfEntry]]


def apply_modifier_properties(entries: List[GpmfEntry]):
    """Applies modifier properties based on GPMF order.

    Only applies: SCAL, UNIT, SIUN
    """
    scal, unit, siun = None, None, None
    for entry in entries:
        if entry.fourcc == b"SCAL":
            scal = entry
        elif entry.fourcc == b"UNIT":
            unit = entry
        elif entry.fourcc == b"SIUN":
            siun = entry
        else:
            entry.scale = scal
            entry.display_units = unit
            entry.standard_units = siun
    return entries


def iter_gpmf_streams(fp: io.BufferedRandom, sample: Mp4Sample):
    """Iterate over all GPMF Streams."""
    # Sequence[Tuple[<STRM GpmfEntry>, Sequence[GpmfEntry]]]
    streams: Sequence[StreamType] = []
    for entry in parse_gpmf(fp, sample.offset):
        if not entry:
            continue

        cur_stream = streams[-1] if streams else None

        if entry.fourcc.startswith(b"STRM"):
            if (
                cur_stream
                and cur_stream[0].level  # pylint: disable=unsubscriptable-object
                >= entry.level
            ):
                # exiting a nested stream
                strm, entries = streams.pop()
                yield strm, apply_modifier_properties(entries)
            streams.append((entry, []))
        elif cur_stream:
            cur_stream[1].append(entry)  # pylint: disable=unsubscriptable-object


def iter_gpmf_gps_streams(fp: io.BufferedRandom, sample: Mp4Sample):
    """Iterate over GPMF Streams that contain GPS child entries."""
    for strm_entry, entries in iter_gpmf_streams(fp, sample):
        if any(b"GPS" in entry.fourcc for entry in entries):
            yield strm_entry, entries


def print_gpmf_streams_from(iterable: Iterable[StreamType]):
    for stream, entries in iterable:
        print("  " * stream.level, stream)
        for entry in entries:
            print("  " * stream.level + "  ", entry)


def print_gpmf_streams(fp: io.BufferedRandom, sample: Mp4Sample):
    print_gpmf_streams_from(iter_gpmf_streams(fp, sample))


def print_gpmf_samples(fp: io.BufferedRandom):
    for sample_idx, sample in enumerate(iter_gopro_meta_samples(fp, debug=True)):
        print(
            f"Parsing sample {sample_idx} from {sample.offset} to {sample.offset + sample.size}"
        )
        for entry in parse_gpmf(fp, sample.offset):
            if not entry:
                continue

            print(
                "  " * entry.level,
                entry,
                f"data_size={entry.data_size}, end_offset_aligned={entry.data_end_offset_aligned}",
            )

            if entry.type == b"c":
                # A SIUN of m/s^2 uses latin-1 encoding, NOT ASCII >:(
                str_samples = map(
                    lambda s: decode_nulterm_bytes(s, "latin-1"),
                    entry.iter_samples(fp),
                )
                print(
                    "  " * (entry.level + 1),
                    f"Value: {list(str_samples)}",
                )
            elif not (entry.is_complex_type or entry.is_nested):
                some_samples = itertools.islice(entry.iter_samples(fp), 5)
                print("  " * (entry.level + 1), f"Value: {list(some_samples)}")


@dataclass
class GpsFrame:
    """Based on WGS84. Represents a GPMF stream "frame" containing GPS samples."""

    @dataclass
    class Sample:
        """A GPS sample inside a GPS5 GPMF entry."""

        latitude: Q = field(default_factory=Q)
        longitude: Q = field(default_factory=Q)
        altitude: Q = field(default_factory=Q)
        speed_2d: Q = field(default_factory=Q)
        speed_3d: Q = field(default_factory=Q)

    fix: int = 0
    video_time_offset: int = 0
    gps_timestamp: str = ""
    precision: int = 0
    data: Sequence[Sample] = field(default_factory=list)


def get_gps_frames(fp: io.BufferedRandom, debug=False):
    """Reads out the GPS frames from a GoPro MP4."""
    frames: List[GpsFrame] = []
    for sample_idx, sample in enumerate(iter_gopro_meta_samples(fp)):
        if debug:
            print(f"----- sample #{sample_idx} (time={sample.decoding_time})")
        for _, entries in iter_gpmf_gps_streams(fp, sample):
            gps_frame = GpsFrame(video_time_offset=sample.decoding_time)
            for entry in entries:
                if debug:
                    print(entry)
                if entry.fourcc == b"GPSF":
                    # should be a single sample
                    gps_frame.fix = next(entry.iter_samples(fp))
                elif entry.fourcc == b"GPSU":
                    # should be a single sample. ASCII is simple enough for numbers.
                    gps_frame.gps_timestamp = next(entry.iter_samples(fp)).decode(
                        "ascii"
                    )
                elif entry.fourcc == b"GPSP":
                    # should be a single sample
                    gps_frame.precision = next(entry.iter_samples(fp))
                elif entry.fourcc == b"GPS5":
                    gps_frame.data = []
                    for gpmf_sample in entry.iter_samples(fp):
                        if len(gpmf_sample) != 5:
                            raise ValueError(
                                f"Received GPS5 sample with {len(gpmf_sample)} values!"
                            )
                        gps_frame.data.append(
                            GpsFrame.Sample(
                                longitude=Q(gpmf_sample[0], entry.get_unit(0, fp)),
                                latitude=Q(gpmf_sample[1], entry.get_unit(1, fp)),
                                altitude=Q(gpmf_sample[2], entry.get_unit(2, fp)),
                                speed_2d=Q(gpmf_sample[3], entry.get_unit(3, fp)),
                                speed_3d=Q(gpmf_sample[4], entry.get_unit(4, fp)),
                            )
                        )
            frames.append(gps_frame)
            if debug:
                print(gps_frame)
    return frames


def analyze_gps_frames(frames: Sequence[GpsFrame]):
    if len(frames) == 0:
        print("No GPS frames")
        return

    # adds an extra second, since the last sample presumably
    # runs until the end of the second.
    end_time = frames[-1].video_time_offset / 1000 + 1
    gps_data_count = 0
    per_frame_gps_data_count = Counter()

    for sample in frames:
        gps_data_count += len(sample.data)
        per_frame_gps_data_count[len(sample.data)] += 1

    print(f"Measured frequency (targeting ~18Hz): {gps_data_count / end_time:.2f}Hz")
    print(
        "Samples per frame -> count of frames w/ that # of samples:",
        per_frame_gps_data_count,
    )


def write_to_flatbuffer(gps_frames: Sequence[GpsFrame]):
    import GoPro.MetaTrack
    import GoPro.TrackSample
    import GoPro.GpsFrame
    import GoPro.GpsSample
    import flatbuffers

    builder = flatbuffers.Builder(0)

    # build inner structures before container structures

    seq_of_frame_samples = []
    for frame in gps_frames:
        frame_samples = []
        seq_of_frame_samples.append(frame_samples)
        for gps_sample in frame.data:
            sample = GoPro.GpsSample.CreateGpsSample(
                builder,
                gps_sample.latitude.v,
                gps_sample.longitude.v,
                gps_sample.altitude.v,
                gps_sample.speed_2d.v,
                gps_sample.speed_3d.v,
            )
            frame_samples.append(sample)

    frame_data_vectors = []
    for frame_samples in seq_of_frame_samples:
        GoPro.GpsFrame.StartDataVector(builder, len(frame_samples))
        # prepend vector items in reverse order
        for sample in frame_samples[::-1]:
            builder.PrependUOffsetTRelative(sample)
        frame_data_vectors.append(builder.EndVector())

    seq_of_frames = []
    for data_vector, gps_frame in zip(frame_data_vectors, gps_frames):
        GoPro.GpsFrame.GpsFrameStart(builder)
        GoPro.GpsFrame.AddFix(builder, int(gps_frame.fix))
        GoPro.GpsFrame.AddPrecision(builder, gps_frame.precision)
        GoPro.GpsFrame.AddData(builder, data_vector)
        seq_of_frames.append(GoPro.GpsFrame.GpsFrameEnd(builder))

    track_samples = []
    for frame, gps_frame in zip(seq_of_frames, gps_frames):
        GoPro.TrackSample.TrackSampleStart(builder)
        GoPro.TrackSample.AddDecodingTime(builder, gps_frame.video_time_offset)
        GoPro.TrackSample.AddGps(builder, frame)
        track_samples.append(GoPro.TrackSample.TrackSampleEnd(builder))

    GoPro.MetaTrack.StartSamplesVector(builder, len(track_samples))
    # prepend vector items in reverse order
    for track_sample in track_samples[::-1]:
        builder.PrependUOffsetTRelative(track_sample)
    track_samples_vector = builder.EndVector()

    GoPro.MetaTrack.MetaTrackStart(builder)
    GoPro.MetaTrack.AddSamples(builder, track_samples_vector)
    meta_track = GoPro.MetaTrack.MetaTrackEnd(builder)

    builder.Finish(meta_track)
    return builder.Output()


def print_cmd(args):
    with open(args.file, "rb") as fp:
        print_gpmf_samples(fp)


def gps_json_cmd(args):
    with open(args.file, "rb") as fp:
        gps_frames = get_gps_frames(fp, debug=args.debug)
        # all of our data model is represented with dataclasses, so
        # asdict() works here.
        print(json.dumps(gps_frames, indent=2, default=asdict))


def gps_analyze_cmd(args):
    with open(args.file, "rb") as fp:
        gps_frames = get_gps_frames(fp, debug=args.debug)
        analyze_gps_frames(gps_frames)


def gps_flatbuffers_cmd(args):
    with open(args.file, "rb") as fp:
        gps_frames = get_gps_frames(fp, debug=args.debug)
    with open(args.output, "wb") as fp:
        fp.write(write_to_flatbuffer(gps_frames))
    print(f"Written to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(run_cmd=parser.print_help)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug output",
    )

    subparsers = parser.add_subparsers()

    print_parser = subparsers.add_parser("print")
    print_parser.add_argument("file", type=str, help="MP4 file to parse")
    print_parser.set_defaults(run_cmd=print_cmd)

    gps_json_parser = subparsers.add_parser("gps-json")
    gps_json_parser.add_argument("file", type=str, help="MP4 file to parse")
    gps_json_parser.set_defaults(run_cmd=gps_json_cmd)

    gps_analyze_parser = subparsers.add_parser("gps-analyze")
    gps_analyze_parser.add_argument("file", type=str, help="MP4 file to parse")
    gps_analyze_parser.set_defaults(run_cmd=gps_analyze_cmd)

    gps_flatbuffers_parser = subparsers.add_parser("gps-flatbuffers")
    gps_flatbuffers_parser.add_argument("file", type=str, help="MP4 file to parse")
    gps_flatbuffers_parser.add_argument(
        "output", type=str, help="Flatbuffer file to write to"
    )
    gps_flatbuffers_parser.set_defaults(run_cmd=gps_flatbuffers_cmd)

    return parser.parse_args()


def main(args):
    args.run_cmd(args)


if __name__ == "__main__":
    main(parse_args())
