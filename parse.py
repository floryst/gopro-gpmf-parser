import bisect
import itertools
import sys
import math
from pathlib import Path
import io
import struct
import os
from dataclasses import dataclass, asdict, field
from typing import Iterable, List, Dict, Generator, Any, Tuple, Optional


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
    idx = b.find(b"\x00")
    if idx == -1:
        idx = len(b)
    return b[:idx].decode(encoding)


def read_nulterm_utf8(stream):
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
    size: int
    type: bytes
    start: int
    hdr_end: int

    @property
    def box_end(self):
        return self.start + self.size


def read_box_header(reader: io.BufferedRandom):
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
class ChunkOffsetBox(FullBox):
    entry_count: int
    entry_chunk_offset: List[int]

    @classmethod
    def read_from(cls, reader: io.BufferedRandom, hdr: BoxHeader):
        box = FullBox.read_from(reader, hdr)
        result = unpack_stream(">I", reader)
        if not result:
            return None

        count = result[0]
        chunk_offsets = []
        for _ in range(count):
            result = unpack_stream(">I", reader)
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
    b"mdhd": MediaHeaderBox,
}


@staticmethod
def parse_box(reader: io.BufferedReader):
    hdr = read_box_header(reader)
    if not hdr:
        return None

    cls = box_parsers.get(hdr.type, None)
    if not cls:
        # print(f"[WARN] Cannot handle box of type {hdr.type}")
        return Box.read_from(reader, hdr)

    return cls.read_from(reader, hdr)


def iterboxes(reader: io.BufferedRandom, container: Box = None) -> Iterable[Box]:
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


def iter_sample_file_offset(
    *, stsz: SampleSizeBox, stco: SampleToChunkBox, stsc: SampleToChunkBox
):
    sample_sizes_iter = stsz.iter_sample_sizes()
    for idx, chunk_offset in enumerate(stco.entry_chunk_offset):
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
    # scale is a list of divisors
    scale: List[float] = field(default_factory=lambda: [1])

    @classmethod
    def from_scal_gpmf(cls, fp: io.BufferedRandom, scal: Optional["GpmfEntry"]):
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
    fourcc: bytes
    type: bytes
    sample_size: int
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
        reader.seek(self.data_start_offset)
        return reader.read(self.data_size)

    def iter_raw_samples(self, reader: io.BufferedRandom) -> Generator[Any, None, None]:
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
    complex_type: bytes

    def _get_struct_fmt(self):
        "".join(gpmf_type_to_struct_fmt[type_] for type_ in self.complex_type)


def read_gpmf_entry(reader: io.BufferedRandom, pos: int, level=0):
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
    reader: io.BufferedRandom, start_offset: int, end_offset: int
) -> Generator[GpmfEntry | None, None, None]:

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


def itersamples(fp: io.BufferedRandom, debug=False):
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

        stbl_boxes = iterboxes(fp, stbl)
        stts: TimeToEntryBox = find_box_by_type("stts", stbl_boxes)
        stsc: SampleToChunkBox = find_box_by_type("stsc", stbl_boxes)
        stsz: SampleSizeBox = find_box_by_type("stsz", stbl_boxes)
        stco: ChunkOffsetBox = find_box_by_type("stco", stbl_boxes)
        if not (stts and stsc and stsz and stco):
            continue

        if debug:
            print("Found GPMF track")
            print(f"Timescale: {mdhd.timescale}")
            print(f"Duration: {mdhd.duration}")
            print(f"There are {stco.entry_count} chunks")
            print(f"There are {stsz.sample_count} samples")

        for sample_offset, sample_size in iter_sample_file_offset(
            stsz=stsz, stco=stco, stsc=stsc
        ):
            yield sample_offset, sample_size


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


def iter_gpmf_streams(fp: io.BufferedRandom, sample_offset: int, sample_size: int):
    # List[Tuple[<STRM GpmfEntry>, List[GpmfEntry]]]
    streams: List[StreamType] = []
    for entry in parse_gpmf(fp, sample_offset, sample_offset + sample_size):
        if not entry:
            continue

        cur_stream = streams[-1] if streams else None

        if entry.fourcc.startswith(b"STRM"):
            if cur_stream and cur_stream[0].level >= entry.level:
                # exiting a nested stream
                strm, entries = streams.pop()
                yield strm, apply_modifier_properties(entries)
            streams.append((entry, []))
        elif cur_stream:
            cur_stream[1].append(entry)


def iter_gpmf_gps_streams(fp: io.BufferedRandom, sample_offset: int, sample_size: int):
    for strm_entry, entries in iter_gpmf_streams(fp, sample_offset, sample_size):
        if any(b"GPS" in entry.fourcc for entry in entries):
            yield strm_entry, entries


def print_gpmf_streams_from(iterable: Iterable[StreamType]):
    for stream, entries in iterable:
        print("  " * stream.level, stream)
        for entry in entries:
            print("  " * stream.level + "  ", entry)


def print_gpmf_streams(fp: io.BufferedRandom, sample_offset: int, sample_size: int):
    print_gpmf_streams_from(iter_gpmf_streams(fp, sample_offset, sample_size))


def print_gpmf_samples(fp: io.BufferedRandom):
    for sample_idx, (sample_offset, sample_size) in enumerate(
        itersamples(fp, debug=True)
    ):
        print(
            f"Parsing sample {sample_idx} from {sample_offset} to {sample_offset + sample_size}"
        )
        for entry in parse_gpmf(fp, sample_offset, sample_offset + sample_size):
            if not entry:
                continue

            # if not entry.fourcc.startswith(b"GPS"):
            #     continue

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
class GpsSample:
    """Based on WGS84"""

    fix: int = 0
    gps_timestamp: str = ""
    precision: int = 0
    # lat, long, alt, 2d speed, 3d speed
    data: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    units: List[str] = ""
    # latitude: float = 0.0
    # longitude: float = 0.0
    # altitude: float = 0.0
    # speed_2d: float = 0.0
    # speed_3d: float = 0.0


def get_gps_samples(fp: io.BufferedRandom):
    samples: List[GpsSample] = []
    for sample_idx, (sample_offset, sample_size) in enumerate(itersamples(fp)):
        print(f"----- sample {sample_idx}")
        # print_gpmf_streams_from(iter_gpmf_gps_streams(fp, sample_offset, sample_size))
        for _, entries in iter_gpmf_gps_streams(fp, sample_offset, sample_size):
            gps_sample = GpsSample()
            for entry in entries:
                print(entry)
                if entry.fourcc == b"GPSF":
                    gps_sample.fix = next(entry.iter_samples(fp))
                elif entry.fourcc == b"GPSU":
                    gps_sample.gps_timestamp = list(entry.iter_samples(fp))
                elif entry.fourcc == b"GPSP":
                    gps_sample.precision = next(entry.iter_samples(fp))
                elif entry.fourcc == b"GPS5":
                    gps_sample.data = list(entry.iter_samples(fp))
            samples.append(gps_sample)
            print(gps_sample)
    return samples


def main(file: Path):
    with open(str(file), "rb") as fp:
        # print_gpmf_samples(fp)
        get_gps_samples(fp)


main(Path(sys.argv[1]))
