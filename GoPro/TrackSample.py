# automatically generated by the FlatBuffers compiler, do not modify

# namespace: GoPro

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TrackSample(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TrackSample()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTrackSample(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # TrackSample
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TrackSample
    def DecodingTime(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # TrackSample
    def Gps(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from GoPro.GpsFrame import GpsFrame
            obj = GpsFrame()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def TrackSampleStart(builder):
    builder.StartObject(2)

def Start(builder):
    TrackSampleStart(builder)

def TrackSampleAddDecodingTime(builder, decodingTime):
    builder.PrependUint32Slot(0, decodingTime, 0)

def AddDecodingTime(builder, decodingTime):
    TrackSampleAddDecodingTime(builder, decodingTime)

def TrackSampleAddGps(builder, gps):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(gps), 0)

def AddGps(builder, gps):
    TrackSampleAddGps(builder, gps)

def TrackSampleEnd(builder):
    return builder.EndObject()

def End(builder):
    return TrackSampleEnd(builder)