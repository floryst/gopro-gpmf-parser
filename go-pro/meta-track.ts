// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

import { TrackSample } from '../go-pro/track-sample.js';


export class MetaTrack {
  bb: flatbuffers.ByteBuffer|null = null;
  bb_pos = 0;
  __init(i:number, bb:flatbuffers.ByteBuffer):MetaTrack {
  this.bb_pos = i;
  this.bb = bb;
  return this;
}

static getRootAsMetaTrack(bb:flatbuffers.ByteBuffer, obj?:MetaTrack):MetaTrack {
  return (obj || new MetaTrack()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
}

static getSizePrefixedRootAsMetaTrack(bb:flatbuffers.ByteBuffer, obj?:MetaTrack):MetaTrack {
  bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
  return (obj || new MetaTrack()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
}

samples(index: number, obj?:TrackSample):TrackSample|null {
  const offset = this.bb!.__offset(this.bb_pos, 4);
  return offset ? (obj || new TrackSample()).__init(this.bb!.__indirect(this.bb!.__vector(this.bb_pos + offset) + index * 4), this.bb!) : null;
}

samplesLength():number {
  const offset = this.bb!.__offset(this.bb_pos, 4);
  return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
}

static startMetaTrack(builder:flatbuffers.Builder) {
  builder.startObject(1);
}

static addSamples(builder:flatbuffers.Builder, samplesOffset:flatbuffers.Offset) {
  builder.addFieldOffset(0, samplesOffset, 0);
}

static createSamplesVector(builder:flatbuffers.Builder, data:flatbuffers.Offset[]):flatbuffers.Offset {
  builder.startVector(4, data.length, 4);
  for (let i = data.length - 1; i >= 0; i--) {
    builder.addOffset(data[i]!);
  }
  return builder.endVector();
}

static startSamplesVector(builder:flatbuffers.Builder, numElems:number) {
  builder.startVector(4, numElems, 4);
}

static endMetaTrack(builder:flatbuffers.Builder):flatbuffers.Offset {
  const offset = builder.endObject();
  return offset;
}

static finishMetaTrackBuffer(builder:flatbuffers.Builder, offset:flatbuffers.Offset) {
  builder.finish(offset);
}

static finishSizePrefixedMetaTrackBuffer(builder:flatbuffers.Builder, offset:flatbuffers.Offset) {
  builder.finish(offset, undefined, true);
}

static createMetaTrack(builder:flatbuffers.Builder, samplesOffset:flatbuffers.Offset):flatbuffers.Offset {
  MetaTrack.startMetaTrack(builder);
  MetaTrack.addSamples(builder, samplesOffset);
  return MetaTrack.endMetaTrack(builder);
}
}
