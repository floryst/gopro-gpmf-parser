// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

import { GpsSample } from '../go-pro/gps-sample.js';


export class GpsFrame {
  bb: flatbuffers.ByteBuffer|null = null;
  bb_pos = 0;
  __init(i:number, bb:flatbuffers.ByteBuffer):GpsFrame {
  this.bb_pos = i;
  this.bb = bb;
  return this;
}

static getRootAsGpsFrame(bb:flatbuffers.ByteBuffer, obj?:GpsFrame):GpsFrame {
  return (obj || new GpsFrame()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
}

static getSizePrefixedRootAsGpsFrame(bb:flatbuffers.ByteBuffer, obj?:GpsFrame):GpsFrame {
  bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
  return (obj || new GpsFrame()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
}

fix():number {
  const offset = this.bb!.__offset(this.bb_pos, 4);
  return offset ? this.bb!.readUint8(this.bb_pos + offset) : 0;
}

precision():number {
  const offset = this.bb!.__offset(this.bb_pos, 6);
  return offset ? this.bb!.readUint16(this.bb_pos + offset) : 0;
}

data(index: number, obj?:GpsSample):GpsSample|null {
  const offset = this.bb!.__offset(this.bb_pos, 8);
  return offset ? (obj || new GpsSample()).__init(this.bb!.__vector(this.bb_pos + offset) + index * 20, this.bb!) : null;
}

dataLength():number {
  const offset = this.bb!.__offset(this.bb_pos, 8);
  return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
}

static startGpsFrame(builder:flatbuffers.Builder) {
  builder.startObject(3);
}

static addFix(builder:flatbuffers.Builder, fix:number) {
  builder.addFieldInt8(0, fix, 0);
}

static addPrecision(builder:flatbuffers.Builder, precision:number) {
  builder.addFieldInt16(1, precision, 0);
}

static addData(builder:flatbuffers.Builder, dataOffset:flatbuffers.Offset) {
  builder.addFieldOffset(2, dataOffset, 0);
}

static startDataVector(builder:flatbuffers.Builder, numElems:number) {
  builder.startVector(20, numElems, 4);
}

static endGpsFrame(builder:flatbuffers.Builder):flatbuffers.Offset {
  const offset = builder.endObject();
  return offset;
}

static createGpsFrame(builder:flatbuffers.Builder, fix:number, precision:number, dataOffset:flatbuffers.Offset):flatbuffers.Offset {
  GpsFrame.startGpsFrame(builder);
  GpsFrame.addFix(builder, fix);
  GpsFrame.addPrecision(builder, precision);
  GpsFrame.addData(builder, dataOffset);
  return GpsFrame.endGpsFrame(builder);
}
}
