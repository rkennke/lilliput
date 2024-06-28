/*
 * Copyright (c) 2016, 2018, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 *
 */

#include "precompiled.hpp"
#include "gc/g1/g1CollectedHeap.inline.hpp"
#include "gc/g1/g1ConcurrentMark.inline.hpp"
#include "gc/g1/g1ConcurrentMarkObjArrayProcessor.inline.hpp"
#include "gc/g1/g1HeapRegion.inline.hpp"
#include "gc/shared/gc_globals.hpp"
#include "memory/memRegion.hpp"
#include "utilities/globalDefinitions.hpp"

size_t G1CMObjArrayProcessor::process_obj(oop obj) {
  assert(should_be_sliced(obj), "Must be an array object %d and large " SIZE_FORMAT, obj->is_objArray(), obj->size());

  assert(obj->is_objArray(), "expect object array");
  objArrayOop array = objArrayOop(obj);

  _task->scan_objArray_start(array);

  int len = array->length();

  int bits = log2i_graceful(len);
  // Compensate for non-power-of-two arrays, cover the array in excess:
  if (len != (1 << bits)) bits++;

  // Only allow full chunks on the queue. This frees do_chunked_array() from checking from/to
  // boundaries against array->length(), touching the array header on every chunk.
  //
  // To do this, we cut the prefix in full-sized chunks, and submit them on the queue.
  // If the array is not divided in chunk sizes, then there would be an irregular tail,
  // which we will process separately.

  int last_idx = 0;

  int chunk = 1;
  int pow = bits;

  // Handle overflow
  if (pow >= 31) {
    assert (pow == 31, "sanity");
    pow--;
    chunk = 2;
    last_idx = (1 << pow);
    _task->push(G1TaskQueueEntry(array, 1, pow));
  }

  // Split out tasks, as suggested in G1TaskQueueEntry docs. Record the last
  // successful right boundary to figure out the irregular tail.
  while ((1 << pow) > (int)ObjArrayMarkingStride &&
         (chunk*2 < G1TaskQueueEntry::chunk_size())) {
    pow--;
    int left_chunk = chunk * 2 - 1;
    int right_chunk = chunk * 2;
    int left_chunk_end = left_chunk * (1 << pow);
    if (left_chunk_end < len) {
      _task->push(G1TaskQueueEntry(array, left_chunk, pow));
      chunk = right_chunk;
      last_idx = left_chunk_end;
    } else {
      chunk = left_chunk;
    }
  }

  // Process the irregular tail, if present
  int from = last_idx;
  if (from < len) {
    return _task->scan_objArray(array, from, len);
  }
  return 0;
}

size_t G1CMObjArrayProcessor::process_slice(oop obj, int chunk, int pow) {

  assert(obj->is_objArray(), "expect object array");
  objArrayOop array = objArrayOop(obj);

  assert (ObjArrayMarkingStride > 0, "sanity");

  // Split out tasks, as suggested in ShenandoahMarkTask docs. Avoid pushing tasks that
  // are known to start beyond the array.
  while ((1 << pow) > (int)ObjArrayMarkingStride && (chunk*2 < G1TaskQueueEntry::chunk_size())) {
    pow--;
    chunk *= 2;
    _task->push(G1TaskQueueEntry(array, chunk - 1, pow));
  }

  int chunk_size = 1 << pow;

  int from = (chunk - 1) * chunk_size;
  int to = chunk * chunk_size;

#ifdef ASSERT
  int len = array->length();
  assert (0 <= from && from < len, "from is sane: %d/%d", from, len);
  assert (0 < to && to <= len, "to is sane: %d/%d", to, len);
#endif

  return _task->scan_objArray(array, from, to);
}
