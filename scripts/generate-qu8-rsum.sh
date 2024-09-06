#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qu8-rsum/scalar.c.in -D CHANNEL_TILE=1  -D ACCUMULATORS=1 -o src/qu8-rsum/gen/qu8-rsum-scalar-u1.c &
tools/xngen src/qu8-rsum/scalar.c.in -D CHANNEL_TILE=2  -D ACCUMULATORS=1 -o src/qu8-rsum/gen/qu8-rsum-scalar-u2.c &
tools/xngen src/qu8-rsum/scalar.c.in -D CHANNEL_TILE=4  -D ACCUMULATORS=1 -o src/qu8-rsum/gen/qu8-rsum-scalar-u4.c &

################################## ARM NEON ###################################
tools/xngen src/qu8-rsum/neon.c.in   -D CHANNEL_TILE=16 -D ACCUMULATORS=1 -o src/qu8-rsum/gen/qu8-rsum-neon-u16.c &
tools/xngen src/qu8-rsum/neon.c.in   -D CHANNEL_TILE=32 -D ACCUMULATORS=1 -o src/qu8-rsum/gen/qu8-rsum-neon-u32.c &
tools/xngen src/qu8-rsum/neon.c.in   -D CHANNEL_TILE=32 -D ACCUMULATORS=2 -o src/qu8-rsum/gen/qu8-rsum-neon-u32-acc2.c &
tools/xngen src/qu8-rsum/neon.c.in   -D CHANNEL_TILE=64 -D ACCUMULATORS=1 -o src/qu8-rsum/gen/qu8-rsum-neon-u64.c &
tools/xngen src/qu8-rsum/neon.c.in   -D CHANNEL_TILE=64 -D ACCUMULATORS=2 -o src/qu8-rsum/gen/qu8-rsum-neon-u64-acc2.c &
tools/xngen src/qu8-rsum/neon.c.in   -D CHANNEL_TILE=64 -D ACCUMULATORS=4 -o src/qu8-rsum/gen/qu8-rsum-neon-u64-acc4.c &

################################### x86 SSE41 #################################
tools/xngen src/qu8-rsum/sse41.c.in -D ACCUMULATORS=1 -D CHANNEL_TILE=16  -o src/qu8-rsum/gen/qu8-rsum-sse41-u16.c &
tools/xngen src/qu8-rsum/sse41.c.in -D ACCUMULATORS=1 -D CHANNEL_TILE=32  -o src/qu8-rsum/gen/qu8-rsum-sse41-u32.c &
tools/xngen src/qu8-rsum/sse41.c.in -D ACCUMULATORS=1 -D CHANNEL_TILE=64  -o src/qu8-rsum/gen/qu8-rsum-sse41-u64.c &

wait
