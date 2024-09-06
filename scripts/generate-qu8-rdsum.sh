#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qu8-rdsum/scalar.c.in -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-scalar.c &

################################## ARM NEON ###################################
tools/xngen src/qu8-rdsum/neon.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-neon-c16.c &
tools/xngen src/qu8-rdsum/neon.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-neon-c32.c &
tools/xngen src/qu8-rdsum/neon.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-neon-c64.c &

################################### x86 SSE41 ###################################
tools/xngen src/qu8-rdsum/sse41.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-sse41-c16.c &
tools/xngen src/qu8-rdsum/sse41.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-sse41-c32.c &
tools/xngen src/qu8-rdsum/sse41.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/qu8-rdsum/gen/qu8-rdsum-7p7x-sse41-c64.c &

wait
