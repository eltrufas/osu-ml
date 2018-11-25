#!/usr/bin/env bash
set -xe

export N_PROCS=32

mkdir -p work examples

python scripts/split_work.py

for i in $(seq 0 $N_PROCS); do
  python scripts/do_work.py $i &
done

wait

python scripts/combine_examples.py


