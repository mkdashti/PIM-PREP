#!/bin/bash

if [ "$#" -ne 5 ]; then
  echo "Usage : ./run-all-workers <data-dir> <out-dir> <num-gpu> <num-cpu> <classes>"
  exit 1
fi

DATA_DIR=$1
OUT_DIR=$2
GPU=$3
CPU=$4
CLASSES=$5

CPU_PER_GPU=$((CPU / GPU))

for arch in 'resnet18'; do
    for workers in $CPU_PER_GPU; do
         for num_gpu in $GPU; do
             python3 harness.py --nproc_per_node=$num_gpu -j $workers -b 128  -a $arch --prefix $OUT_DIR/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval --classes $CLASSES --data-profile $DATA_DIR
         done
    done
done

for arch in 'resnet18'; do
    for workers in $CPU_PER_GPU; do
         for num_gpu in $GPU; do
             python3 harness.py --nproc_per_node=$num_gpu -j $workers -b 128  -a $arch --prefix $OUT_DIR/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval --dali_cpu --classes $CLASSES --data-profile $DATA_DIR
         done
    done
done

