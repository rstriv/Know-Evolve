#!/bin/bash

DATA_ROOT=../../data/icews/
RESULT_ROOT=./

cur_iter=0
max_iter=3258
bptt=200
lr=0.0005
l2=0.00

embed_E=64
embed_R=32
hidden=64
warm=0
skip=0

t_scale=0.0001
w_scale=0.1
min_dur=24
max_dur=500

int_report=100
int_test=3258
int_save=3258

meta=$DATA_ROOT/stat_500.txt
train=$DATA_ROOT/train_500.txt
test=$DATA_ROOT/test_500.txt

save_dir=$RESULT_ROOT/E_$embed_E-R_$embed_R-H_$hidden

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

./build/main \
    -cur_iter $cur_iter \
    -max_iter $max_iter \
    -bptt $bptt \
    -lr $lr \
    -l2 $l2 \
    -embed_E $embed_E \
    -embed_R $embed_R \
    -hidden $hidden \
    -warm $warm \
    -t_scale $t_scale \
    -min_dur $min_dur \
    -max_dur $max_dur \
    -w_scale $w_scale \
    -int_report $int_report \
    -int_test $int_test \
    -int_save $int_save \
    -skip $skip \
    -meta $meta \
    -train $train \
    -test $test \
    -svdir $save_dir \
    2>&1 | tee $save_dir/log.txt


