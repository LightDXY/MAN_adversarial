#!/bin/bash

EPS=8
LR_POLICY=step
BATCH=128

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
NAME=cifar_ManR-$now
python -u cifar_manr.py \
	--dataroot /home/jfhan/attack_multi_target/cifar_10_png \
	--checkpoints_dir checkpoints \
	--name ${NAME} \
	--model 1 \
	--phase cifar \
	--no_dropout \
	--batchSize ${BATCH} \
	--print_freq 10 \
	--port 23459 \
	--workers 4 \
	--niter 0 \
	--niter_decay 200 \
	--loadSize 32 \
	--fineSize 32 \
	--max_epsilon ${EPS} \
	--save_epoch_freq 1 \
	--save_latest_freq 10000 \
	--set_class 10 \
	--div_coeff 1 \
	--lr 0.001 \
	--lr_policy=$LR_POLICY \
	--lr_decay_factor 0.1 \
	--lr_decay_iters 160 \
	--alpha 1 \
	--beta 1200 \
	2>&1 | tee log/train-${NAME}.log
