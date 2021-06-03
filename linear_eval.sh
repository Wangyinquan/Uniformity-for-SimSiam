#!/bin/bash
python linear_eval.py \
--data_dir /tmp/CIFAR/ \
--log_dir ./logs/ \
-c configs/simsiam_cifar_eval_sgd.yaml \
--ckpt_dir /tmp/ckpts/ \
--eval_from xxx\
--hide_progress
