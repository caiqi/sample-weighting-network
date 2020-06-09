#!/usr/bin/env bash
set -x
model_name=Retina_R101
./tools/dist_train.sh configs/${model_name}.py 4 --validate
./tools/dist_test.sh configs/${model_name}.py work_dirs/${model_name}/latest.pth 4 --out work_dirs/${model_name}/results.pkl --eval bbox --in_detail 1 >>work_dirs/${model_name}/results.log
