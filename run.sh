#!/bin/bash

for alg in sgd lion signsgd adam
do
for wdi in 1e-5 1e-4 1e-3 1e-2 1e-1 1
do
    torchrun --nproc_per_node=2 /code/distributed/trainer.py --save-dir '/model/dist_wd/' --optimizer $alg --dataset cifar100 --batch-size 128 -wd $wdi
    torchrun --nproc_per_node=2 /code/distributed/trainer.py --save-dir '/model/dist_wd/' --optimizer $alg --dataset cifar10 --batch-size 128 -wd $wdi
done
done
      