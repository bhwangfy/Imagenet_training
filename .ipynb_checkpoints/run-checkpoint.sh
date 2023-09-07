#!/bin/bash

for seedi in 1 2 3
do
    python -u /code/resnet/trainer.py --dataset cifar10 --optimizer=$optimi -hp 0.93 -sd=$seedi --save-dir /model/cifar10_resnet18_da_nwd_150 
done

 
    
 
      