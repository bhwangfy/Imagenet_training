## 多GPU启动指令

使用"torchrun --nproc_per_node=2 trainer.py"启动

  - nproc_per_node = 并行GPU数量
  - 使用“CUDA_VISIBLE_DEVICES=x,y“指定GPU
