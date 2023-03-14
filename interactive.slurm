#!/bin/bash
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=42 --mem-per-cpu=3850 --gres=gpu:ampere_a100:3 --partition=gpu --time=40:00:00 --account=su004 --pty bash -i

#srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --mem-per-cpu=3850 --time=48:00:00 --account=su004 --pty bash -i

