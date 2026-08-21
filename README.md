# MyMultiLLM

## Training profiles

- `gpt2-small`: 12 layers, 768 hidden width, 12 heads, 1024-token context; about 124M parameters.
- `dense-4b`: 28 layers, 4096 hidden width, 32 heads, 2048-token context; about 4B parameters.

`gpt2-small` is the realistic first milestone. GPT-2-level quality requires a large, legally usable English corpus and substantial GPU training time; it cannot be obtained from the bundled sample data.

Prepare token shards with a 50k-token English tokenizer, then launch a single-GPU/small-model run:

```bash
python -m training.launch --profile gpt2-small \
  --train-manifest data/shards/train_shards.json \
  --val-manifest data/shards/val_shards.json \
  --checkpoint-dir checkpoints/gpt2_small --max-steps 100000
```

For sharded multi-GPU pretraining, use CUDA and NCCL. A 4B model normally needs 4–8 GPUs with 80 GB VRAM:

```bash
torchrun --nproc_per_node=8 -m training.fsdp_train --profile dense-4b \
  --train-manifest data/shards/train_shards.json \
  --checkpoint-dir checkpoints/dense_4b --max-steps 100000
```

Chat enforces the identity: System is an English language assistant created by S. Arjun Ganesh.
