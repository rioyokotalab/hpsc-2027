# Lecture 7: Distributed training strategies

What to split when the training state does not fit in one GPU, and what each split must communicate,
measured on the four H100s of one TSUBAME4.0 node: collectives, ZeRO/FSDP, tensor, pipeline and context parallelism.
The Python programs use the environment of Lecture 3 (`~/hpsc-venv`, see `../03_python/README.md`).

## Run

```
cd 07_distributed
qsub -g <group> job.sh                                         # Exercises 1-4 and the ring-attention demo (one node, about 8 min)
qsub -g <group> job.sh torchrun --standalone --nproc_per_node=4 10_memory.py fsdp 4096 32 32 4   # one other size
qsub -g <group> job.sh python check_train.py 50_train.py       # Homework: the template, for the baseline time
qsub -g <group> job.sh python check_train.py my_train.py       # Homework: your version
```

| File | What it shows |
| --- | --- |
| `00_collectives.py` | all-reduce, reduce-scatter, all-gather, all-to-all, broadcast and send/recv on 4 GPUs: time and bus bandwidth, 64 KiB to 1 GiB (Exercise 1) |
| `gpt.py` | the GPT used by Lectures 6 and 7, at any width and depth |
| `10_memory.py` | memory per GPU and step time of DDP, ZeRO-1, FSDP and FSDP without resharding, for a 1.5B and a 6.7B GPT (Exercise 2) |
| `20_tensor_parallel.py` | a tensor-parallel MLP (columns, then rows, one all-reduce) against one GPU (Exercise 3) |
| `30_pipeline.py` | a 4-stage pipeline, GPipe and 1F1B, step time and memory for 1 to 32 micro-batches (Exercise 4) |
| `40_ring_attention.py` | ring attention: keys and values travel around the 4 GPUs, checked against attention on one GPU |
| `50_train.py`, `check_train.py` | Homework: train a 1.5B GPT on one node, faster, with the same losses |
| `ref_loss.txt` | losses of the template on one node, used by `check_train.py` |
| `job_output_example.txt` | output of `job.sh` on one node, if your own job is still in the queue |
