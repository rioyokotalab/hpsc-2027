# Lecture 8: Interconnect and scaling measurement

What leaving the node costs: one message, an all-reduce and DDP training of a 1.5B GPT,
measured on two TSUBAME4.0 nodes (8 H100s, four InfiniBand NDR200 NICs per node).
GPU time is shared in this lecture: the job below runs once for the whole class, and its output is
`job_output_example.txt`. The exercises and the homework use that file; you do not need to submit a job.
The Python programs use the environment of Lecture 3 (`~/hpsc-venv`, see `../03_python/README.md`).

## Run

```
cd 08_scaling
qsub -g <group> job.sh                 # the class run: two nodes, about 10 min (TA only)
python check_model.py 30_model.py      # Homework: no GPU, runs anywhere
```

| File | What it shows |
| --- | --- |
| `dist_env.py` | sets RANK, WORLD_SIZE and LOCAL_RANK from `mpirun`, picks this rank's GPU |
| `00_pingpong.py` | one message between two GPUs, inside a node and between nodes: latency α and bandwidth β (Exercise 1) |
| `10_allreduce.py` | all-reduce on 8 GPUs in two nodes and on subsets (one node; 1 + 1, 2 + 2 GPUs), 16 B to 1 GiB; with `all`, 8 GPUs only (Exercise 2) |
| `20_scaling.py` | DDP step, the same step without communication (`no_sync`) and one all-reduce of all gradients, on 1, 2, 4, 8 GPUs (Exercise 3) |
| `gpt.py` | the GPT of Lectures 6 and 7 |
| `job.sh` | the class run, with NCCL's ring, tree and a single NIC |
| `30_model.py`, `check_model.py` | Homework: a model of the DDP step, checked against the class run, then extended to 16 and 64 GPUs |
| `job_output_example.txt` | output of `job.sh` on two nodes |

## Homework

Copy `30_model.py` to `my_model.py` and fit it to `job_output_example.txt`. `python check_model.py my_model.py`
passes when your step time is within 15 % of every measured point. Then write `homework.md` with your fitted
parameters, the table of predicted vs measured steps, a table for 16 and 64 GPUs (step, tokens/s, GPU-hours
per 10⁹ tokens) and one paragraph: which assumption of your model you would test first, and how.
