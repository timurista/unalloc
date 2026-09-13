"""Distributed inference with torch.distributed, and why multi-pod serving leaks attribution.

A tiny decoder-only transformer is served three ways on CPU with the gloo backend:
single process, Megatron-style tensor parallelism (TP) and a two-stage pipeline
(PP). The runs establish two facts the cost story depends on:

1. Sharded serving is numerically the same model, so every pod in a replica is
   doing the same tenant's work, and none of them is optional overhead.
2. A measurable share of each rank's time is spent blocked in collectives or
   point-to-point transfers, and that share differs between rank roles and
   parallelism strategies.

Those measurements then drive a LeaderWorkerSet-shaped OpenCost month in which
the owner label lives only on the leader template, which is how the worker
pods' spend (and most of the communication cost) ends up unallocated.

Run from the repo root::

    python -m case_studies.distributed [--quick]
"""
