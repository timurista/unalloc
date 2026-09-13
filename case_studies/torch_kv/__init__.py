"""Real transformer inference with a KV cache, and what it does to cost attribution.

A tiny decoder-only transformer (random weights, real compute) serves a
multi-tenant request trace. The deployment's GPU bill is then apportioned to
callers by request count, tokens, measured compute time and KV-cache
memory-time, and every split is fed through unalloc as OpenCost allocations.

    python -m case_studies.torch_kv [--quick]
"""
