"""Case studies: inference workloads, simulated or run for real, fed through unalloc.

Each study lives in its own subpackage and is runnable from the repo root:

    python -m case_studies.kv_cache
    python -m case_studies.torch_kv
    python -m case_studies.distributed
    python -m case_studies.hybrid_e2e

Every study writes provider-shaped payloads (OpenCost /allocation, LiteLLM
/spend/logs, ...) into case_studies/results/<study>/ and then runs unalloc on
them exactly as a user would, so the case studies double as integration tests
of the adapter contract.
"""
