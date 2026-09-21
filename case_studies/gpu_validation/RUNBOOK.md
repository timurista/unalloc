# GPU validation runbook (DigitalOcean)

How the vLLM validation run was brought up, verified and torn down. Every command below was run
for the published results; the captured output lives in
[`../results/gpu_validation/evidence/`](../results/gpu_validation/evidence/).

**Budget:** one `gpu-h100x1-80gb` droplet at $4.41/hour, capped at one hour. DigitalOcean bills
per second with a 5-minute minimum, and **a powered-off GPU droplet still bills**, so the only
way to stop charges is to delete it.

## 0. Prerequisites (local)

```bash
doctl auth list                     # an authenticated DigitalOcean context
doctl compute size list | grep gpu  # sizes and hourly prices on this account
doctl compute region list --output json \
  | jq -r '.[] | select(.sizes | index("gpu-h100x1-80gb")) | .slug'   # regions with capacity
doctl compute ssh-key import unalloc-gpu-validation --public-key-file ~/.ssh/id_ed25519.pub
```

Before spending anything, dry-run the benchmark client against a fake server:

```bash
python -m case_studies.gpu_validation.fake_server --port 8011 &
python -m case_studies.gpu_validation.bench --base http://127.0.0.1:8011 --model fake \
  --rates 3 --duration 12 --out /tmp/dryrun
kill %1
```

## 1. Create the droplet (evidence: `01_create.txt`)

```bash
doctl compute droplet create unalloc-gpu-validation-h100 \
  --region nyc2 --size gpu-h100x1-80gb --image gpu-h100x1-base \
  --ssh-keys <key-id> --tag-names unalloc-gpu-validation --wait
```

`gpu-h100x1-base` is DigitalOcean's "NVIDIA AI/ML Ready" Ubuntu image: driver, CUDA toolkit
and container runtime preinstalled.

**Arm a watchdog immediately.** It deletes anything carrying the tag after 55 minutes,
whatever else happens:

```bash
( sleep 3300; doctl compute droplet delete --tag-name unalloc-gpu-validation --force ) &
```

## 2. Verify the hardware (evidence: `02_machine.txt`, `droplet.json`)

```bash
doctl compute droplet get <id> --output json > droplet.json
ssh root@<ip> 'nvidia-smi; nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv;
               lscpu; free -g; df -h /; cat /etc/os-release'
ping -c 10 <ip>     # operator-to-droplet round trip; not part of serving latency
```

## 3. Install (evidence: `03_setup.txt`)

```bash
rsync -az --exclude .git --exclude case_studies/results --exclude paper --exclude blog ./ root@<ip>:/root/unalloc/
ssh root@<ip> 'bash /root/unalloc/case_studies/gpu_validation/remote.sh setup'
```

`setup` installs `uv`, creates a Python 3.12 environment, installs vLLM and this repository,
and downloads `Qwen/Qwen2.5-7B-Instruct` (about 15 GB) in parallel with the install.

## 4. Serve (evidence: `04_serve.txt`)

```bash
ssh root@<ip> 'bash /root/unalloc/case_studies/gpu_validation/remote.sh serve'
```

This starts per-second `nvidia-smi` logging and `vllm serve` with prefix caching and
cached-token reporting enabled, then polls `/health`.

**What went wrong the first time.** vLLM 0.29.0's default sampler JIT-compiles a FlashInfer
kernel during warm-up. On this image that compile failed and the engine exited. Setting
`VLLM_USE_FLASHINFER_SAMPLER=0` switches to the PyTorch sampler; `remote.sh` now sets it by
default. The failed log is kept as `raw/vllm_attempt1_failed.log`.

## 5. Benchmark (evidence: `05_bench.txt`)

```bash
ssh root@<ip> 'bash /root/unalloc/case_studies/gpu_validation/remote.sh bench --rates 2 4 8 16 --duration 120'
```

The client runs **on the droplet**, next to the server, so latency figures measure serving,
not the internet. It aborts before the sweep if the warm-up sees any error.

### Repeated runs (recommended for any new campaign)

The published dataset is one run per load level, which gives no run-to-run error bar — the
single largest weakness a reviewer will name. `--repeats` fixes that:

```bash
ssh root@<ip> 'bash /root/unalloc/case_studies/gpu_validation/remote.sh bench \
  --rates 2 4 8 16 --duration 120 --repeats 5'
```

Each repeat draws a fresh seed, so the arrival process and prompt mix differ between runs
rather than replaying identical traffic. The load levels are the **inner** loop, so anything
that drifts over the session — clock throttling, cache state, a noisy neighbour — spreads
across all four loads instead of landing on one.

Budget it before you start. Benchmark time is `rates x duration x repeats` plus about 20 s of
warm-up and inter-run settling per run:

| repeats | benchmark time | droplet time incl. bring-up | approx. cost at $4.41/h |
| --- | --- | --- | --- |
| 1 (published) | 8 min | ~29 min | $2.15 |
| 3 | 24 min | ~45 min | $3.31 |
| 5 | 40 min | ~61 min | $4.48 |

`analyze.py` groups `rate_<r>_r<n>.json` files by rate automatically and reports, per load, the
median and range across repeats alongside the within-run window statistics. No flag needed; the
single-run layout (`rate_<r>.json`) still works unchanged.

## 6. Copy results back

```bash
rsync -az root@<ip>:/root/results/ case_studies/results/gpu_validation/raw/
```

## 7. Tear down and verify (evidence: `07_teardown.txt`)

```bash
doctl compute droplet delete --tag-name unalloc-gpu-validation --force
doctl compute droplet list --tag-name unalloc-gpu-validation      # must be empty
doctl compute droplet get <id>                                    # must return 404
kill <watchdog-pid>
```

Check the billing page (or `doctl balance get`) the next day for the final charge.

## 8. Analyze (local, no GPU)

```bash
python -m case_studies.gpu_validation.analyze
```

Writes `case_studies/results/gpu_validation/metrics.json`: latency (TTFT, time per output
token, end-to-end) per tenant and load, the four metering rules computed from real telemetry,
GPU utilization and power, and the serving simulator run at the same rates for comparison.
