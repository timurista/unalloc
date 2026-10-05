# Bundled fixtures

Synthetic payloads for `--fixtures`. They describe one month (August 2026) of a
made-up organization and are built so that **no dollar appears in two sources**:

| Source | Covers |
| --- | --- |
| `opencost_allocation.json` | Self-hosted GPU and CPU workloads in the cluster (vLLM, llm-d, batch jobs, idle). |
| `litellm_spend.json` | Gateway traffic routed to Azure OpenAI, Vertex AI and Bedrock. unalloc has no direct adapter for those, so the gateway is their only record. |
| `openai_costs.json` | Direct OpenAI API usage that does not go through the gateway. |
| `anthropic_cost_report.json` | Direct Anthropic API usage that does not go through the gateway. |
| `invoices.json` | What OpenAI and Anthropic billed for the month, used by `reconcile --fixtures`. |

Real deployments rarely split this cleanly. If your gateway proxies calls to
OpenAI or Anthropic, pulling the gateway and the direct provider sources together
counts that spend twice; choose sources with `--source`.
