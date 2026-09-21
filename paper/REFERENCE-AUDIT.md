# Reference audit — 21 September 2026

Scope: the 23 numbered references and their uses in `paper/unalloc.typ`, plus a targeted search for closely overlapping prior work. The manuscript, figures, data and PDFs were not changed. This is a source-support audit, not independent replication of the cited experiments or an exhaustive proof of novelty.

All 23 entries resolve to identifiable works or official projects. No fabricated title, author-work mismatch, or wrong arXiv identifier was found. Bibliographic existence is stronger than the support for several surrounding claims. The highest-priority correction is the treatment of OpenCost's existing capabilities.

## Changes recommended before submission

### 1. Acknowledge OpenCost's OpenAI plugin and inference accounting

Locations: Introduction; §10, “What this paper adds”; reference [11]. Also check README's claim that OpenCost structurally cannot see provider bills.

OpenCost announced its OpenAI plugin on **21 November 2024**, explicitly describing provider API costs alongside Kubernetes workloads. Its July 2026 inference integration discusses allocation versus usage costs, input/output token costs, cache effects and shared infrastructure. These are directly relevant antecedents. The README already acknowledges the inference integration, but the paper's related-work discussion does not.

Sources: [OpenAI plugin announcement](https://opencost.io/blog/Latest%20Updates%20-%20New%20OpenCost%20Plugins%20and%20%241%2C000%20incentive%20for%20Community%20Developers), [plugin documentation](https://opencost.io/docs/integrations/plugins/openai/), [inference accounting](https://opencost.io/blog/opencost-llmd-inference-cost/).

This does **not** establish that OpenCost already performs unalloc's exact four-source workflow, fallback reporting or experiments. It does invalidate the broad framing that bringing external AI costs alongside Kubernetes is unavailable in existing tooling. Documentation was inspected; the plugin was not executed.

Suggested replacement framing:

> OpenCost supports Kubernetes allocation, external-cost plugins including OpenAI, and inference-specific accounting. We build on this ecosystem with a lightweight ledger spanning OpenCost, LiteLLM and direct provider billing, and study how source overlap, incomplete reads and ownership metadata affect attribution. Our contribution is the implementation and reproducible analysis of these failure modes, rather than the general idea of consolidating cost data.

### 2. Add FOCUS to the normalized-ledger discussion

Locations: §2 and §10.

FOCUS is a directly relevant prior standard for normalizing billing records. Its specification also represents split-cost allocation. The paper should explain how its deliberately small `CostRow` relates to this work; do not imply standards compliance without a field-level mapping.

Sources: [FOCUS introduction](https://focus.finops.org/docs/specification/v1-4/sections/introduction/), [split-cost handling in v1.3](https://focus.finops.org/docs/specification/v1-3/attributes/data-generator-calculated-split-cost-allocation-handling/).

Suggested addition:

> FOCUS standardizes billing data across providers, including representations of split-cost allocation. Our internal CostRow is a smaller application-specific representation for ownership analysis; this work does not establish FOCUS conformance.

### 3. Bound the meaning of Shapley ground truth

Locations: §9 after the GPU comparison; §10, reference [15], with [19] as the mathematical foundation.

JouleShare measures a coalition function for **active GPU energy**, subtracting idle power, and evaluates an explicitly chosen Shapley allocation. That is not a unique ground truth for allocating a fixed monthly GPU rental bill or idle capacity. Its eight-request exhaustive protocol and the “roughly a quarter” summary are supported: normalized L1 values of 0.440 and 0.458 correspond to approximately 22.0% and 22.9% of energy reassigned between requests.

Source: [JouleShare, §§3 and 5](https://arxiv.org/html/2608.00026v1).

Suggested replacement for the future experiment sentence:

> Replaying tenant subsets could provide a measured Shapley reference for a specified cost function, such as active GPU energy. With four tenants there are 15 non-empty coalitions per load level, before repetitions and baseline measurements. Translating that reference into rental charges would still require an overhead-allocation policy.

### 4. Limit the claim that cached decoding stays flat

Location: §5.2, reference [7].

Pope et al. support KV caching and the roles of bandwidth, context and batching. They explicitly discuss increasing generation latency as the KV cache grows. They do not support a general claim that cached decoding is constant-time in context length. The locally measured 36.5× result remains the paper's own result.

Source: [Pope et al., §§2 and 4.2](https://arxiv.org/html/2211.05102v1).

Suggested wording:

> Cached decoding avoids recomputing prior tokens. In our measured 32–1,024-token CPU range, cached-step latency changes much less than full recomputation; it should not be interpreted as independent of context length in general.

### 5. Distinguish phase asymmetry from proof of cross-subsidy

Location: §5 paragraph following the headline, reference [10].

DistServe supports differing prefill/decode resource requirements and the benefit of separate scheduling. It does not establish a universal financial subsidy from prompt-heavy to decode-heavy tenants. That interpretation depends on workload, batching and the chosen reference meter.

Source: [DistServe, official OSDI 2024 publication](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin).

Suggested wording:

> In these workloads, token-based allocation assigns the prompt-heavy tenant a larger share than the timing-based meters. The phase asymmetry is consistent with the motivation for DistServe, while the size and direction of allocation differences depend on the workload and allocation policy.

### 6. Describe ABACUS as a design, and qualify the comparison

Location: §10, reference [22].

The abstract describes deployment blocking and predeployment cost awareness. In §4.3, OPA/Sentinel/Infracost integration is presented as an extension; §5 calls enforcement reactive and predictive methods future work. The current summary makes the maturity of these features sound more established than the body demonstrates. An absolute spending-budget gate also differs from unalloc's ownership-coverage gate.

Source: [ABACUS, §§4.3–5](https://arxiv.org/html/2501.14753v1).

Suggested wording:

> ABACUS proposes budget monitoring and enforcement and discusses integration with infrastructure-as-code cost checks. Our gate instead tests the fraction of observed spend lacking ownership metadata.

## Reference-by-reference results

“Supported” means the cited mechanism or description is supported at the level used in this paper, not that every claim in the cited work was independently verified.

| Ref. | Identity and source | Assessment / action |
| --- | --- | --- |
| 1 | Kwon et al., PagedAttention, SOSP 2023; [2309.06180](https://arxiv.org/abs/2309.06180) | Supported for paged KV storage and sharing. The 2023 paper does not document vLLM 0.29.0 behavior; cite a release or captured software record separately for that version. |
| 2 | Yu et al., Orca, OSDI 2022; [official proceedings](https://www.usenix.org/conference/osdi22/presentation/yu) | Supported for iteration-level scheduling and batching. Add this stable publication URL. |
| 3 | Agrawal et al., Sarathi-Serve, OSDI 2024; [official proceedings](https://www.usenix.org/conference/osdi24/presentation/agrawal) | Supported for chunked prefill and mixed scheduling; 2403.02310 matches. |
| 4 | Zheng et al., SGLang, NeurIPS 2024; [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) | Supported for RadixAttention and prefix reuse; 2312.07104 matches. |
| 5 | Shoeybi et al., Megatron-LM; [1909.08053](https://arxiv.org/abs/1909.08053) | Supported as the source of the tensor-partitioning pattern. It is a training paper; the inference adaptation and correctness tests are unalloc's own work. |
| 6 | Huang et al., GPipe, NeurIPS 2019; [official proceedings](https://proceedings.neurips.cc/paper/2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html) | Supported for pipeline model parallelism; 1811.06965 matches. Say “inspired by pipeline model parallelism” if the implementation is merely two stages with send/recv, rather than implying reproduction of GPipe's training schedule. |
| 7 | Pope et al., MLSys 2023; [official proceedings](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html) | Correct venue and identity; 2211.05102 matches. Qualify context-independent latency as discussed above. |
| 8 | Kaplan et al., Scaling Laws, 2020; [§2.1](https://arxiv.org/html/2001.08361v1) | Valid citation: Table 1 and Eq. 2.2 contain a forward-pass FLOP estimate, despite the paper's training focus. Specify that unalloc uses an adapted analytic estimate including context and LM-head terms; FLOPs are not measured latency. |
| 9 | Su et al., RoFormer; [2104.09864](https://arxiv.org/abs/2104.09864) | Supported for rotary position embeddings. The 2021 preprint citation is valid. |
| 10 | Zhong et al., DistServe, OSDI 2024; [official proceedings](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | Supported for phase asymmetry and disaggregation; 2401.09670 matches. Not proof of a universal subsidy. |
| 11 | OpenCost; [official project](https://opencost.io/) | Real and relevant, but homepage alone is inadequate for the capability comparison. Add plugin and inference sources above. |
| 12 | LiteLLM; [official repository](https://github.com/BerriAI/litellm) | Supported for provider gateway and cost tracking. Pin a release or retrieval date; the current README is evolving. |
| 13 | LeaderWorkerSet; [dual-template documentation](https://lws.sigs.k8s.io/docs/concepts/leaderworkerset/pod-templates/) | Supports separate leader/worker templates. If the leader template is omitted, the worker template also applies to the leader. The failure scenario is conditional, not inevitable. The [label reference](https://lws.sigs.k8s.io/docs/reference/labels-annotations-and-environment-variables/) also supports the LWS name key used in the scenario. |
| 14 | FinOps Allocation; [specific capability page](https://www.finops.org/framework/capabilities/allocation/) | Supported for ownership metadata and shared-cost policy. Replace the generic Framework URL with this page. |
| 15 | Luo et al., JouleShare; [2608.00026](https://arxiv.org/abs/2608.00026) | Authors, title and identifier match. Quantitative summary and exhaustive eight-request protocol checked in full text. Qualify the target as active-energy Shapley attribution. |
| 16 | Vellaisamy et al.; [2608.28044](https://arxiv.org/abs/2608.28044) | Summary of fixed/marginal energy and amortization is supported. The record now reports acceptance at IISWC 2026; optionally update venue while retaining the preprint identifier. Energy findings do not directly determine rental-dollar shares. |
| 17 | Jin et al., LLMVisor; [2608.08382](https://arxiv.org/abs/2608.08382) | Supported for latency decomposition, tenant aggregation and improvement over token baselines. Keep this distinct from measured energy or invoice reconciliation. |
| 18 | Wang and Buyya, PrefixShield; [2608.01657](https://arxiv.org/abs/2608.01657) | Materialization responsibility summary is supported. Add that the prototype isolates prefix lookup by accounting group; cross-group content reuse is outside its scope (§II-C). It does not solve how to divide a shared cached prefix's monetary cost among tenants. |
| 19 | Shapley, A Value for n-Person Games, 1953 | Identity and original publication supported by the [publisher's reprint record](https://doi.org/10.1515/9781400829156-012). Add original pages 307–317. Do not accidentally substitute the reprint's 2020 date for the original. |
| 20 | Ghodsi et al., DRF, NSDI 2011; [official program](https://www.usenix.org/legacy/event/nsdi11/tech/) | Supported as background for multi-resource fairness; not a monetary attribution ground truth. Add a stable proceedings link. |
| 21 | Schneider and Mattia; [2406.09645](https://arxiv.org/abs/2406.09645) | Title/authors match. Summary is supported; the method combines resource reservations and hourly usage, not only direct machine measurements. |
| 22 | Deochake, ABACUS; [2501.14753](https://arxiv.org/abs/2501.14753) | Identity matches. Qualify implementation maturity as above. Bibliographic date needs reconciliation: retrieved abstract/history and HTML display 22 Dec 2024, while the manuscript says 2025 and the identifier begins 2501. Check the canonical export before changing the year; do not infer it solely from the identifier. |
| 23 | Shukla, Cost-Governed RAG; [2607.12188](https://arxiv.org/abs/2607.12188) | Summary of embedding/retrieval/generation attribution and linear tenant memory is supported. It integrates TurboVec and GovLLM inside a common governance boundary; saying all prior work is “inside one system” is imprecise. Distinguish the governance boundary and data sources instead. |

## Editorial cleanup and limits

- Add hyperlinks for arXiv identifiers and official publication links for conference papers. Include retrieval dates and, where practical, versions for mutable software documentation.
- Keep recent preprints labeled as preprints unless their venue status is confirmed. Existence on arXiv is not evidence of peer review.
- Add a model citation or versioned model card for Qwen2.5-7B-Instruct when updating experiment provenance.
- Do not characterize gateway/provider overlap, incomplete pagination or missing labels as newly discovered general phenomena. The defensible claim is their concrete, reproducible characterization in this workflow.
- This targeted search identified relevant missing antecedents; it cannot prove that no other tool combines the exact sources. Prefer a positive description of the implemented contribution over an unbounded “none does this” claim.
- No GPU experiments were repeated, external systems modified, or manuscript edits made during this audit.
