# arXiv submission kit

Everything arXiv's submission form asks for, ready to paste.

## Files
- **PDF:** `paper/unalloc-case-studies.pdf`, produced by Typst. arXiv accepts PDF-only submissions when the PDF is not generated from TeX. Confirm against arXiv's current submission guidelines before uploading.
- Rebuild first if anything changed: `make paper`.

## Metadata

| Field | Value |
| --- | --- |
| Title | Who Pays for the KV Cache? Attributing Shared AI Inference Spend Across Kubernetes and LLM Provider Bills |
| Authors | Timothy Urista |
| Abstract | contents of `abstract.txt` (plain text, under arXiv's 1,920-character limit) |
| Comments | 13 pages, 9 figures, 4 tables. Includes a validation run with vLLM on an NVIDIA H100. Code, data and reproduction scripts: https://github.com/timurista/unalloc |
| Primary category | cs.DC (Distributed, Parallel, and Cluster Computing) |
| Cross-lists | cs.PF (Performance), cs.SE (Software Engineering) |
| MSC / ACM class | ACM: C.4 (Performance of Systems); K.6.2 (Installation Management — pricing and resource allocation) |
| License | Your choice at submission. CC BY 4.0 is the usual choice for open research and matches an Apache-2.0 codebase in spirit. |

## Before you submit
1. **Read the whole paper yourself.** arXiv holds every author fully responsible for all content, however it was produced. Check each number against `case_studies/results/*/metrics.json`, and check that every cited paper says what the text says it does.
2. **AI-use disclosure.** Required by arXiv for significant use of generative AI. The paper's "Use of AI tools" section states what was produced with Claude Code; edit it if your own contribution differs from that description.
3. **Affiliation.** Set `#let affiliation = ...` at the top of `paper/unalloc.typ` ("Independent Researcher" is common), then `make paper`.
4. **Endorsement.** First-time submitters to cs.DC are often asked for an endorsement. arXiv shows an endorsement code during submission; send it to someone who has published in cs.DC, or add a co-author who has.
5. **Zenodo, then the release.** Enable the Zenodo GitHub integration first, then publish the GitHub Release for `v0.2.1`. That publishes to PyPI and mints the DOI; add it to Comments as "Software: doi:10.5281/zenodo.NNNNNNN".
6. **Category.** This is a systems and FinOps measurement paper, not an ML methods paper: keep cs.DC as primary.
