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
| Comments | 13 pages, 9 figures, 4 tables. Includes a validation run with vLLM on an NVIDIA H100. Code, data and reproduction scripts: https://github.com/timurista/unalloc. Software: doi:10.5281/zenodo.22761013. Use of generative AI is disclosed in the paper ("Use of AI tools"). |
| Primary category | cs.DC (Distributed, Parallel, and Cluster Computing) |
| Cross-lists | cs.PF (Performance), cs.SE (Software Engineering) |
| MSC / ACM class | ACM: C.4 (Performance of Systems); K.6.2 (Installation Management — pricing and resource allocation) |
| License | Your choice at submission. CC BY 4.0 is the usual choice for open research and matches an Apache-2.0 codebase in spirit. |

## Before you submit
1. **Read the whole paper yourself.** arXiv holds every author fully responsible for all content, however it was produced. Check each number against `case_studies/results/*/metrics.json`, and check that every cited paper says what the text says it does.
2. **AI-use disclosure — where it goes.** arXiv wants the disclosure *in the paper itself*, not in a separate form field. It is already there: an unnumbered **"Use of AI tools"** section on **page 11**, after §12 Reproducibility and immediately before the References, naming Claude Code (Claude Opus 5), stating it is not an author, and stating that the author is responsible for all content. Read it and edit it if your own contribution differs from that description. It is also flagged in the Comments metadata above, so it shows on the abstract page. Do not put it in the abstract — that space is for results. If the submission form asks about generative-AI use, answer yes and point to that section.
3. ~~**Affiliation.**~~ Done 2026-09-15: `#let affiliation = "Independent Researcher"` in `paper/unalloc.typ`, matching the submitting arXiv account (user `timurista`, default category cs.DC).
4. **Endorsement.** First-time submitters to cs.DC are asked for an endorsement, and it is the long pole: endorsers reply on their own schedule and many never do. Start it first and let everything else run in parallel — the Zenodo DOI already makes the work public and citable. arXiv shows an endorsement code during submission; send it to individual authors who publish in cs.DC. Never mass-mail the request; arXiv treats that as abuse. Codes lapse, so regenerate if the wait runs long.
5. ~~**Zenodo, then the release.**~~ Done 2026-09-15: v0.2.1 is on PyPI and archived as doi:10.5281/zenodo.22761013 (all versions: doi:10.5281/zenodo.22761012). The DOI is already in the paper's title block, its Reproducibility section and the Comments field above.
6. **Category.** This is a systems and FinOps measurement paper, not an ML methods paper: keep cs.DC as primary.
