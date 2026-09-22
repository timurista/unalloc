# arXiv submission kit

**Submitted.** [arXiv:2609.24991](https://arxiv.org/abs/2609.24991) (cs.DC, cross-listed cs.PF),
v1 announced 21 September 2026, [doi:10.48550/arXiv.2609.24991](https://doi.org/10.48550/arXiv.2609.24991).
It is a preprint: arXiv moderates submissions, it does not peer review them.

Keep this file for the next version. arXiv replaces a paper in place under the same identifier, so
a v2 reuses the metadata below; only the PDF and the Comments field change.

## Files
- **PDF:** `paper/unalloc-case-studies.pdf`, produced by Typst. arXiv accepts PDF-only submissions when the PDF is not generated from TeX. Confirm against arXiv's current submission guidelines before uploading.
- Rebuild first if anything changed: `make paper`.
- **The page count, figure and table counts and the disclosure page above are read off the built PDF.** They shift when the text does; `tests/test_paper_numbers.py` fails if this file and the PDF disagree.

## Metadata

| Field | Value |
| --- | --- |
| Title | Who Pays for the KV Cache? Attributing Shared AI Inference Spend Across Kubernetes and LLM Provider Bills |
| Authors | Timothy Urista |
| Abstract | contents of `abstract.txt` (plain text, under arXiv's 1,920-character limit) |
| Comments | 14 pages, 9 figures, 4 tables. Includes a validation run with vLLM on an NVIDIA H100. Code, data and reproduction scripts: https://github.com/timurista/unalloc. Software: doi:10.5281/zenodo.22761012. Use of generative AI is disclosed in the paper ("Use of AI tools"). |
| Primary category | cs.DC (Distributed, Parallel, and Cluster Computing) |
| Cross-lists | cs.PF (Performance), cs.SE (Software Engineering) |
| MSC / ACM class | ACM: C.4 (Performance of Systems); K.6.2 (Installation Management — pricing and resource allocation) |
| License | Your choice at submission. CC BY 4.0 is the usual choice for open research and matches an Apache-2.0 codebase in spirit. |

## Before you submit
1. **Read the whole paper yourself.** arXiv holds every author fully responsible for all content, however it was produced. Check each number against `case_studies/results/*/metrics.json`, and check that every cited paper says what the text says it does.
2. **AI-use disclosure — where it goes.** arXiv wants the disclosure *in the paper itself*, not in a separate form field. It is already there: an unnumbered **"Use of AI tools"** section on **page 13**, after §12 Reproducibility and immediately before the References, naming Claude Code (Claude Opus 5), stating it is not an author, and stating that the author is responsible for all content. Read it and edit it if your own contribution differs from that description. It is also flagged in the Comments metadata above, so it shows on the abstract page. Do not put it in the abstract — that space is for results. If the submission form asks about generative-AI use, answer yes and point to that section.
3. ~~**Affiliation.**~~ Done 2026-09-15: `#let affiliation = "Independent Researcher"` in `paper/unalloc.typ`, matching the submitting arXiv account (user `timurista`, default category cs.DC).
4. ~~**Endorsement.**~~ Done: v1 is announced in cs.DC. Later submissions to the same category do not need a fresh endorsement.
5. ~~**Zenodo, then the release.**~~ Done. The paper, README and Comments field all cite the **concept DOI `10.5281/zenodo.22761012`**, which always resolves to the current release, so no DOI needs editing when a new version ships. Each release is also on PyPI. Cut and archive the release that matches the PDF you upload, so the archive contains the code the paper describes.
6. **Category.** This is a systems and FinOps measurement paper, not an ML methods paper: keep cs.DC as primary.

## After announcement

1. **Check what the listing says.** The abstract page renders the Comments field verbatim, including the code and DOI links. Fix anything wrong with a metadata-only update rather than a new version.
2. **Cite it consistently.** `CITATION.cff` carries the preprint as `preferred-citation`, so GitHub's "Cite this repository" button produces it. The README, the project site and timurista.ai all point at the abstract page.
3. **Call it a preprint.** Public and citable is not peer reviewed, and a reader who finds out otherwise stops trusting the rest of the page.
4. **Replacing it later.** `make paper`, re-run the tests (they check the page count, figure and table counts, the DOI and the abstract against the committed artifacts), then upload as a new version under the same identifier. Cut a matching release so the archived code still matches the PDF.
