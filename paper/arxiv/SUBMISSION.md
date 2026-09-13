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
| Comments | 9 pages, 8 figures, 3 tables. Code, data and reproduction scripts: https://github.com/timurista/unalloc |
| Primary category | cs.DC (Distributed, Parallel, and Cluster Computing) |
| Cross-lists | cs.PF (Performance), cs.SE (Software Engineering) |
| MSC / ACM class | ACM: C.4 (Performance of Systems); K.6.2 (Installation Management — pricing and resource allocation) |
| License | Your choice at submission. CC BY 4.0 is the usual choice for open research and matches an Apache-2.0 codebase in spirit. |

## Before you submit
1. **Endorsement.** First-time submitters to cs.DC are often asked for an endorsement. arXiv shows an endorsement code during submission; send it to someone who has published in cs.DC, or add a co-author who has.
2. **Affiliation.** The title block lists no affiliation. Add one in `paper/unalloc.typ` if you have one ("Independent Researcher" is common).
3. **DOI for the code.** If the Zenodo DOI exists by then, add it to Comments: "Software: doi:10.5281/zenodo.NNNNNNN".
4. **Version.** Submit from a tagged release (v0.2.0) so the paper and the code it describes stay in sync.
