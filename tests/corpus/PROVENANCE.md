# Corpus provenance & licensing

Every file under `tests/corpus/` is **original content authored for the indx test suite**
and is released under the same Apache-2.0 license as the project (NFR-LIC-1). No third-party
or scraped documents are included, so the corpus may be redistributed with the repository.

| Corpus | Bytes | Source | License |
|---|---|---|---|
| `nested_handbook/` | text (`.md`) | authored for indx | Apache-2.0 |
| `split_report/` | text (`.txt`) | authored for indx | Apache-2.0 |
| `acme_kb/` | text (`.md`, `.txt`, `.rst`) | authored for indx (deep, mixed-format company KB) | Apache-2.0 |
| `airgap_smoke/` | text (`.md`) | generated for indx | Apache-2.0 |
| `ocr/sample.png` | image (`.png`) | rendered for indx (DejaVu text on white) | Apache-2.0 |

## Heavy binaries (`_heavy/`)

Multi-MB PDF/DOCX fixtures, when added, are tracked with **Git LFS** and checksummed in
`heavy.sha256`. Tests that consume them **skip** (not fail) when the LFS objects are absent,
so a lightweight clone still runs the text corpus (testing-strategy §3.3). Each heavy file
must be license-clean (CC0 / public-domain / authored here); record its source in this table.
