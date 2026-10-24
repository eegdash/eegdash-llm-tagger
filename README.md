# EEGDash LLM Tagger

A Python library and command-line tools for proposing dataset-level labels from
BIDS electrophysiology documentation. Version 2 of the vocabulary separates
clinical population, sensory stimulus modality and study type. Returned labels
are checked against that vocabulary, while evidence and request provenance are
retained for curator review. Vocabulary membership and model confidence do not
establish factual correctness; verify source documentation before selecting a
clinical cohort.

## Installation

Python 3.11 or later is required. A normal installation includes the prompt,
15 labelled examples and the three command-line entry points:

```bash
git clone https://github.com/eegdash/eegdash-llm-tagger.git
cd eegdash-llm-tagger
python -m pip install .
```

For a reproducible installation, check out the published release tag or the exact
commit identified with that release before installing. No PyPI publication is
implied by these source-install instructions. Building a wheel uses the single
`pyproject.toml` configuration:

```bash
python -m pip install build
python -m build
python -m pip install dist/eegdash_tagger-0.1.1-py3-none-any.whl
```

Git and network access are needed by the remote scraping tools. A model endpoint
key is needed for tagging, and a GitHub token is needed when using the GitHub API
route. Parsing local metadata and running the offline unit suite need neither.

## Parse and tag a dataset

```python
from eegdash_tagger.metadata import build_dataset_summary_from_path
from eegdash_tagger.tagging import OpenRouterTagger, TaggingError

summary = build_dataset_summary_from_path("/data/ds001971")
print(summary.to_llm_json())  # Inspect the metadata before submitting it.
tagger = OpenRouterTagger(model="openai/gpt-5.2")
try:
    record = tagger.tag_with_details(summary.to_llm_json(), dataset_id="ds001971")
except TaggingError as exc:
    print(f"Tagging failed: {exc}")
else:
    for axis in ("pathology", "modality", "type"):
        print(axis, record[axis], record["confidence"][axis])
```

Set `OPENROUTER_API_KEY` or `LLM_API_KEY` in the process environment. To load a local
`.env` file explicitly, call `dotenv.load_dotenv()` before constructing the tagger.
`LLM_API_BASE` selects another OpenAI-compatible chat endpoint; it is read when
an instance is constructed. `LLM_TIMEOUT` controls the HTTP timeout in seconds
(default 300). `max_tokens` defaults to 4,000. Endpoint/model capabilities differ;
JSON-mode rejection and authentication failures are errors, not successful tags.

`tag()` and `tag_with_details()` raise `TaggingError` on HTTP/transport failures,
invalid response envelopes, malformed JSON, missing label axes, or missing or
nonfinite/out-of-range confidence scores. This is a behavior change in 0.1.1:
callers must handle errors and must not persist or cache them as classifications.
A valid response containing `Unknown`, including zero confidence, is still a
successful abstention. Vocabulary aliases and invalid strings are normalized by
`tagging/taxonomy.py`; this is distinct from response-shape validation. The full
reasoning mapping is retained, including JSON arrays/objects. Its semantic
support for the labels is not verified.

The default resources come from the installed wheel. A source/editable checkout
uses the canonical `prompt.md` and `data/processed/few_shot_examples.json` files;
the wheel includes their exact bytes. Explicit `prompt_path` and `few_shot_path`
arguments allow overrides. A missing prompt fails instead of silently replacing
it with different instructions.

## Request provenance

New detailed records contain `input_metadata` (the actual filtered dataset field
sent in the user message) and `tagger_meta`. The latter records SHA-256 hashes of
that metadata, the complete request payload, UTF-8 prompt text, filtered examples,
vocabulary and configuration; it also records the vocabulary and library
versions, requested model, timestamp and provider origin. A model name or request
ID returned by the provider is stored separately as `provider_response_model`
or `provider_response_id`; neither proves which model a proxy actually served.
Authentication headers and endpoint credentials are not stored.

The hashes identify inputs, not generated reasoning. Preserve the input file,
output file and software release alongside them; hashes alone cannot recover
missing text. These fields describe new calls made by this version and do not
reconstruct earlier catalogue runs.

## Command-line tools

```bash
eegdash-fetch-incomplete --output-json incomplete.json --limit 10
eegdash-fetch-complete --output-json complete.json --limit 10
eegdash-update-csv --llm-json results.json --csv dataset_summary.csv --dry-run
```

The first two commands use remote EEGDash/OpenNeuro metadata services. The CSV
updater fills empty fields only and applies its confidence threshold (default
0.5); inspect a dry run before applying updates. The equivalent scripts under
`scripts/` remain as compatibility wrappers. `--help` works without credentials.

The source checkout also contains batch tagging and catalogue-writer tools.
`tag_with_llm.py` records unsuccessful attempts explicitly with `status=error`
and exits nonzero if any call failed. Genuine zero-confidence abstention is not
counted as a failed call. The API and Mongo writers verify captured request
provenance before connecting or writing, skip explicit failure records, and
refuse legacy outputs without the needed provenance. They do not replace a
missing historical input with model reasoning or with today's prompt.

```bash
python scripts/tag_with_llm.py --input incomplete.json --output results.json --workers 3
python scripts/push_tags_to_api.py --llm-json results.json --dry-run
```

An actual API write requires an explicit `EEGDASH_ADMIN_TOKEN` (or `--admin-token`);
`EEGDASH_API_URL` (or `--api-url`) selects the catalogue. It checks dataset
existence and validates the admin endpoint response. Because that endpoint is an
upsert, a concurrent deletion can still cause insertion; the writer reports an
unexpected insertion as a failure. A direct Mongo write requires `pymongo` and an explicit
`MONGO_URI` environment value or `--mongo-uri`; no credential is embedded in the
source. Neither writer's dry-run needs a database client or connection. Review
output and source scope before using an administrator-authorized writer.

## Tests and distribution checks

The unit suite uses synthetic local BIDS files and mocked HTTP responses; a
network guard prevents live calls. It includes extraction assertions, malformed
response/error tests, request-provenance tests and failed-write-input rejection.
Dependency installation requires network access unless packages are cached.

```bash
uv sync --frozen
uv run pytest --cov=eegdash_tagger --cov-report=term
uv build
uv venv /tmp/tagger-wheel --python 3.12
uv pip install --python /tmp/tagger-wheel/bin/python dist/*.whl
cd /tmp
/tmp/tagger-wheel/bin/python /path/to/eegdash-llm-tagger/tools/smoke_installed.py
```

The last check must use a normal wheel installation, outside the checkout. It
checks resource lookup, all three installed `--help` commands, successful
abstention, malformed-output failure, and CSV dry-run/update behavior on temporary
fixtures. The GitHub Actions workflow is configured to run unit and wheel checks on Linux/Python 3.11–3.13.
Test coverage is a measure of exercised code, not evidence of annotation accuracy.

## Development and support

The vocabulary, prompt and examples must remain consistent; the tests check
this. Add or update offline regression tests with changes to metadata extraction,
response handling and persistence-related utilities. File issues or pull requests
at <https://github.com/eegdash/eegdash-llm-tagger>. There is no guaranteed response
time. See `CHANGELOG.md` for the behavior changes in this release candidate.

## Licence

The original software and documentation in this repository are available under
the [MIT licence](LICENSE), copyright (c) 2026 EEGDash contributors.

Third-party dependencies retain their own licences. Dataset metadata and
documentation, including source material reproduced in the bundled few-shot
examples and ground-truth tables, retain their original rights and licence
terms. The MIT licence does not relicense those materials; consult the source
dataset's licence and attribution requirements before reusing them.
