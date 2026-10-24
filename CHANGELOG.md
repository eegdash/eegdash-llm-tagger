# Changelog

## 0.1.1 — release candidate

- Include the canonical v2 prompt and 15 examples in non-editable wheels.
- Install the three CLI implementations as package modules; retain script wrappers.
- Raise `TaggingError` for transport and invalid model responses instead of
  returning fabricated successful `Unknown` records. Valid abstentions remain valid.
- Validate required label axes and finite per-axis confidence, preserve structured
  reasoning values, and reject missing prompt files explicitly.
- Capture input/request/prompt/example/vocabulary/configuration identities at the
  time of each new call. Writers validate those identities and reject legacy
  results lacking evidence of actual inputs; this does not reconstruct old runs.
- Mark batch failures explicitly and exit nonzero when any call fails.
- Require explicit Mongo configuration instead of an embedded credential.
- Replace the unasserted network parser check with offline synthetic fixtures,
  add error/provenance regression tests, and test ordinary wheel installations in CI.

- Read CSV fields as text for compatibility with pandas 3, preserving existing and low-confidence values.
- Use the catalogue HTTP admin protocol directly, require an admin token for writes, check dataset existence, and validate the success/count envelope instead of calling a nonexistent client method.


Version 0.1.0 remains the historical version described by the archived catalogue
analysis. The v2 vocabulary and canonical prompt/example content are unchanged in
this release candidate; the repair concerns packaging, execution and provenance.
