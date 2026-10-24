# 0.1.1 release-candidate checks

`release_checks.json` records the local offline suite, whole-package line coverage,
and a normal wheel installation outside the checkout. No model or catalogue
write calls were made. The package and fixtures are synthetic-test evidence,
not evidence of clinical classification performance.

To repeat the checks, follow the commands in the repository README. The wheel
smoke test includes constructor resource lookup, valid abstention and malformed
response handling, every installed CLI's help, and temporary CSV dry-run/write
behavior. Writer protocol tests mock the documented HTTP response envelope;
these checks do not prove production access or deployment.

The prompt, examples and vocabulary are unchanged from base commit
`9f8973034f2ee1459d8a34b815aa85fdf181725f`. The new provenance fields capture
actual request-time metadata and configuration. They cannot recover records
from the historical writer. Build artifact digests should be recorded by the
release publisher after identifying the final source commit.
