# Testing Strategy

The roadmap calls for expanded integration and performance coverage. This
strategy describes the additions implemented in this update.

## Integration suite

* `tests/test_integration_workflows.py` exercises end-to-end CLI flows using the
  training and forecasting commands on a synthetic dataset. The tests generate
  temporary GloVe embeddings so that alternative architectures can be swapped in
  through `ModelFactory`.
* The suite is marked with `@pytest.mark.integration` and can be invoked
  independently via `pytest -m integration`.

## Performance checks

* `tests/test_performance_benchmarks.py` benchmarks the lightweight
  `entropy_news.utils.benchmark` helper to ensure the timing harness behaves as
  expected without introducing slow regressions.
* Performance-focused cases use the `performance` marker. They run in isolation
  (`pytest -m performance`) to keep routine unit test runs fast.

## Continuous integration

The `Quality Assurance` GitHub Actions workflow (`.github/workflows/ci.yml`)
drives the automated suites:

1. **Cross-platform validation matrix:** Ubuntu (x86_64), Ubuntu (ARM64), and
   Windows runners execute the core test suite. The Linux x86_64 job captures
   coverage via `pytest --cov=entropy_news --cov-fail-under=95`, ensuring pull
   requests fail if coverage dips below the Release 1.0 baseline.
2. **Integration tests:** `pytest -m integration -q`
3. **Performance checks:** `pytest -m performance -q`

Each job uploads its textual report (and coverage XML for the Linux job) so
historical artefacts can be trended in external dashboards.

## Quality metrics & coverage tracking

- Coverage reporting now runs automatically on the Linux x86_64 matrix job via
  `pytest --cov=entropy_news --cov-report=xml --cov-report=term --cov-fail-under=95`.
  Generated reports are uploaded as artefacts and mirrored into dashboards for
  longitudinal trend tracking.
- The `--cov-fail-under=95` guard enforces the Release 1.0 coverage threshold.
  Any module additions without sufficient tests or regressions below the limit
  fail the pull request check.
- Weekly report reviews feed updates into the implementation plan so newly
  discovered gaps can be triaged promptly.

## Cross-platform automation

- Core test jobs now run on Linux (x86_64 and ARM64) and Windows runners to
  validate CLI tooling and distributed helpers on the Release 1.0 target
  platforms.
- Report artefacts from each runner surface parity dashboards so environment
  regressions are visible immediately.
- Platform-specific nuances (e.g., filesystem semantics, PowerShell path
  handling) continue to be cross-referenced in deployment runbooks as gaps are
  observed.

## Stress & reliability testing

- Synthetic stress scenarios are implemented in `entropy_news.utils.stress` and
  exercised via `pytest -m stress`.
- A dedicated scheduled workflow (`.github/workflows/stress.yml`) runs every
  Monday at 05:00 UTC and on manual dispatch, uploading its report artefact for
  review alongside monitoring dashboards.
- Failure signatures uncovered in stress runs feed into runbooks and
  troubleshooting guides ahead of Release 1.0.

## Local execution

Developers can run the individual suites with dedicated commands:

```bash
pytest -q                       # unit tests with coverage gating in CI
pytest -m integration -q        # integration suite
pytest -m performance -q        # performance checks
pytest -m stress -q             # synthetic reliability scenarios
```

The new markers are declared in `pytest.ini` to avoid unregistered-marker
warnings.
