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

A GitHub Actions workflow (`.github/workflows/ci.yml`) orchestrates the test
matrix:

1. Unit tests: `pytest -q`
2. Integration tests: `pytest -m integration -q`
3. Performance checks: `pytest -m performance -q`

The workflow installs the project via Poetry, provisions CPU-only PyTorch, and
exposes caching so the expanded suites remain maintainable for future releases.

## Quality metrics & coverage tracking

- Coverage reporting will be generated via `pytest --cov entropy_news` on every
  workflow job. Historical trends will be published as artefacts and surfaced in
  dashboards to ensure the Release 1.0 target of 95% coverage stays on track.
- Failing thresholds: pull requests will fail when coverage drops below the
  rolling 95% baseline or when uncovered new modules are introduced without
  accompanying tests.
- A weekly scheduled run will refresh coverage badges and annotate the
  implementation plan with any regressions that require engineering follow-up.

## Cross-platform automation

- The CI matrix will expand to run on Linux (x86_64 and ARM64) and Windows
  runners so distributed helpers and CLI tooling are validated on the platforms
  targeted for Release 1.0 adoption.
- Container images produced by the deployment workflow will be smoke-tested on
  these operating systems to catch dependency drift early.
- Platform-specific quirks (e.g., filesystem semantics, PowerShell path
  handling) will be documented and cross-referenced in the deployment runbooks.

## Stress & reliability testing

- Automated stress scenarios will execute extended rolling-window training jobs
  with synthetic burst workloads to surface performance regressions or resource
  leaks.
- Stress runs will emit metrics (throughput, memory, checkpoint durations) to
  the monitoring stack defined in the enterprise scalability milestone so they
  can be reviewed alongside functional test results.
- Failure signatures uncovered in stress tests will feed into runbooks and
  troubleshooting guides ahead of Release 1.0.

## Local execution

Developers can run the individual suites with dedicated commands:

```bash
pytest -q                       # unit tests
pytest -m integration -q        # integration suite
pytest -m performance -q        # performance checks
```

The new markers are declared in `pytest.ini` to avoid unregistered-marker
warnings.
