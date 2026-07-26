# Contributing

Thank you for considering a contribution to CausalLoom.

This repository is an experimental causal-inference toolkit. Contributions are
most useful when they improve correctness, diagnostics, documentation, or
reproducibility without overstating what an estimator can identify.

By submitting a contribution, you agree that your contribution may be
distributed under the repository's [Apache License 2.0](LICENSE).

## Before opening a change

- Search existing issues and pull requests to avoid duplicate work.
- For a bug, include a minimal reproducible example, expected behavior, actual
  behavior, Python version, platform, and commit or package version.
- For a new estimator or inference procedure, open an issue first. State the
  estimand, identification assumptions, primary methodological reference, and
  intended validation strategy.
- Do not report vulnerabilities in a public issue. Follow
  [SECURITY.md](SECURITY.md).

## Development setup

Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and reproduce
the locked development environment:

```bash
git clone https://github.com/zugzvangg/causal-loom.git
cd causal-loom
uv sync --locked --all-groups
```

Create a focused branch:

```bash
git switch -c fix/short-description
```

Run the local checks before opening a pull request:

```bash
PYTHONPATH=. uv run --locked pytest -q
uv run --locked ruff check causal_loom tests
uv run --locked ruff format --check causal_loom tests
uv build
```

If you change a notebook, restart the kernel and run all cells from top to
bottom. Keep outputs only when they materially help the reader and do not expose
local paths, tokens, personal data, or unnecessarily large artifacts.

## Engineering guidelines

- Preserve the public API unless the change is intentionally documented as
  breaking.
- Prefer explicit validation with actionable `ValueError` messages over bare
  assertions at public API boundaries.
- Add type hints and concise docstrings for public functions, classes, result
  fields, and non-obvious arguments.
- Return structured result objects when an estimator has multiple outputs.
- Make randomness explicit through a `seed` or `random_state` argument.
- Avoid silently changing an estimand, donor pool, sample, or treatment window.
- Keep plotting separate from estimation where practical.
- Add a changelog entry for user-visible behavior.

Do not mix unrelated formatting or refactoring into a methodological change.
Small pull requests are easier to review and validate.

## Testing causal methods

Every estimator change should include tests proportional to its scientific risk.
Depending on the change, include:

- deterministic recovery on simulated data with a known effect;
- comparison with a value from a paper, replication package, or established
  implementation, including the exact citation and tolerance;
- invariants such as finite outputs, normalized non-negative weights, stable row
  ordering, and deterministic seeded behavior;
- invalid-input and weak-design cases with clear failures;
- regression tests for a reported bug;
- sensitivity to relevant choices such as donor exclusions, treatment timing,
  optimizer tolerance, and clustering.

A passing test suite is necessary but does not prove identification or establish
scientific validity. Document the assumptions and known failure modes of a new
method in both its docstring and the README.

## Data and third-party material

Do not add a dataset, figure, text excerpt, or adapted implementation unless its
redistribution and modification terms are known.

For every third-party addition:

1. identify the author or rights holder and primary source URL;
2. record the applicable license and required attribution;
3. update `THIRD_PARTY_NOTICES.md`;
4. describe dataset provenance in the README or a dedicated data card;
5. avoid committing personal, confidential, proprietary, or access-controlled
   data.

Prefer generated data for tests. Use the smallest redistributable sample needed
for a demonstration.

## Documentation

- Use English for the canonical API documentation and `README.md`.
- Keep user-facing changes in `README.ru.md` aligned when they affect setup,
  capabilities, limitations, provenance, or licensing.
- Use relative links for repository files and stable DOI or canonical project
  links for external references.
- Examples must be executable from a fresh clone using the documented commands.
- Avoid claims such as “production-ready,” “unbiased,” or “validated” unless the
  repository contains evidence supporting the exact claim.

## Commit and pull-request guidance

Write descriptive, imperative commit subjects, for example:

```text
Add clustered-inference edge-case tests
Document donor-pool sensitivity
Fix date normalization in placebo fits
```

A pull request should explain:

- what changed and why;
- which public behavior is affected;
- how the change was verified;
- the methodological reference, if applicable;
- assumptions, trade-offs, and known limitations;
- any license or provenance implications.

### Pull-request checklist

- [ ] The change is focused and contains no unrelated generated files.
- [ ] Tests cover the success path and relevant failures.
- [ ] `pytest`, `ruff`, and the package build pass locally.
- [ ] Public API changes have type hints and documentation.
- [ ] README files and `CHANGELOG.md` are updated when needed.
- [ ] New third-party material has verified redistribution terms and attribution.
- [ ] No secrets, personal data, machine-specific paths, or notebook checkpoints
      were added.

## Review

Maintainers may request changes for correctness, scope, maintainability,
methodological clarity, or licensing. A contribution can be declined even when
the code works if its estimand or assumptions cannot be explained clearly.

All participants must follow the [Code of Conduct](CODE_OF_CONDUCT.md).
