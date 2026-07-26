# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project intends to follow [Semantic Versioning](https://semver.org/) once its
public API stabilizes. During the `0.x` phase, minor releases may contain
breaking API changes when they are documented here.

## Unreleased

## 0.1.0 - 2026-07-26

### Added

- Synthetic Control with constrained donor weights, feature and propensity-score
  matching, effect paths, placebo analysis, and conformal inference.
- Classic 2×2 Difference-in-Differences with OLS inference and trend diagnostics.
- Synthetic Difference-in-Differences with unit/time weights, optional outcome
  residualization, fixed effects, clustered standard errors, period effects, and
  cohort aggregation for staggered adoption.
- Batch synthetic-control experiments with fit-quality filtering and aggregate
  paired-test or permutation inference.
- Plotly visualization helpers and three executable demonstration notebooks.
- Automated tests for the public APIs, validation, numerical behavior, and
  figure constructors.
- English and Russian project overviews with reproducible quickstarts,
  identification assumptions, implementation limits, roadmap, and provenance.
- Contribution, security, and community guidelines.
- Explicit third-party attribution for implementation influences and bundled
  demonstration data.
- Continuous-integration checks for supported Python versions, linting, tests,
  notebook execution, and package builds.

### Changed

- Synthetic-control optimizer arguments use the descriptive names `max_iter`
  and `tolerance`.
- Legacy `epochs` and `lr` optimizer arguments are deprecated aliases.
- Project packaging metadata and package discovery are explicit.
- Source distributions include the release documentation, tests, executable
  notebooks, and their documented example data.
- Demonstration notebooks use repository-relative data paths.

### Fixed

- Package discovery excludes the top-level `notebooks` directory, allowing
  source and wheel builds to complete with only the intended Python package.
