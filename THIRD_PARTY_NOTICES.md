# Third-party notices

This file identifies third-party material copied or adapted into this source
repository. It is provided for attribution and license compliance; it does not
replace or modify the terms in [`LICENSE`](LICENSE).

## License map

- Material authored specifically for this project is made available under the
  Apache License 2.0, except where a file or notice says otherwise.
- The portions and data identified below originate from Matheus Facure's
  *Causal Inference for the Brave and True* / `python-causality-handbook`.
  They remain subject to the upstream MIT License reproduced in full below.
  Project modifications and additional code are licensed under Apache-2.0,
  subject to preservation of the upstream MIT notice.
- Research papers and public data sources cited below are references and data
  provenance, not a claim that their text, figures, or trademarks are licensed
  under this project's Apache-2.0 license.
- Python packages declared as dependencies are not vendored in this source
  repository. Each dependency remains governed by the license supplied by its
  distributor.
- `CODE_OF_CONDUCT.md` is adapted from Contributor Covenant 2.1 under the
  Creative Commons Attribution 4.0 International license, as detailed below.

## Causal Inference for the Brave and True

**Upstream project:** *Causal Inference for the Brave and True*  
**Author:** Matheus Facure  
**Repository:** <https://github.com/matheusfacure/python-causality-handbook>  
**Upstream revision reviewed:** `2edbd69f2e3629dd2f6ec805c72f0dbe651d713a`
(2026-07-08)  
**License:** MIT

This project was inspired by the handbook and includes adaptations of its
educational implementations and workflows, including:

- the simplex-constrained synthetic-control formulation and related workflow
  in `causal_loom/model.py` and
  `causal_loom/synthetic_control.py`, based in part on
  `15-Synthetic-Control.ipynb`;
- the conformal-inference residual, cyclic-permutation, p-value-grid, and
  confidence-interval workflow in
  `causal_loom/conformal_inference.py`, related methods in
  `causal_loom/synthetic_control.py`, and conformal visualizations
  in `causal_loom/visualization.py`, based in part on
  `Conformal-Inference-for-Synthetic-Control.ipynb`;
- the SDID unit-weight, time-weight, regularization, and weighted-DiD workflow
  in `causal_loom/_sdid_core.py` and
  `causal_loom/synthetic_difference_in_differences.py`, based in
  part on `25-Synthetic-Diff-in-Diff.ipynb`;
- the educational framing and example selection in the notebooks under
  `notebooks/`, based on or extending handbook chapters 13, 15, and 25; and
- the two example datasets listed in the next section.

The code in this repository substantially restructures and extends those
materials with reusable APIs, validation, tests, inference features,
visualizations, and additional estimators. Those changes do not remove the
obligation to preserve the upstream notice.

## Bundled example data

The detailed provenance, dimensions, field descriptions, and checksums are in
[`notebooks/data/README.md`](notebooks/data/README.md).

### `notebooks/data/billboard_impact.csv`

This file is byte-for-byte identical to:

<https://github.com/matheusfacure/python-causality-handbook/blob/2edbd69f2e3629dd2f6ec805c72f0dbe651d713a/causal-inference-for-the-brave-and-true/data/billboard_impact.csv>

It was generated as synthetic example data by the code in upstream
`Simulated-Data.ipynb` (NumPy seed `123`) and introduced in upstream commit
`0c3898c377241cdeaae210e422c801923c326dd0`. The accompanying chapter says
that the motivating real experiment was confidential and that the published
example is only similar to it. The distributed CSV and generator are covered
here by Matheus Facure's MIT notice.

### `notebooks/data/smoking.csv`

This file is byte-for-byte identical to:

<https://github.com/matheusfacure/python-causality-handbook/blob/2edbd69f2e3629dd2f6ec805c72f0dbe651d713a/causal-inference-for-the-brave-and-true/data/smoking.csv>

It was introduced in upstream commit
`1d2615b82b5f68dd79904180e80aa47352a2d978` and is distributed here under the
upstream MIT notice. It represents the California Proposition 99 application
from:

> Alberto Abadie, Alexis Diamond, and Jens Hainmueller (2010), “Synthetic
> Control Methods for Comparative Case Studies: Estimating the Effect of
> California's Tobacco Control Program,” *Journal of the American Statistical
> Association* 105(490), 493–505.
> <https://doi.org/10.1198/jasa.2009.ap08746>

Appendix A of that paper identifies the underlying variable sources:

- cigarette sales and retail prices: Orzechowski and Walker (2005),
  *The Tax Burden on Tobacco*, historical compilation, volume 40;
- state personal income: U.S. Census, *United States Statistical Abstract*,
  converted to 1997 dollars using the Consumer Price Index;
- state population and the share aged 15–24: U.S. Census Bureau; and
- beer consumption: Beer Institute's *Brewer's Almanac*.

The handbook repository supplies the CSV under its repository-level MIT
License, but neither the handbook nor the paper states a separate license for
the underlying compiled observations. The attribution above documents the
known chain of provenance; it is not a warranty that every upstream source
grants unrestricted redistribution in every jurisdiction. If a use case
requires a formal data-license warranty, replace this teaching copy with data
obtained directly under terms verified with the original data providers.

## Contributor Covenant

`CODE_OF_CONDUCT.md` is adapted from Contributor Covenant, version 2.1.

**Copyright:** Contributor Covenant contributors  
**Source:** <https://www.contributor-covenant.org/version/2/1/code_of_conduct/>  
**License:** Creative Commons Attribution 4.0 International  
**License text:** <https://creativecommons.org/licenses/by/4.0/legalcode>

The required attribution and license link are also preserved directly in
`CODE_OF_CONDUCT.md`. This attribution applies only to that file; it does not
change the Apache-2.0 license for original project material.

## Full upstream MIT License

MIT License

Copyright (c) 2020 Matheus Facure

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
