<!-- PROJECT SHIELDS -->
[![arXiv][arxiv-shield]][arxiv-url]
[![MIT License][license-shield]][license-url]
[![ReseachGate][researchgate-shield]][researchgate-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![GIT][git-shield]][git-url]
<!-- [![finalpaper][finalpaper-shield]][finalpaper-url] -->
<!-- [![Scholar][scholar-shield]][scholar-url] -->
<!-- [![Webpage][webpage-shield]][webpage-url] -->

# Accelerated ADMM: Automated Parameter Tuning and Improved Linear Convergence
This repository contains the code from our paper

> M. Tavakoli, F. Jakob, G. Carnevale, G. Notarstefano, and A. Iannelli. "Accelerated ADMM: Automated Parameter Tuning and Improved Linear Convergence." arXiv preprint. arXiv:2511.21210 (2025). 

## Installation

The code has been developed and tested with Python 3.10.7.  
All required packages can be installed via

```bash
pip install -r requirements.txt
```

All SDPs are solved with the commericial solver MOSEK.

```bash
pip install Mosek
``` 

An academic license can be requested [here](https://www.mosek.com/products/academic-licenses/). Other open-source solvers might work as well (e.g. cvxopt), however, we observed best numerical stability with MOSEK.

## Running Experiments

The main numerical experiments presented in the paper can be reproduced in the ``src`` directory using the notebooks:

- ``convergence_rates.ipynb``: Reproduces Figure 1, Figure 3, and Figure 4.

- ``parameter_grid_search.ipynb``: Reproduces Figure 2.

- ``lasso_regression.ipynb``: Reproduces Figure 5.

## Contact

🧑‍💻 Fabian Jakob \
📧 [fabian.jakob@ist.uni-stuttgart.de](mailto:fabian.jakob@ist.uni-stuttgart.de)

🧑‍💻 Meisam Tavakoli \
📧 [meisam.tavakoli@studio.unibo.it](mailto:meisam.tavakoli@studio.unibo.it)

[git-shield]: https://img.shields.io/badge/Github-fjakob-white?logo=github
[git-url]: https://github.com/fjakob
[license-shield]: https://img.shields.io/badge/License-MIT-T?style=flat&color=blue
[license-url]: https://github.com/col-tasas/2025-accelerated-admm/blob/main/LICENSE
<!-- [webpage-shield]: https://img.shields.io/badge/Webpage-Fabian%20Jakob-T?style=flat&logo=codementor&color=green
[webpage-url]: https://www.ist.uni-stuttgart.de/institute/team/Jakob-00004/ add personal webpage -->
[arxiv-shield]: https://img.shields.io/badge/arXiv-2511.21210-t?style=flat&logo=arxiv&logoColor=white&color=red
[arxiv-url]: https://arxiv.org/abs/2511.21210
<!-- [finalpaper-shield]: https://img.shields.io/badge/SIAM-Paper-T?style=flat&color=red
[finalpaper-url]: https://google.com -->
[researchgate-shield]: https://img.shields.io/badge/ResearchGate-Fabian%20Jakob-T?style=flat&logo=researchgate&color=darkgreen
[researchgate-url]: https://www.researchgate.net/profile/Fabian-Jakob-4
[linkedin-shield]: https://img.shields.io/badge/Linkedin-Fabian%20Jakob-T?style=flat&logo=linkedin&logoColor=blue&color=blue
[linkedin-url]: https://www.linkedin.com/in/fabian-jakob/

