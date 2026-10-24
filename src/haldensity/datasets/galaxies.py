"""Galaxy velocities: the classic multimodal density-estimation data set."""

from __future__ import annotations

from importlib import resources

import pandas as pd


def load_galaxies() -> pd.DataFrame:
    """Velocities (km/s) of 82 galaxies in the Corona Borealis region.

    The classic data set of Roeder (1990), also shipped in R as
    ``MASS::galaxies``; this copy agrees with it value for value. As noted
    there, it omits an 83rd measurement (5607 km/s) reported by Postman et al.
    (1986), and its 78th value, 26690, is a typo for 26960. Both are kept so
    that published analyses of the classic set reproduce.

    Returns
    -------
    pd.DataFrame
        Columns ``id`` (1 to 82) and ``velocity`` (km/s, not rescaled). The
        HAL estimators expect data in ``[0, 1]`` in column ``W1``; rescale with
        a support fixed in advance, e.g. ``velocity / 3e5`` (fraction of the
        speed of light).

    References
    ----------
    Roeder, K. (1990). Density estimation with confidence sets exemplified by
    superclusters and voids in the galaxies. *Journal of the American
    Statistical Association* 85, 617-624.

    Postman, M., Huchra, J. P. and Geller, M. J. (1986). Probes of large-scale
    structures in the Corona Borealis region. *Astronomical Journal* 92,
    1238-1247.
    """
    with resources.files("haldensity.datasets").joinpath("data/galaxies.csv").open("r") as f:
        return pd.read_csv(f)
