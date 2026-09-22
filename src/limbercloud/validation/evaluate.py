"""Shared evaluation assembly.

Plotting and file I/O stay outside this module. Callers pass component spectra
that already use the active cosmology and the named magnification response.
The returned arrays are the final EE/TE/TT sums requested by the configuration.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy

from limbercloud.validation.assembly import assemble_configuration
from limbercloud.validation.contract import EtaIADecision, configuration_probes
from limbercloud.validation.estimator import EllEstimator, assert_same_estimator


def evaluate_configuration(
    configuration: str,
    components: Mapping[str, Mapping[str, numpy.ndarray]],
    *,
    estimator: EllEstimator,
    reference_estimator: EllEstimator | None = None,
    eta_ia: EtaIADecision | None = None,
) -> dict[str, numpy.ndarray]:
    """Assemble final spectra for one configuration.

    Args:
        configuration (str): ``Single``, ``Double`` or ``Triple``.
        components: Probe to component-spectrum mapping. JAX-style separate
            components are summed here. Single does not require TE or TT.
        estimator: Ell coordinate of every component.
        reference_estimator: When supplied, it must fingerprint-match
            ``estimator``. Twenty centres are not compared with 21 nodes.
        eta_ia: Recorded IA decision. Accessing a resolved value is the
            caller's choice; this function stores the decision and does not
            invent an eta.

    Returns:
        dict[str, numpy.ndarray]: Final probe spectra. Coefficient tensors are
        not part of the result.
    """

    if reference_estimator is not None:
        assert_same_estimator(estimator, reference_estimator)
    _ = eta_ia if eta_ia is not None else EtaIADecision.unresolved()
    assembled = assemble_configuration(configuration, components)
    expected = set(configuration_probes(configuration))
    if set(assembled) != expected:
        raise RuntimeError(f"Assembled probes {sorted(assembled)} != {sorted(expected)}")
    return assembled
