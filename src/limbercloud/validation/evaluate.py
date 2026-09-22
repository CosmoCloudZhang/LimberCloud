"""Shared evaluation assembly.

Plotting and file I/O stay outside this module. Callers pass component spectra
that already use the active cosmology and the named magnification response.
The returned arrays are the final EE/TE/TT sums requested by the configuration,
in raw-node and bandpower form.

This is the assembler, not the evaluator. It does not build a cosmology, choose
a backend or run a sample; Phase 2 supplies the one-cosmology execution path
that calls it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy

from limbercloud.validation.assembly import assemble_configuration
from limbercloud.validation.contract import EtaIADecision, configuration_probes
from limbercloud.validation.estimator import (
    AngularContract,
    EllEstimator,
    assert_same_estimator,
    canonical_angular_contract,
)


@dataclass(frozen=True)
class AssembledSpectra:
    """Raw samples and bandpowers of one configuration.

    Args:
        raw: Probe to spectra sampled on the contract's raw nodes.
        bands: Probe to bandpowers on the contract's intervals.
        contract: The angular operator that produced ``bands``.
        eta_ia: The intrinsic-alignment decision recorded with the result.
    """

    raw: dict[str, numpy.ndarray]
    bands: dict[str, numpy.ndarray]
    contract: AngularContract
    eta_ia: EtaIADecision

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-ready identity of this assembly."""

        return {
            "probes": sorted(self.raw),
            "angular_operator": self.contract.as_dict(),
            "angular_fingerprint": self.contract.fingerprint(),
            "eta_ia": self.eta_ia.as_dict(),
        }


def evaluate_configuration(
    configuration: str,
    components: Mapping[str, Mapping[str, numpy.ndarray]],
    *,
    estimator: EllEstimator,
    reference_estimator: EllEstimator | None = None,
    eta_ia: EtaIADecision | None = None,
) -> dict[str, numpy.ndarray]:
    """Assemble final raw spectra for one configuration.

    Args:
        configuration (str): ``Single``, ``Double`` or ``Triple``.
        components: Probe to component-spectrum mapping. JAX-style separate
            components are summed here. Single does not require TE or TT.
        estimator: Ell coordinate of every component.
        reference_estimator: When supplied, it must fingerprint-match
            ``estimator``. Twenty centres are not compared with 21 nodes.
        eta_ia: Recorded IA decision. The campaign value is ``eta_IA = 0``.

    Returns:
        dict[str, numpy.ndarray]: Final probe spectra on the estimator's
        coordinates. Coefficient tensors are not part of the result.
    """

    if reference_estimator is not None:
        assert_same_estimator(estimator, reference_estimator)
    decision = eta_ia if eta_ia is not None else EtaIADecision.adopted()
    if not isinstance(decision, EtaIADecision):
        raise TypeError("eta_ia must be an EtaIADecision")
    assembled = assemble_configuration(configuration, components)
    expected = set(configuration_probes(configuration))
    if set(assembled) != expected:
        raise RuntimeError(f"Assembled probes {sorted(assembled)} != {sorted(expected)}")
    for probe, values in assembled.items():
        if values.shape[-1] != len(estimator.ell):
            raise ValueError(
                f"Probe {probe} has {values.shape[-1]} multipoles but estimator "
                f"{estimator.name} declares {len(estimator.ell)}"
            )
    return assembled


def assemble_with_bandpowers(
    configuration: str,
    components: Mapping[str, Mapping[str, numpy.ndarray]],
    *,
    contract: AngularContract | None = None,
    eta_ia: EtaIADecision | None = None,
) -> AssembledSpectra:
    """Assemble one configuration and apply the shared angular operator.

    Args:
        configuration (str): ``Single``, ``Double`` or ``Triple``.
        components: Probe to component-spectrum mapping on the contract's raw
            nodes.
        contract: Angular operator. Defaults to the current 21-node, 20-band
            natural-spline contract.
        eta_ia: Recorded IA decision. The campaign value is ``eta_IA = 0``.

    Returns:
        AssembledSpectra: Raw 21-node samples and 20 bandpowers. The two remain
        distinct datasets; the bandpowers are the comparison vector.
    """

    operator = contract if contract is not None else canonical_angular_contract()
    decision = eta_ia if eta_ia is not None else EtaIADecision.adopted()
    raw = evaluate_configuration(
        configuration,
        components,
        estimator=operator.raw_estimator(),
        eta_ia=decision,
    )
    bands = {probe: operator.bandpowers(values) for probe, values in raw.items()}
    return AssembledSpectra(raw=raw, bands=bands, contract=operator, eta_ia=decision)
