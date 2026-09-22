"""Scientific contract, samples, angular operator and spectrum assembly."""

from limbercloud.validation.assembly import (
    active_lensing_amplitude,
    analytical_magnification_response,
    ccl_magnification_bias,
    hubble_distance_factor,
)
from limbercloud.validation.contract import (
    ETA_IA_ADOPTED_VALUE,
    EtaIADecision,
    configuration_probes,
    require_accepted_eta,
)
from limbercloud.validation.cosmology import (
    FIDUCIAL_SOLVER,
    build_effective_cosmology,
    effective_cosmology_kwargs,
)
from limbercloud.validation.estimator import (
    AngularContract,
    canonical_angular_contract,
    canonical_band_estimator,
    canonical_ell_nodes,
    canonical_raw_estimator,
    natural_spline_bandpowers,
)
from limbercloud.validation.method import MethodIdentity
from limbercloud.validation.nuisance import (
    load_alignment,
    load_galaxy_bias,
    load_magnification_slope,
)
from limbercloud.validation.samples import (
    ccl_cosmology_kwargs,
    evaluation_sample_ids,
    generate_cosmology_table,
    load_cosmology_table,
    sampled_parameter_rows,
)

__all__ = [
    "ETA_IA_ADOPTED_VALUE",
    "FIDUCIAL_SOLVER",
    "AngularContract",
    "EtaIADecision",
    "MethodIdentity",
    "active_lensing_amplitude",
    "analytical_magnification_response",
    "build_effective_cosmology",
    "canonical_angular_contract",
    "canonical_band_estimator",
    "canonical_ell_nodes",
    "canonical_raw_estimator",
    "ccl_cosmology_kwargs",
    "ccl_magnification_bias",
    "configuration_probes",
    "effective_cosmology_kwargs",
    "evaluation_sample_ids",
    "generate_cosmology_table",
    "hubble_distance_factor",
    "load_alignment",
    "load_cosmology_table",
    "load_galaxy_bias",
    "load_magnification_slope",
    "natural_spline_bandpowers",
    "require_accepted_eta",
    "sampled_parameter_rows",
]
