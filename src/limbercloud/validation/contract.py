"""Shared evaluation contract for the fiducial-plus-ensemble campaign.

The paper campaign is sample ID 0 (fiducial, no random draw) plus sampled IDs
1–1000. One table is shared across Y1/Y10, Single/Double/Triple, every backend
and every NUMERIC order. This module records the scientific choices that must
agree before spectra are treated as interchangeable.

The intrinsic-alignment redshift slope is resolved: ``eta_IA = 0.0`` is the
adopted LSST DESC SRD fiducial. The historical generator value 0.5 remains as
provenance only, and products carrying it are refused for accepted comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "limbercloud.evaluation.v1"
SPECTRA_SCHEMA_VERSION = "limbercloud.spectra.v1"
CAMPAIGN_SAMPLED_COUNT = 1000
FIDUCIAL_SAMPLE_ID = 0

# Proposed common domain for currently nonzero sampled parameters.
RELATIVE_HALF_WIDTH = 0.10

PRIMARY_PARAMETERS = (
    "H",
    "W0",
    "WA",
    "NS",
    "AS",
    "M_NU",
    "N_EFF",
    "OMEGA_B",
    "OMEGA_K",
    "OMEGA_CDM",
    "OMEGA_GAMMA",
)
SAMPLED_PARAMETERS = (
    "H",
    "W0",
    "NS",
    "AS",
    "M_NU",
    "N_EFF",
    "OMEGA_B",
    "OMEGA_CDM",
)
FIXED_ZERO_PARAMETERS = ("WA", "OMEGA_K")

# Photon density parameter. It is passed explicitly for every sample so the
# nuisance generators and the sampled constructor share one radiation
# convention. Holding it fixed is the declared choice, not an oversight.
FIXED_NONZERO_PARAMETERS = ("OMEGA_GAMMA",)

CONFIGURATION_PROBES = {
    "Single": ("EE",),
    "Double": ("TE", "TT"),
    "Triple": ("EE", "TE", "TT"),
}

PROBE_COMPONENTS = {
    "EE": ("SS", "SI", "IS", "II"),
    "TE": ("MS", "MI", "GS", "GI"),
    "TT": ("MM", "MG", "GM", "GG"),
}

# Current generators store a fiducial table and reuse it for every sample.
NUISANCE_COSMOLOGY_POLICY = "fixed_tabulated_at_fiducial"

# Adopted intrinsic-alignment redshift slope. The LSST DESC SRD fiducial is 0.
ETA_IA_ADOPTED_VALUE = 0.0
ETA_IA_SOURCE = "LSST DESC SRD fiducial intrinsic-alignment redshift slope"
ETA_IA_STATUS_ADOPTED = "adopted"
ETA_IA_STATUS_DIAGNOSTIC = "diagnostic"

# Provenance only. Products generated with 0.5 keep their own identity and are
# refused for accepted comparisons; relabelling their metadata is not enough.
ETA_IA_HISTORICAL_GENERATOR_VALUE = 0.5

# The distinct amplitude and pivot. Do not replace every 0.5 in the generator.
A_IA_AMPLITUDE = 0.5
IA_PIVOT_REDSHIFT = 0.5
IA_CRITICAL_DENSITY_CONSTANT = "5e-14/h**2"
IA_DENSITY_CONVENTION = (
    "rho_x(a=1, species=matter, is_comoving=True); no (1+z)^3 factor"
)
IA_GROWTH_NORMALIZATION = "D(0)=1"

# NN observer interval for P = P1 * (chi / chi1)^3. Do not replace 1/4 by 1/2.
NN_OBSERVER_FACTORS = {
    "element1": 1.0 / 12.0,
    "element2": 1.0 / 12.0,
    "element3": 1.0 / 4.0,
}

# NN stores all four placements on every interval, including the rising-hat
# diagonal of the final interval. Density is zero outside the finite domain,
# not forced to zero at its last node.
NN_FINAL_DIAGONAL_POLICY = "full_basis_including_final_diagonal"


class UnresolvedScienceDecision(ValueError):
    """Raised when a caller asks for a value this stage must not invent."""


@dataclass(frozen=True)
class EtaIADecision:
    """The adopted intrinsic-alignment redshift slope and its provenance.

    Args:
        value: The slope in use. The campaign default is 0.
        status: ``adopted`` for the campaign value, ``diagnostic`` for an
            explicitly labelled alternate-eta fixture outside acceptance.
        source: Citation for the adopted value.
        historical_generator_value: The retired generator value, kept so
            products written with it remain identifiable.
    """

    value: float = ETA_IA_ADOPTED_VALUE
    status: str = ETA_IA_STATUS_ADOPTED
    source: str = ETA_IA_SOURCE
    historical_generator_value: float = ETA_IA_HISTORICAL_GENERATOR_VALUE

    def __post_init__(self) -> None:
        if self.status not in {ETA_IA_STATUS_ADOPTED, ETA_IA_STATUS_DIAGNOSTIC}:
            raise ValueError(f"Unknown eta_IA status {self.status!r}")
        object.__setattr__(self, "value", float(self.value))
        if self.status == ETA_IA_STATUS_ADOPTED and self.value != ETA_IA_ADOPTED_VALUE:
            raise ValueError(
                f"The adopted campaign eta_IA is {ETA_IA_ADOPTED_VALUE}; "
                f"got {self.value}. Label an alternate slope as diagnostic."
            )

    @classmethod
    def adopted(cls) -> "EtaIADecision":
        """Return the adopted campaign decision, ``eta_IA = 0``."""

        return cls()

    @classmethod
    def diagnostic(cls, value: float) -> "EtaIADecision":
        """Return an explicitly labelled alternate slope for diagnostics only.

        Args:
            value (float): Slope used by a diagnostic fixture. It never enters
                an accepted campaign product.

        Returns:
            EtaIADecision: A decision whose status is ``diagnostic``.
        """

        return cls(value=float(value), status=ETA_IA_STATUS_DIAGNOSTIC)

    @property
    def is_accepted(self) -> bool:
        """Return whether this decision may label an accepted product."""

        return self.status == ETA_IA_STATUS_ADOPTED

    @property
    def resolved_value(self) -> float:
        """Return the slope in use."""

        return float(self.value)

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready provenance."""

        return {
            "status": self.status,
            "value": float(self.value),
            "source": self.source,
            "historical_generator_value": self.historical_generator_value,
            "nuisance_cosmology_policy": NUISANCE_COSMOLOGY_POLICY,
        }


def require_accepted_eta(record) -> float:
    """Validate the IA metadata of a loaded nuisance artifact.

    Args:
        record: Mapping loaded from ``config/intrinsic_alignment.json`` or from
            a product manifest.

    Returns:
        float: The adopted slope.

    Raises:
        UnresolvedScienceDecision: When the slope is missing, unresolved or is
        not the adopted campaign value. Updating metadata alone does not
        regenerate an array written with another slope.
    """

    if "eta_pivot" not in record:
        raise UnresolvedScienceDecision(
            "The intrinsic-alignment artifact records no eta_pivot. Regenerate it "
            f"with the adopted value {ETA_IA_ADOPTED_VALUE}."
        )
    status = str(record.get("eta_decision", ""))
    if status != ETA_IA_STATUS_ADOPTED:
        raise UnresolvedScienceDecision(
            f"The intrinsic-alignment artifact records eta_decision={status!r}. "
            f"Accepted products require {ETA_IA_STATUS_ADOPTED!r}."
        )
    value = float(record["eta_pivot"])
    if value != ETA_IA_ADOPTED_VALUE:
        raise UnresolvedScienceDecision(
            f"The intrinsic-alignment artifact records eta_pivot={value}. The "
            f"adopted campaign value is {ETA_IA_ADOPTED_VALUE}; regenerate the array."
        )
    return value


def configuration_probes(configuration: str) -> tuple[str, ...]:
    """Return the probe list for a title-case configuration label.

    Args:
        configuration (str): ``Single``, ``Double`` or ``Triple``.

    Returns:
        tuple[str, ...]: ``EE``, ``TE`` and/or ``TT`` in that order.
    """

    try:
        return CONFIGURATION_PROBES[configuration]
    except KeyError as error:
        choices = ", ".join(CONFIGURATION_PROBES)
        raise ValueError(
            f"Unknown configuration {configuration!r}; expected one of: {choices}"
        ) from error


def multiplicative_bounds(value: float, half_width: float = RELATIVE_HALF_WIDTH) -> tuple[float, float]:
    """Return sorted multiplicative bounds.

    Args:
        value (float): Fiducial parameter value. Negative fiducials such as
            ``w0`` reverse the raw products, so the returned pair is ordered.
        half_width (float): Fractional half-width. The proposed default is 0.10.

    Returns:
        tuple[float, float]: ``(lower, upper)`` with ``lower <= upper``.
    """

    if half_width < 0:
        raise ValueError(f"half_width must be >= 0; got {half_width}")
    low = float(value) * (1.0 - float(half_width))
    high = float(value) * (1.0 + float(half_width))
    return (min(low, high), max(low, high))
