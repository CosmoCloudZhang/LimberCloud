"""Shared evaluation contract for the fiducial-plus-ensemble campaign.

The paper campaign is sample ID 0 (fiducial, no random draw) plus sampled IDs
1–1000. One table is shared across Y1/Y10, Single/Double/Triple, every backend
and every NUMERIC order. This module records the scientific choices that must
agree before spectra are treated as interchangeable. It does not choose the
unresolved ``eta_IA`` value.
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

# Generator source at the audited revision hard-codes 0.5. The manuscript
# states 0. Neither the plans nor this checkout authorise picking one.
ETA_IA_GENERATOR_VALUE = 0.5
ETA_IA_MANUSCRIPT_VALUE = 0.0
ETA_IA_STATUS_UNRESOLVED = "unresolved"
ETA_IA_STATUS_EXPLICIT = "explicit"

# NN observer interval for P = P1 * (chi / chi1)^3. Do not replace 1/4 by 1/2.
NN_OBSERVER_FACTORS = {
    "element1": 1.0 / 12.0,
    "element2": 1.0 / 12.0,
    "element3": 1.0 / 4.0,
}

# Current NN.coefficient omits the rising-hat diagonal on the last interval.
NN_FINAL_DIAGONAL_POLICY = "current_implementation_omits_final_diagonal"


class UnresolvedScienceDecision(ValueError):
    """Raised when a caller asks for a value this stage must not invent."""


@dataclass(frozen=True)
class EtaIADecision:
    """Record the intrinsic-alignment redshift slope without inventing it.

    Args:
        status: ``unresolved`` or ``explicit``.
        explicit_value: Caller-supplied eta when ``status`` is ``explicit``.
        generator_value: Historical generator value, recorded as provenance.
        manuscript_value: Manuscript value, recorded as provenance.
    """

    status: str = ETA_IA_STATUS_UNRESOLVED
    explicit_value: float | None = None
    generator_value: float = ETA_IA_GENERATOR_VALUE
    manuscript_value: float = ETA_IA_MANUSCRIPT_VALUE

    def __post_init__(self) -> None:
        if self.status not in {ETA_IA_STATUS_UNRESOLVED, ETA_IA_STATUS_EXPLICIT}:
            raise ValueError(f"Unknown eta_IA status {self.status!r}")
        if self.status == ETA_IA_STATUS_EXPLICIT and self.explicit_value is None:
            raise ValueError("An explicit eta_IA decision requires a value")
        if self.status == ETA_IA_STATUS_UNRESOLVED and self.explicit_value is not None:
            raise ValueError("An unresolved eta_IA decision cannot carry a chosen value")

    @classmethod
    def unresolved(cls) -> "EtaIADecision":
        """Return the recorded disagreement, with no selected value."""

        return cls()

    @classmethod
    def explicit(cls, value: float) -> "EtaIADecision":
        """Record a caller-supplied eta. This does not validate a preferred law."""

        return cls(status=ETA_IA_STATUS_EXPLICIT, explicit_value=float(value))

    @property
    def resolved_value(self) -> float:
        """Return an explicit eta.

        Returns:
            float: The caller-supplied slope.

        Raises:
            UnresolvedScienceDecision: When the generator/manuscript disagreement
            is still open. The message names both provenance values.
        """

        if self.status != ETA_IA_STATUS_EXPLICIT or self.explicit_value is None:
            raise UnresolvedScienceDecision(
                "eta_IA is unresolved: the generator records "
                f"{self.generator_value} and the manuscript records "
                f"{self.manuscript_value}. Pass an explicit decision; "
                "this code will not invent one."
            )
        return float(self.explicit_value)

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready provenance."""

        return {
            "status": self.status,
            "explicit_value": self.explicit_value,
            "generator_value": self.generator_value,
            "manuscript_value": self.manuscript_value,
            "nuisance_cosmology_policy": NUISANCE_COSMOLOGY_POLICY,
        }


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
