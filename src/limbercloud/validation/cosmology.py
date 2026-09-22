"""One effective fiducial cosmology and solver specification.

Before this module the nuisance generators and the sampled constructor built
different CCL objects: the generators passed ``Omega_g`` explicitly and CAMB
``kmax=100``, while the sampled constructor omitted ``Omega_g`` and used
``kmax=50``. Matching primary JSON does not prove agreement, so the effective
constructor is centralised here and recorded with its radiation, temperature,
neutrino, transfer and nonlinear conventions plus the solver package versions.

``OMEGA_GAMMA`` is a fixed nonzero primary parameter of the campaign table. It
is not sampled: holding the photon density parameter at its fiducial value is
the declared radiation convention, and it is passed explicitly for every sample
so generators and evaluators share one background.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TypedDict

COSMOLOGY_SCHEMA_VERSION = "limbercloud.effective-cosmology.v1"

# Radiation convention. CCL would otherwise derive Omega_g from T_CMB and h.
RADIATION_POLICY = "explicit_fixed_omega_gamma_from_fiducial_configuration"
TEMPERATURE_POLICY = "ccl_default_T_CMB; not overridden"
NEUTRINO_POLICY = (
    "single massive species (mass_split='single') with sampled M_NU and N_EFF"
)

CAMB_KMAX = 100
CAMB_LMAX = 5000
HALOFIT_VERSION = "mead2020_feedback"
HMCODE_LOG_T_AGN = 7.8

TRANSFER_FUNCTION = "boltzmann_camb"
MATTER_POWER_SPECTRUM = "halofit"
MASS_SPLIT = "single"


class CosmologyContractError(ValueError):
    """Raised when a parameter row cannot build the effective cosmology."""


@dataclass(frozen=True)
class SolverSpecification:
    """The effective CCL/CAMB settings shared by generators and evaluators.

    Args:
        transfer_function: CCL transfer-function backend.
        matter_power_spectrum: CCL nonlinear power prescription.
        mass_split: Neutrino mass splitting passed to CCL.
        camb_kmax: CAMB ``kmax`` in ``Mpc^-1``.
        camb_lmax: CAMB ``lmax``.
        halofit_version: HMCode/halofit variant.
        hmcode_log_t_agn: HMCode AGN feedback temperature.
        radiation_policy: How ``Omega_g`` is supplied.
        temperature_policy: How ``T_CMB`` is supplied.
        neutrino_policy: How massive neutrinos are configured.
    """

    transfer_function: str = TRANSFER_FUNCTION
    matter_power_spectrum: str = MATTER_POWER_SPECTRUM
    mass_split: str = MASS_SPLIT
    camb_kmax: int = CAMB_KMAX
    camb_lmax: int = CAMB_LMAX
    halofit_version: str = HALOFIT_VERSION
    hmcode_log_t_agn: float = HMCODE_LOG_T_AGN
    radiation_policy: str = RADIATION_POLICY
    temperature_policy: str = TEMPERATURE_POLICY
    neutrino_policy: str = NEUTRINO_POLICY

    def extra_parameters(self) -> dict[str, dict[str, int | float | str]]:
        """Return the CAMB ``extra_parameters`` block."""

        return {
            "camb": {
                "kmax": int(self.camb_kmax),
                "lmax": int(self.camb_lmax),
                "halofit_version": self.halofit_version,
                "HMCode_logT_AGN": float(self.hmcode_log_t_agn),
            }
        }

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-ready solver description."""

        return {
            "schema_version": COSMOLOGY_SCHEMA_VERSION,
            "transfer_function": self.transfer_function,
            "matter_power_spectrum": self.matter_power_spectrum,
            "mass_split": self.mass_split,
            "camb": self.extra_parameters()["camb"],
            "radiation_policy": self.radiation_policy,
            "temperature_policy": self.temperature_policy,
            "neutrino_policy": self.neutrino_policy,
        }

    def fingerprint(self) -> str:
        """Return a deterministic hash of the solver specification."""

        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FIDUCIAL_SOLVER = SolverSpecification()


class EffectiveCosmologyKwargs(TypedDict):
    """Keyword arguments accepted by ``pyccl.Cosmology`` for this project."""

    h: float
    w0: float
    wa: float
    n_s: float
    A_s: float
    m_nu: float
    Neff: float
    Omega_b: float
    Omega_k: float
    Omega_c: float
    Omega_g: float
    mass_split: str
    matter_power_spectrum: str
    transfer_function: str
    extra_parameters: dict[str, dict[str, int | float | str]]


# Keys every effective constructor needs from a parameter row.
REQUIRED_PARAMETERS = (
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


def effective_cosmology_kwargs(
    row,
    solver: SolverSpecification = FIDUCIAL_SOLVER,
) -> EffectiveCosmologyKwargs:
    """Build the one effective ``pyccl.Cosmology`` keyword set.

    Args:
        row: Mapping with every name in :data:`REQUIRED_PARAMETERS`. Both the
            fiducial ``config/cosmology.json`` and one row of the canonical
            sample table satisfy this.
        solver (SolverSpecification): Settings shared by every consumer.

    Returns:
        EffectiveCosmologyKwargs: Keyword arguments for ``pyccl.Cosmology``. No
        fresh random draw and no per-caller settings difference.

    Raises:
        CosmologyContractError: When a required parameter is missing.
    """

    missing = [name for name in REQUIRED_PARAMETERS if name not in row]
    if missing:
        raise CosmologyContractError(
            "Effective cosmology parameters missing: " + ", ".join(missing)
        )
    return {
        "h": float(row["H"]),
        "w0": float(row["W0"]),
        "wa": float(row["WA"]),
        "n_s": float(row["NS"]),
        "A_s": float(row["AS"]),
        "m_nu": float(row["M_NU"]),
        "Neff": float(row["N_EFF"]),
        "Omega_b": float(row["OMEGA_B"]),
        "Omega_k": float(row["OMEGA_K"]),
        "Omega_c": float(row["OMEGA_CDM"]),
        "Omega_g": float(row["OMEGA_GAMMA"]),
        "mass_split": solver.mass_split,
        "matter_power_spectrum": solver.matter_power_spectrum,
        "transfer_function": solver.transfer_function,
        "extra_parameters": solver.extra_parameters(),
    }


def build_effective_cosmology(row, solver: SolverSpecification = FIDUCIAL_SOLVER):
    """Instantiate ``pyccl.Cosmology`` from the centralised specification.

    Args:
        row: Parameter mapping accepted by :func:`effective_cosmology_kwargs`.
        solver (SolverSpecification): Shared solver settings.

    Returns:
        pyccl.Cosmology: The effective cosmology. CCL is imported lazily so
        readers and path helpers never initialise a scientific runtime.
    """

    import pyccl

    return pyccl.Cosmology(**effective_cosmology_kwargs(row, solver))


def sample_limber_power(cosmology, chi_grid, scale_factor_grid, ell_grid, *, power_provider):
    """Sample physical power only at positive Limber distances.

    Args:
        cosmology: Active cosmology passed to the supplied power provider.
        chi_grid: Increasing distances in Mpc, optionally starting at zero.
        scale_factor_grid: Positive scale factors on the radial grid.
        ell_grid: Positive multipoles, defining k=(ell+1/2)/chi in Mpc^-1.
        power_provider: Callable accepting ``cosmo``, ``k`` and ``a`` keywords
            and returning power in Mpc^3 for the vector of wavenumbers.

    Returns:
        numpy.ndarray: Float64 power with shape (multipole, radial node).
        The analytical reconstruction declares cubic power on the observer
        interval, whose zero-distance ordinate is zero. No provider query at
        infinite or substituted maximum-float k is made. Positive-node provider
        failures and nonfinite powers are errors, never replaced with zeros.
    """

    import numpy

    chi = numpy.asarray(chi_grid, dtype=numpy.float64)
    scale = numpy.asarray(scale_factor_grid, dtype=numpy.float64)
    ell = numpy.asarray(ell_grid, dtype=numpy.float64)
    if (
        chi.ndim != 1 or chi.size < 2 or scale.shape != chi.shape
        or ell.ndim != 1 or ell.size == 0
        or not numpy.all(numpy.isfinite(chi))
        or not numpy.all(numpy.isfinite(scale))
        or not numpy.all(numpy.isfinite(ell))
        or numpy.any(chi < 0) or numpy.any(numpy.diff(chi) <= 0)
        or numpy.any(scale <= 0) or numpy.any(ell <= 0)
    ):
        raise CosmologyContractError("Invalid radial, scale-factor or multipole grid")
    power = numpy.zeros((ell.size, chi.size), dtype=numpy.float64)
    for index in numpy.flatnonzero(chi > 0):
        with numpy.errstate(over="ignore"):
            k = (ell + 0.5) / chi[index]
        if not numpy.all(numpy.isfinite(k)):
            raise CosmologyContractError("Positive-node Limber wavenumber is nonfinite")
        values = numpy.asarray(
            power_provider(cosmo=cosmology, k=k, a=scale[index]), dtype=numpy.float64
        )
        if values.shape != ell.shape or not numpy.all(numpy.isfinite(values)):
            raise CosmologyContractError(
                f"Power provider returned invalid values at radial node {index}"
            )
        power[:, index] = values
    return power


def parameter_hash(row) -> str:
    """Hash the primary parameters that define one effective cosmology.

    Args:
        row: Mapping with every name in :data:`REQUIRED_PARAMETERS`.

    Returns:
        str: Hexadecimal SHA-256 digest over the sorted name/value pairs. Extra
        derived keys in ``config/cosmology.json`` do not change this digest.
    """

    missing = [name for name in REQUIRED_PARAMETERS if name not in row]
    if missing:
        raise CosmologyContractError(
            "Effective cosmology parameters missing: " + ", ".join(missing)
        )
    payload = json.dumps(
        {name: float(row[name]) for name in sorted(REQUIRED_PARAMETERS)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_fingerprint(row, solver: SolverSpecification = FIDUCIAL_SOLVER) -> str:
    """Hash the parameters and the solver specification together.

    Args:
        row: Parameter mapping.
        solver (SolverSpecification): Shared solver settings.

    Returns:
        str: Hexadecimal SHA-256 digest identifying the generating model. Two
        tables produced under different solver settings never share it.
    """

    digest = hashlib.sha256()
    digest.update(parameter_hash(row).encode("utf-8"))
    digest.update(b"\n")
    digest.update(solver.fingerprint().encode("utf-8"))
    return digest.hexdigest()


def solver_package_versions() -> dict[str, str]:
    """Return the installed solver package versions recorded with a product.

    Returns:
        dict[str, str]: Versions of the packages that determine the numerical
        background and power spectrum. Missing packages are recorded as
        ``unavailable`` rather than omitted.
    """

    import importlib

    versions: dict[str, str] = {}
    for name in ("pyccl", "camb", "numpy", "scipy"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            versions[name] = "unavailable"
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


@dataclass(frozen=True)
class BackgroundSummary:
    """Derived background quantities compared across generators and evaluators.

    Args:
        redshift: Redshifts at which the arrays are evaluated.
        growth: Normalised growth factor ``D(z)`` with ``D(0)=1``.
        hubble: ``H(z)/H0``.
        comoving_distance: Comoving radial distance in ``Mpc``.
        matter_density: Comoving present-day matter density ``rho_m(0)``.
    """

    redshift: tuple[float, ...]
    growth: tuple[float, ...]
    hubble: tuple[float, ...]
    comoving_distance: tuple[float, ...]
    matter_density: float = field(default=0.0)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-ready summary."""

        return {
            "redshift": list(self.redshift),
            "growth": list(self.growth),
            "hubble_over_h0": list(self.hubble),
            "comoving_distance_mpc": list(self.comoving_distance),
            "rho_m_comoving_present_day": self.matter_density,
        }


def background_summary(cosmology, redshift) -> BackgroundSummary:
    """Evaluate ``D(z)``, ``H(z)/H0``, ``chi(z)`` and ``rho_m(0)``.

    Args:
        cosmology: A ``pyccl.Cosmology`` built from this module.
        redshift: Redshifts to evaluate. ``D(0)`` is 1 by CCL's normalisation.

    Returns:
        BackgroundSummary: Derived quantities used to compare two effective
        constructors. Comparing primary JSON alone does not prove agreement.
    """

    import numpy
    import pyccl

    z_values = numpy.asarray(redshift, dtype=numpy.float64)
    scale = 1.0 / (1.0 + z_values)
    growth = pyccl.background.growth_factor(cosmo=cosmology, a=scale)
    hubble = pyccl.background.h_over_h0(cosmo=cosmology, a=scale)
    distance = pyccl.background.comoving_radial_distance(cosmo=cosmology, a=scale)
    density = pyccl.background.rho_x(
        cosmo=cosmology, a=1.0, species="matter", is_comoving=True
    )
    return BackgroundSummary(
        redshift=tuple(float(value) for value in z_values),
        growth=tuple(float(value) for value in growth),
        hubble=tuple(float(value) for value in hubble),
        comoving_distance=tuple(float(value) for value in distance),
        matter_density=float(density),
    )
