"""Validated method identity: family, device and radial interpolation order.

Radial interpolation orders belong to the NUMERIC family only. CCL and NUMBA
are CPU families. JAX requires an explicit CPU or GPU device. NUMERIC requires
one of ``linear``, ``quadratic`` or ``cubic`` and currently runs CPU
quadrature. Every path helper, artifact basename, run identity and benchmark
reader resolves a method through this module so an order token cannot leak into
a CCL, NUMBA or JAX product.

The radial order reconstructs ``phi(chi)``, ``a(chi)`` and the component
effective power. It is a different operation from the angular natural cubic
spline in :mod:`limbercloud.validation.estimator`.
"""

from __future__ import annotations

from dataclasses import dataclass

METHOD_SCHEMA_VERSION = "limbercloud.method.v1"

FAMILY_CCL = "CCL"
FAMILY_NUMBA = "NUMBA"
FAMILY_JAX = "JAX"
FAMILY_NUMERIC = "NUMERIC"
FAMILIES = (FAMILY_CCL, FAMILY_NUMBA, FAMILY_JAX, FAMILY_NUMERIC)

DEVICE_CPU = "CPU"
DEVICE_GPU = "GPU"
DEVICES = (DEVICE_CPU, DEVICE_GPU)

ORDER_LINEAR = "LINEAR"
ORDER_QUADRATIC = "QUADRATIC"
ORDER_CUBIC = "CUBIC"
ORDERS = (ORDER_LINEAR, ORDER_QUADRATIC, ORDER_CUBIC)

# Families that select a device. Everything else is CPU-only.
DEVICE_SELECTING_FAMILIES = (FAMILY_JAX,)

# Families that take a radial interpolation order.
ORDER_SELECTING_FAMILIES = (FAMILY_NUMERIC,)


class MethodError(ValueError):
    """Raised for an unsupported family, device or interpolation combination."""


def normalise_family(family: str) -> str:
    """Return a canonical upper-case family token.

    Args:
        family (str): ``CCL``, ``NUMBA``, ``JAX`` or ``NUMERIC``, any case.

    Returns:
        str: Canonical family token.

    Raises:
        MethodError: When the family is unknown.
    """

    if not isinstance(family, str) or not family.strip():
        raise MethodError(f"Method family must be a non-empty string; got {family!r}")
    token = family.strip().upper()
    if token not in FAMILIES:
        choices = ", ".join(FAMILIES)
        raise MethodError(f"Unknown method family {family!r}; expected one of: {choices}")
    return token


def normalise_device(family: str, device: str | None) -> str:
    """Return the canonical device token for one family.

    Args:
        family (str): Family token, normalised internally.
        device (str | None): ``CPU`` or ``GPU``. JAX requires an explicit
            choice. Other families accept ``None`` or ``CPU`` only.

    Returns:
        str: ``CPU`` or ``GPU``.

    Raises:
        MethodError: When a device is missing, unknown or unsupported.
    """

    family_token = normalise_family(family)
    if device is None or (isinstance(device, str) and not device.strip()):
        if family_token in DEVICE_SELECTING_FAMILIES:
            raise MethodError(
                f"{family_token} requires device='CPU' or device='GPU'; none was given"
            )
        return DEVICE_CPU
    token = str(device).strip().upper()
    if token not in DEVICES:
        choices = ", ".join(DEVICES)
        raise MethodError(f"Unknown device {device!r}; expected one of: {choices}")
    if family_token not in DEVICE_SELECTING_FAMILIES and token != DEVICE_CPU:
        raise MethodError(f"{family_token} only supports the CPU device; got {device!r}")
    return token


def normalise_order(family: str, interpolation: str | None) -> str:
    """Return the canonical radial interpolation order for one family.

    Args:
        family (str): Family token, normalised internally.
        interpolation (str | None): ``linear``, ``quadratic`` or ``cubic`` for
            NUMERIC. Every other family must pass ``None`` or an empty string.

    Returns:
        str: Upper-case order token for NUMERIC, otherwise an empty string.

    Raises:
        MethodError: When NUMERIC has no order, an order is unknown, or a
        non-NUMERIC family is given an order.
    """

    family_token = normalise_family(family)
    supplied = interpolation is not None and str(interpolation).strip() != ""
    if family_token not in ORDER_SELECTING_FAMILIES:
        if supplied:
            raise MethodError(
                f"{family_token} does not take a radial interpolation order; "
                f"got {interpolation!r}. Orders belong to NUMERIC only."
            )
        return ""
    if not supplied:
        choices = ", ".join(order.lower() for order in ORDERS)
        raise MethodError(f"NUMERIC requires an interpolation order: {choices}")
    token = str(interpolation).strip().upper()
    if token not in ORDERS:
        choices = ", ".join(order.lower() for order in ORDERS)
        raise MethodError(
            f"Unknown interpolation {interpolation!r}; expected one of: {choices}"
        )
    return token


@dataclass(frozen=True)
class MethodIdentity:
    """One validated family/device/order combination.

    Args:
        family: ``CCL``, ``NUMBA``, ``JAX`` or ``NUMERIC``.
        device: ``CPU`` or ``GPU``. Only JAX selects ``GPU``.
        interpolation: NUMERIC radial order, or an empty string.
    """

    family: str
    device: str = DEVICE_CPU
    interpolation: str = ""

    @classmethod
    def create(
        cls,
        family: str,
        device: str | None = None,
        interpolation: str | None = None,
    ) -> "MethodIdentity":
        """Normalise and validate one method selection.

        Args:
            family (str): Family token in any case.
            device (str | None): Device token, required for JAX.
            interpolation (str | None): NUMERIC radial order.

        Returns:
            MethodIdentity: Canonical identity.
        """

        family_token = normalise_family(family)
        return cls(
            family=family_token,
            device=normalise_device(family_token, device),
            interpolation=normalise_order(family_token, interpolation),
        )

    def __post_init__(self) -> None:
        family_token = normalise_family(self.family)
        object.__setattr__(self, "family", family_token)
        object.__setattr__(self, "device", normalise_device(family_token, self.device))
        object.__setattr__(
            self, "interpolation", normalise_order(family_token, self.interpolation)
        )

    @property
    def selects_device(self) -> bool:
        """Return whether this family chooses between CPU and GPU."""

        return self.family in DEVICE_SELECTING_FAMILIES

    @property
    def selects_order(self) -> bool:
        """Return whether this family carries a radial interpolation order."""

        return self.family in ORDER_SELECTING_FAMILIES

    @property
    def order_token(self) -> str:
        """Return the filename order token, empty for non-NUMERIC families."""

        return self.interpolation

    def label(self) -> str:
        """Return a compact human label such as ``JAX-GPU`` or ``NUMERIC-LINEAR``."""

        if self.selects_order:
            return f"{self.family}-{self.interpolation}"
        if self.selects_device:
            return f"{self.family}-{self.device}"
        return self.family

    def as_dict(self) -> dict[str, str]:
        """Return the identity fields stored beside every product."""

        return {
            "method_schema_version": METHOD_SCHEMA_VERSION,
            "family": self.family,
            "device": self.device,
            "interpolation": self.interpolation,
        }
