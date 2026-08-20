"""Centralized canonical runtime paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_NAMES = {
    "cosmology": "cosmology.json",
    "survey": "survey.json",
    "number_density": "number_density.json",
    "galaxy_bias": "galaxy_bias.json",
    "magnification_bias": "magnification_bias.json",
    "intrinsic_alignment": "intrinsic_alignment.json",
}


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve LimberCloud runtime inputs and outputs."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @classmethod
    def from_root(
        cls,
        root: str | os.PathLike[str],
    ) -> "ProjectPaths":
        """Create paths rooted at the canonical runtime directory."""

        return cls(root=Path(root))

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def survey_data(self, survey: str) -> Path:
        """Return the data directory for ``Y1`` or ``Y10``."""

        return self.data / _validate_survey(survey)

    def config_file(self, name: str) -> Path:
        """Return a named configuration JSON path."""

        try:
            filename = _CONFIG_NAMES[name]
        except KeyError as error:
            choices = ", ".join(sorted(_CONFIG_NAMES))
            raise ValueError(f"Unknown configuration file {name!r}; expected: {choices}") from error
        return self.config / filename

    def spectrum_results(
        self,
        backend: str,
        survey: str,
        device: str | None = None,
    ) -> Path:
        """Return a spectrum-result directory without changing its matrix."""

        backend_name = backend.upper()
        survey_name = _validate_survey(survey)
        if backend_name not in {"CCL", "NUMBA", "JAX"}:
            raise ValueError(f"Unknown backend {backend!r}")

        if backend_name == "JAX":
            if device is None or device.upper() not in {"CPU", "GPU"}:
                raise ValueError("JAX results require device='CPU' or device='GPU'")
            device_name = device.upper()
            return self.results / "spectra" / "JAX" / device_name / survey_name

        if device is not None and device.upper() != "CPU":
            raise ValueError(f"{backend_name} only supports the CPU device")
        return self.results / "spectra" / backend_name / survey_name

    def covariance_results(self, survey: str) -> Path:
        """Return the covariance directory for a survey."""

        survey_name = _validate_survey(survey)
        return self.results / "covariance" / survey_name

    def validation_results(self, survey: str) -> Path:
        """Return spectra written by the validation notebooks."""

        survey_name = _validate_survey(survey)
        return self.results / "validation" / "spectra" / survey_name

    def plot_group(self, group: str, survey: str | None = None) -> Path:
        """Return a named plot directory."""

        path = self.plots / group.lower()
        if survey is not None:
            path /= _validate_survey(survey)
        return path


def _validate_survey(survey: str) -> str:
    survey_name = survey.upper()
    if survey_name not in {"Y1", "Y10"}:
        raise ValueError(f"Unknown survey {survey!r}; expected 'Y1' or 'Y10'")
    return survey_name
