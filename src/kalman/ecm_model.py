"""1RC Thevenin equivalent circuit model (ECM) for a Li-ion cell.

State vector x = [SOC, V1]
  SOC : state of charge, in [0, 1]
  V1  : polarization voltage across the single RC branch (volts)

Discrete-time dynamics (current I positive = discharging):
  SOC_{k+1} = SOC_k - (I_k * dt) / (3600 * Q_capacity_Ah)
  V1_{k+1}  = exp(-dt / (R1*C1)) * V1_k + R1 * (1 - exp(-dt / (R1*C1))) * I_k

Measurement (terminal voltage):
  V_term_k = OCV(SOC_k) - V1_k - I_k * R0
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class OCVCurve:
    """Open-circuit-voltage vs SOC lookup, fit from (soc, voltage) sample pairs."""

    def __init__(self, soc_samples: np.ndarray, voltage_samples: np.ndarray, degree: int = 6):
        order = np.argsort(soc_samples)
        self.soc = np.asarray(soc_samples)[order]
        self.voltage = np.asarray(voltage_samples)[order]
        self.coeffs = np.polyfit(self.soc, self.voltage, degree)
        self.deriv_coeffs = np.polyder(self.coeffs)

    def voltage_at(self, soc) -> np.ndarray:
        soc = np.clip(soc, 0.0, 1.0)
        return np.polyval(self.coeffs, soc)

    def slope_at(self, soc) -> np.ndarray:
        """d(OCV)/d(SOC), used for the EKF measurement Jacobian."""
        soc = np.clip(soc, 0.0, 1.0)
        return np.polyval(self.deriv_coeffs, soc)

    @classmethod
    def fit_from_ocv_cycle(cls, charge_mah: np.ndarray, voltage: np.ndarray, capacity_mah: float):
        """Build from a slow (pseudo-OCV) charge or discharge cycle where q(t)/Q ~= SOC."""
        soc = np.clip(charge_mah / capacity_mah, 0.0, 1.0)
        return cls(soc, voltage)


@dataclass
class ECMParams:
    r0: float  # ohmic resistance, ohms
    r1: float  # polarization resistance, ohms
    c1: float  # polarization capacitance, farads
    capacity_ah: float  # cell capacity, amp-hours
    dt: float  # timestep, seconds
    coulombic_efficiency: float = 1.0

    @property
    def tau(self) -> float:
        return self.r1 * self.c1

    @property
    def alpha(self) -> float:
        """RC-branch decay factor exp(-dt/tau)."""
        return float(np.exp(-self.dt / self.tau))


class ThreveninECM:
    """1RC Thevenin ECM, exposing f/h and their Jacobians for KF/EKF/UKF use."""

    def __init__(self, params: ECMParams, ocv_curve: OCVCurve):
        self.p = params
        self.ocv = ocv_curve

    def f(self, x: np.ndarray, current: float) -> np.ndarray:
        """State transition: x=[soc, v1] -> next x, given applied current (A, +discharge)."""
        soc, v1 = x
        soc_next = soc - (self.p.coulombic_efficiency * current * self.p.dt) / (3600.0 * self.p.capacity_ah)
        v1_next = self.p.alpha * v1 + self.p.r1 * (1 - self.p.alpha) * current
        return np.array([np.clip(soc_next, 0.0, 1.0), v1_next])

    def F_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        """Jacobian of f w.r.t. x (2x2), constant for this linear-in-state model."""
        return np.array([[1.0, 0.0], [0.0, self.p.alpha]])

    def h(self, x: np.ndarray, current: float) -> float:
        """Measurement: terminal voltage given state and applied current."""
        soc, v1 = x
        return float(self.ocv.voltage_at(soc) - v1 - current * self.p.r0)

    def H_jacobian(self, x: np.ndarray, current: float) -> np.ndarray:
        """Jacobian of h w.r.t. x (1x2)."""
        soc, _v1 = x
        return np.array([[self.ocv.slope_at(soc), -1.0]])
