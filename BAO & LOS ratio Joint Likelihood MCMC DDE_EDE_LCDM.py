#%%
# -*- coding: utf-8 -*-
"""

@author: hazmc

DESI-consistent BAO + LOS-ratio joint MCMC with selectable cosmological model.

MODEL OPTIONS:
- "LCDM"      : flat ΛCDM
- "LCDM_EDE"  : flat ΛCDM + EDE
- "DDE_EDE"   : flat CPL dark energy + EDE
- "DDE"       : original DDE model (CPL only, no EDE)

For the selected model this script:
- runs the MCMC
- writes a model-specific original and trimmed backend
- computes credible intervals
- finds best chain sample and a post-optimised MLE
- produces the plots:
    1) BAO observable
    2) LOS-ratio observable
    3) dark-sector evolution
    4) parameter diagnostics
    5) log-posterior diagnostics
"""

import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from emcee.backends import HDFBackend
import h5py
from emcee.autocorr import integrated_time
from scipy.optimize import minimize

# ================================================================
# USER MODEL SWITCH
# ================================================================
MODEL_CHOICE = "DDE_EDE"
# Allowed:
# "LCDM"
# "LCDM_EDE"
# "DDE_EDE"
# "DDE"

# ================================================================
# USER INITIAL GUESS SWITCH
# ================================================================
INITIAL_GUESS = "Planck"
# Allowed:
# "Planck"
# "SH0ES"

# ================================================================
# H0 and log_a_c Prior Selection
# ================================================================
H0_BOUNDS = (50, 100)
log_ac_bounds = (-6, -3)

# ================================================================
# guess config
# ================================================================
ALLOWED_GUESS = {"Planck", "SH0ES"}
if INITIAL_GUESS not in ALLOWED_GUESS:
    raise ValueError(f"INITIAL_GUESS must be one of {ALLOWED_GUESS}")

# parameter sets by initial guess
if INITIAL_GUESS == "Planck":
    H_ing = 67.5
    OM_ing = 0.315

elif INITIAL_GUESS == "SH0ES":
    H_ing = 73
    OM_ing = 0.3

print(f"INITIAL_GUESS = {INITIAL_GUESS}")

# ================================================================
# figsize
# ================================================================
fig_size = (12, 10)

# ================================================================
# constants
# ================================================================
c = 299792458.0
MPC_M = 3.085677581e22
H0_to_SI = 3.085677581e19  # H0 [km/s/Mpc] -> s^-1
G = 6.67430e-11  # m^3 kg^-1 s^-2

# --- fixed physical densities ---
omega_b = 0.02237  # Ω_b h^2

T_CMB = 2.7255
N_eff = 3.046
omega_gamma = 2.469e-5 * (T_CMB / 2.7255) ** 4
omega_r_phys = omega_gamma * (1.0 + 0.22710731766 * N_eff)

# ================================================================
# plotting grid
# ================================================================
z_plot = np.linspace(1e-5, 2.5, 1500)

# ================================================================
# DESI data (A) BAO: D_V/r_d * z^{-2/3}
# ================================================================
DESI_x_BAO = np.array([0.295, 0.51, 0.706, 0.934, 1.321, 1.484, 2.33])
DESI_y_BAO = np.array([
    17.92107988, 19.9284877, 20.24367201, 20.63979055,
    20.14492019, 20.03381526, 17.79120736
])
y_err_BAO = np.array([
    0.171708098, 0.209956032, 0.181650363, 0.129585913,
    0.190402378, 0.429596377, 0.176017138
])

# ================================================================
# DESI data (B) LOS ratio: D_M/(z D_H)
# ================================================================
DESI_x_ratio = np.array([0.51, 0.706, 0.934, 1.321, 2.33])
DESI_y_ratio = np.array([1.218606184, 1.263202694, 1.309427402, 1.473891013, 1.938640639])
y_err_ratio = np.array([0.028274248, 0.025328288, 0.017862769, 0.02907657, 0.034844299])

# ================================================================
# model config
# ================================================================
ALLOWED_MODELS = {"LCDM", "LCDM_EDE", "DDE_EDE", "DDE"}
if MODEL_CHOICE not in ALLOWED_MODELS:
    raise ValueError(f"MODEL_CHOICE must be one of {ALLOWED_MODELS}")

USE_DDE = MODEL_CHOICE in {"DDE", "DDE_EDE"}
USE_EDE = MODEL_CHOICE in {"LCDM_EDE", "DDE_EDE"}

# ================================================================
# model-dependent plot colours
# ================================================================
if MODEL_CHOICE == "LCDM":
    BAND_COLOR = "tomato"
    MED_COLOR = "forestgreen"

elif MODEL_CHOICE == "DDE":
    BAND_COLOR = "mediumseagreen"
    MED_COLOR = "darkorange"

elif MODEL_CHOICE == "LCDM_EDE":
    BAND_COLOR = "darkorange"
    MED_COLOR = "forestgreen"

elif MODEL_CHOICE == "DDE_EDE":
    BAND_COLOR = "cornflowerblue"
    MED_COLOR = "darkorange"

# parameter sets by model
if MODEL_CHOICE == "LCDM":
    # theta = (H0, OmegaM)
    param_names = ["H0", "OmegaM"]
    init = np.array([
        H_ing,    # H0
        OM_ing    # OmegaM
    ], dtype=float)
    pos_scales = np.array([1.0, 0.02], dtype=float)
    repair_scales = np.array([2.0, 0.04], dtype=float)
    bounds = [
        H0_BOUNDS,   # H0
        (0.05, 1.0)  # OmegaM
    ]

elif MODEL_CHOICE == "LCDM_EDE":
    # theta = (H0, OmegaM, log10_OmegaEDE0, log10_a_c, w_f)
    param_names = ["H0", "OmegaM", "log10_OmegaEDE0", "log10_a_c", "w_f"]
    init = np.array([
        H_ing,                   # H0
        OM_ing,                  # OmegaM
        -38.0,                   # log10_OmegaEDE0
        np.log10(1.0 / 3000.0),  # log10_a_c
        3.0                      # w_f
    ], dtype=float)
    pos_scales = np.array([1.0, 0.02, 0.25, 0.20, 0.10], dtype=float)
    repair_scales = np.array([2.0, 0.04, 3.0, 0.35, 0.35], dtype=float)
    bounds = [
        H0_BOUNDS,         # H0
        (0.05, 1.0),       # OmegaM
        (-55.0, -25.0),    # log10_OmegaEDE0
        log_ac_bounds,     # log10_a_c
        (-0.999999, 10.0)  # w_f
    ]

elif MODEL_CHOICE == "DDE_EDE":
    # theta = (w0, w1, H0, OmegaM, log10_OmegaEDE0, log10_a_c, w_f)
    param_names = ["w0", "w1", "H0", "OmegaM", "log10_OmegaEDE0", "log10_a_c", "w_f"]
    init = np.array([
        -1.0,                   # w0
        0.0,                    # w1
        H_ing,                  # H0
        OM_ing,                 # OmegaM
        -38.0,                  # log10_OmegaEDE0
        np.log10(1.0 / 3000.0), # log10_a_c
        3.0                     # w_f
    ], dtype=float)
    pos_scales = np.array([0.05, 0.05, 1.0, 0.02, 0.25, 0.20, 0.10], dtype=float)
    repair_scales = np.array([0.10, 0.10, 2.0, 0.04, 3.0, 0.35, 0.35], dtype=float)
    bounds = [
        (-3.0, 1.0),       # w0
        (-4.0, 2.0),       # w1
        H0_BOUNDS,         # H0
        (0.05, 1.0),       # OmegaM
        (-55.0, -25.0),    # log10_OmegaEDE0
        log_ac_bounds,     # log10_a_c
        (-0.999999, 10.0)  # w_f
    ]

elif MODEL_CHOICE == "DDE":
    # theta = (w0, w1, H0, OmegaM)
    param_names = ["w0", "w1", "H0", "OmegaM"]
    init = np.array([
        -1.0,   # w0
        0.0,    # w1
        H_ing,  # H0
        OM_ing  # OmegaM
    ], dtype=float)
    pos_scales = np.array([0.05, 0.05, 1.0, 0.02], dtype=float)
    repair_scales = np.array([0.10, 0.10, 2.0, 0.04], dtype=float)
    bounds = [
        (-3.0, 1.0),   # w0
        (-3.0, 2.0),   # w1
        H0_BOUNDS,     # H0
        (0.05, 1.0)    # OmegaM
    ]

ndim = len(param_names)
nwalkers = max(10 * ndim, 40)
nsteps = 50000

orig_path = f"DESI_MCMC_joint_chain_{MODEL_CHOICE}_{INITIAL_GUESS}.h5"
trim_path = f"DESI_MCMC_joint_chain_trimmed_{MODEL_CHOICE}_{INITIAL_GUESS}.h5"

print(f"MODEL_CHOICE = {MODEL_CHOICE}")
print(f"Parameters   = {param_names}")
print(f"ndim         = {ndim}")
print(f"nwalkers     = {nwalkers}")
print(f"nsteps       = {nsteps}")

# ================================================================
# helpers
# ================================================================
def Omega_r_of_H0(H0):
    h = H0 / 100.0
    return omega_r_phys / (h * h)

def _gamma_from_wf(w_f):
    return 3.0 * (1.0 + w_f)

def OmegaEDE_of_a(a, OmegaEDE0, log10_a_c, w_f):
    """
    Omega_EDE(a) = OmegaEDE0 * (1 + a_c^{-gamma}) / (1 + (a/a_c)^gamma)
    """
    a = np.asarray(a, dtype=float)
    a_c = 10.0 ** log10_a_c
    gamma = _gamma_from_wf(w_f)

    if (a_c <= 0.0) or (not np.isfinite(gamma)) or (gamma <= 0.0) or (OmegaEDE0 < 0.0):
        return np.full_like(a, np.nan, dtype=float)

    num = 1.0 + a_c ** (-gamma)
    den = 1.0 + (a / a_c) ** gamma
    out = OmegaEDE0 * num / den

    if np.any(~np.isfinite(out)) or np.any(out < 0.0):
        return np.full_like(a, np.nan, dtype=float)

    return out

def f_CPL_of_z(z, w0, w1):
    z = np.asarray(z, dtype=float)
    return (1.0 + z) ** (3.0 * (1.0 + w0 + w1)) * np.exp(-3.0 * w1 * (1.0 - 1.0 / (1.0 + z)))

def unpack_theta(theta):
    """
    Unified cosmological parameter dictionary for all models.

    Always returns:
      H0, OmegaM, w0, w1, OmegaDE0, OmegaEDE0, log10_a_c, w_f
    """
    if MODEL_CHOICE == "LCDM":
        H0, OmegaM = theta
        Omr = Omega_r_of_H0(H0)
        OmegaDE0 = 1.0 - OmegaM - Omr
        return {
            "H0": H0,
            "OmegaM": OmegaM,
            "w0": -1.0,
            "w1": 0.0,
            "OmegaDE0": OmegaDE0,
            "OmegaEDE0": 0.0,
            "log10_a_c": None,
            "w_f": None
        }

    if MODEL_CHOICE == "LCDM_EDE":
        H0, OmegaM, log10_OmegaEDE0, log10_a_c, w_f = theta
        OmegaEDE0 = 10.0 ** log10_OmegaEDE0
        Omr = Omega_r_of_H0(H0)
        OmegaDE0 = 1.0 - OmegaM - Omr - OmegaEDE0
        return {
            "H0": H0,
            "OmegaM": OmegaM,
            "w0": -1.0,
            "w1": 0.0,
            "OmegaDE0": OmegaDE0,
            "OmegaEDE0": OmegaEDE0,
            "log10_a_c": log10_a_c,
            "w_f": w_f
        }

    if MODEL_CHOICE == "DDE_EDE":
        w0, w1, H0, OmegaM, log10_OmegaEDE0, log10_a_c, w_f = theta
        OmegaEDE0 = 10.0 ** log10_OmegaEDE0
        Omr = Omega_r_of_H0(H0)
        OmegaDE0 = 1.0 - OmegaM - Omr - OmegaEDE0
        return {
            "H0": H0,
            "OmegaM": OmegaM,
            "w0": w0,
            "w1": w1,
            "OmegaDE0": OmegaDE0,
            "OmegaEDE0": OmegaEDE0,
            "log10_a_c": log10_a_c,
            "w_f": w_f
        }

    if MODEL_CHOICE == "DDE":
        w0, w1, H0, OmegaM = theta
        Omr = Omega_r_of_H0(H0)
        OmegaDE0 = 1.0 - OmegaM - Omr
        return {
            "H0": H0,
            "OmegaM": OmegaM,
            "w0": w0,
            "w1": w1,
            "OmegaDE0": OmegaDE0,
            "OmegaEDE0": 0.0,
            "log10_a_c": None,
            "w_f": None
        }

    raise RuntimeError("Unhandled MODEL_CHOICE.")

def E2_model(z, theta):
    """
    E^2(z) assembled with one unified late-time DE variable OmegaDE0.

    LCDM:
      E2 = Or(1+z)^4 + Om(1+z)^3 + Ode0

    DDE:
      E2 = Or(1+z)^4 + Om(1+z)^3 + Ode0 * f_CPL(z)

    LCDM_EDE:
      E2 = Or(1+z)^4 + Om(1+z)^3 + Ode0 + Oede(a)

    DDE_EDE:
      E2 = Or(1+z)^4 + Om(1+z)^3 + Ode0 * f_CPL(z) + Oede(a)
    """
    z = np.asarray(z, dtype=float)
    p = unpack_theta(theta)

    H0 = p["H0"]
    OmegaM = p["OmegaM"]
    OmegaDE0 = p["OmegaDE0"]
    OmegaEDE0 = p["OmegaEDE0"]

    Omr = Omega_r_of_H0(H0)

    E2 = Omr * (1.0 + z) ** 4 + OmegaM * (1.0 + z) ** 3

    # late-time DE term
    if USE_DDE:
        E2 += OmegaDE0 * f_CPL_of_z(z, p["w0"], p["w1"])
    else:
        E2 += OmegaDE0

    # EDE term
    if USE_EDE:
        a = 1.0 / (1.0 + z)
        EDE = OmegaEDE_of_a(a, OmegaEDE0, p["log10_a_c"], p["w_f"])
        if np.any(~np.isfinite(EDE)):
            return np.full_like(z, np.nan, dtype=float)
        E2 += EDE

    if np.any(~np.isfinite(E2)) or np.any(E2 <= 0.0):
        return np.full_like(z, np.nan, dtype=float)

    return E2

def H_z(z, theta):
    z = np.asarray(z, dtype=float)
    H0 = unpack_theta(theta)["H0"]
    H0_si = H0 / H0_to_SI
    E2 = E2_model(z, theta)
    if np.any(~np.isfinite(E2)) or np.any(E2 <= 0.0):
        return np.full_like(z, np.nan, dtype=float)
    return H0_si * np.sqrt(E2)

def comoving_distance_trapz(z_in, theta):
    z_in = np.asarray(z_in, dtype=float)
    z_full = np.concatenate(([0.0], z_in))
    Hz = H_z(z_full, theta)
    if np.any(~np.isfinite(Hz)) or np.any(Hz <= 0.0):
        return np.full_like(z_in, np.nan, dtype=float)
    integrand = c / Hz
    dc_full = cumulative_trapezoid(integrand, z_full, initial=0.0)
    return dc_full[1:]

# ================================================================
# Drag redshift z_d (EH98)
# ================================================================
def z_drag_EH98(theta):
    p = unpack_theta(theta)
    H0 = p["H0"]
    OmegaM = p["OmegaM"]

    h = H0 / 100.0
    omega_m = OmegaM * h * h
    if omega_m <= 0.0:
        return np.nan

    b1 = 0.313 * omega_m ** (-0.419) * (1.0 + 0.607 * omega_m ** 0.674)
    b2 = 0.238 * omega_m ** 0.223
    z_d = 1291.0 * omega_m ** 0.251 / (1.0 + 0.659 * omega_m ** 0.828) * (1.0 + b1 * omega_b ** b2)
    return z_d

# ================================================================
# Sound horizon r_d
# ================================================================
def r_d_integral(theta, zmax=5e4, npts=1000):
    z_d = z_drag_EH98(theta)
    if (not np.isfinite(z_d)) or (z_d <= 0.0):
        return np.nan

    z_vals = np.logspace(np.log10(z_d), np.log10(zmax), npts)
    R = (3.0 * omega_b) / (4.0 * omega_gamma) * 1.0 / (1.0 + z_vals)
    c_s = c / np.sqrt(3.0 * (1.0 + R))

    Hz = H_z(z_vals, theta)
    if np.any(~np.isfinite(Hz)) or np.any(Hz <= 0.0):
        return np.nan

    integrand = c_s / Hz
    r_m = cumulative_trapezoid(integrand, z_vals, initial=0.0)[-1]
    r_mpc = r_m / MPC_M

    if (not np.isfinite(r_mpc)) or (r_mpc < 10.0) or (r_mpc > 300.0):
        return np.nan
    return r_mpc

# ================================================================
# observables
# ================================================================
def model_BAO(theta, z_array):
    z_array = np.asarray(z_array, dtype=float)

    Hz = H_z(z_array, theta)
    Dc = comoving_distance_trapz(z_array, theta) / MPC_M
    if np.any(~np.isfinite(Hz)) or np.any(~np.isfinite(Dc)):
        return np.full_like(z_array, np.nan, dtype=float)

    Dm = Dc
    Dh = (c / Hz) / MPC_M
    Dv = (Dm * Dm * Dh * z_array) ** (1.0 / 3.0)

    rd = r_d_integral(theta)
    if (not np.isfinite(rd)) or (rd <= 0.0):
        return np.full_like(z_array, np.nan, dtype=float)

    return (Dv / rd) * z_array ** (-2.0 / 3.0)

def model_ratio(theta, z_array):
    z_array = np.asarray(z_array, dtype=float)

    Hz = H_z(z_array, theta)
    Dc = comoving_distance_trapz(z_array, theta) / MPC_M
    if np.any(~np.isfinite(Hz)) or np.any(~np.isfinite(Dc)):
        return np.full_like(z_array, np.nan, dtype=float)

    Dm = Dc
    Dh = (c / Hz) / MPC_M
    return Dm / (z_array * Dh)

# ================================================================
# likelihood + prior
# ================================================================
def log_likelihood(theta, z_bao, y_bao, yerr_bao, z_rat, y_rat, yerr_rat):
    m_bao = model_BAO(theta, z_bao)
    m_rat = model_ratio(theta, z_rat)
    if np.any(~np.isfinite(m_bao)) or np.any(~np.isfinite(m_rat)):
        return -np.inf
    ll_bao = -0.5 * np.sum(((y_bao - m_bao) / yerr_bao) ** 2)
    ll_rat = -0.5 * np.sum(((y_rat - m_rat) / yerr_rat) ** 2)
    return ll_bao + ll_rat

def log_prior(theta):
    # hard box bounds
    for val, (lo, hi) in zip(theta, bounds):
        if not (lo < val < hi):
            return -np.inf

    p = unpack_theta(theta)
    H0 = p["H0"]
    OmegaM = p["OmegaM"]
    OmegaDE0 = p["OmegaDE0"]
    OmegaEDE0 = p["OmegaEDE0"]
    Omr = Omega_r_of_H0(H0)

    if OmegaM <= 0.0 or Omr <= 0.0:
        return -np.inf

    if OmegaDE0 < 0.0:
        return -np.inf

    if OmegaEDE0 < 0.0:
        return -np.inf

    total0 = OmegaM + Omr + OmegaDE0 + OmegaEDE0
    if not np.isfinite(total0):
        return -np.inf
    if total0 <= 0.0:
        return -np.inf

    # require valid E^2 at high z
    zstar = 1100.0
    E2star = E2_model(np.array([zstar]), theta)[0]
    if (not np.isfinite(E2star)) or (E2star <= 0.0):
        return -np.inf

    # suppress excessive early-time CPL dark energy if DDE present
    if USE_DDE and (OmegaDE0 > 0.0):
        f_de = f_CPL_of_z(np.array([zstar]), p["w0"], p["w1"])[0]
        OmegaDE_z = (OmegaDE0 * f_de) / E2star
        if OmegaDE_z > 0.005:
            return -np.inf

    # suppress excessive EDE near recombination if present
    if USE_EDE and (OmegaEDE0 > 0.0):
        a_star = 1.0 / (1.0 + zstar)
        OmegaEDE_zstar = float(
            OmegaEDE_of_a(np.array([a_star]), OmegaEDE0, p["log10_a_c"], p["w_f"])[0]
        )
        if (not np.isfinite(OmegaEDE_zstar)) or (OmegaEDE_zstar > 0.9999):
            return -np.inf

    rd = r_d_integral(theta, zmax=5e4, npts=1000)
    if not np.isfinite(rd):
        return -np.inf

    return 0.0

def log_probability(theta, z_bao, y_bao, yerr_bao, z_rat, y_rat, yerr_rat):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, z_bao, y_bao, yerr_bao, z_rat, y_rat, yerr_rat)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# ================================================================
# stats helpers
# ================================================================
def cred68(x):
    p16, p50, p84 = np.percentile(x, [16.0, 50.0, 84.0])
    return p50, (p50 - p16), (p84 - p50)

def print_cred(name, med, lo, hi, fmt="{:.3f}"):
    print(f"{name} = {fmt.format(med)} -{fmt.format(lo)} +{fmt.format(hi)}")

# ================================================================
# HDF helpers
# ================================================================
def _find_chain_group(h5file, expect_nwalkers=None, expect_ndim=None):
    best = None

    def visitor(name, obj):
        nonlocal best
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == "chain" and obj.ndim == 3:
            nsteps_here, nw, nd = obj.shape
            score = float(nsteps_here)
            if expect_nwalkers is not None and expect_ndim is not None:
                if (nw, nd) == (expect_nwalkers, expect_ndim):
                    score += 1e12
            group_path = "/".join(name.split("/")[:-1])
            if best is None or score > best[0]:
                best = (score, group_path)

    h5file.visititems(visitor)
    if best is None:
        raise RuntimeError(f"Could not find a 3D 'chain' dataset in {h5file.filename}")
    return best[1]

def _rewrite_backend_file(path, nwalkers, ndim):
    if os.path.exists(path):
        os.remove(path)
    b = HDFBackend(path)
    b.reset(nwalkers, ndim)
    return b

def write_trimmed_backend(orig_path, trim_path, discard, thin, nwalkers, ndim):
    if thin < 1:
        raise ValueError("thin must be >= 1")
    if discard < 0:
        raise ValueError("discard must be >= 0")
    if not os.path.exists(orig_path):
        raise FileNotFoundError(f"Original backend not found: {orig_path}")

    with h5py.File(orig_path, "r") as f_in:
        grp_in_path = _find_chain_group(f_in, expect_nwalkers=nwalkers, expect_ndim=ndim)
        g_in = f_in[grp_in_path]

        if "chain" not in g_in or "log_prob" not in g_in:
            raise RuntimeError(f"Group '{grp_in_path}' must contain 'chain' and 'log_prob'.")

        chain = g_in["chain"]
        logp = g_in["log_prob"]

        nsteps_here = chain.shape[0]
        if nsteps_here <= 0:
            raise RuntimeError("Original backend contains 0 steps.")

        discard = int(max(0, min(discard, nsteps_here - 1)))
        sel = slice(discard, None, thin)

        chain_trim = np.asarray(chain[sel, :, :])
        logp_trim = np.asarray(logp[sel, :])

        ntrim = chain_trim.shape[0]
        if ntrim <= 0:
            raise RuntimeError("Trim produced 0 steps.")

        root_attrs = dict(f_in.attrs.items())
        grp_attrs = dict(g_in.attrs.items())

    _rewrite_backend_file(trim_path, nwalkers, ndim)

    with h5py.File(trim_path, "r+") as f_out:
        grp_out_path = _find_chain_group(f_out, expect_nwalkers=nwalkers, expect_ndim=ndim)
        g_out = f_out[grp_out_path]

        for k, v in root_attrs.items():
            try:
                f_out.attrs[k] = v
            except Exception:
                pass

        for k, v in grp_attrs.items():
            try:
                g_out.attrs[k] = v
            except Exception:
                pass

        g_out["chain"].resize((ntrim, nwalkers, ndim))
        g_out["log_prob"].resize((ntrim, nwalkers))
        g_out["chain"][:, :, :] = chain_trim
        g_out["log_prob"][:, :] = logp_trim

        g_out.attrs["iteration"] = ntrim
        g_out.attrs["nwalkers"] = nwalkers
        g_out.attrs["ndim"] = ndim

    print(f"Trimmed backend written: {trim_path}")
    print(f"Kept steps: {ntrim} (discard={discard}, thin={thin})")

#%%
# ================================================================
# initial sanity check
# ================================================================
rd_init = r_d_integral(init)
print("Sanity check: r_d(init) [Mpc] =", rd_init)
if not np.isfinite(rd_init):
    raise RuntimeError("Initial parameters produce invalid r_d. Fix init/prior before running.")

# ================================================================
# backend and walkers
# ================================================================
backend_1 = _rewrite_backend_file(orig_path, nwalkers, ndim)
pos = init + pos_scales * np.random.randn(nwalkers, ndim)

def _ensure_valid_starts(pos, max_tries=3000):
    good = np.isfinite([
        log_probability(
            p, DESI_x_BAO, DESI_y_BAO, y_err_BAO,
            DESI_x_ratio, DESI_y_ratio, y_err_ratio
        )
        for p in pos
    ])

    tries = 0
    while np.sum(good) < max(10, nwalkers // 3) and tries < max_tries:
        bad_idx = np.where(~good)[0]
        pos[bad_idx] = init + repair_scales * np.random.randn(len(bad_idx), ndim)
        good = np.isfinite([
            log_probability(
                p, DESI_x_BAO, DESI_y_BAO, y_err_BAO,
                DESI_x_ratio, DESI_y_ratio, y_err_ratio
            )
            for p in pos
        ])
        tries += 1

    print(f"Valid starting walkers: {np.sum(good)}/{nwalkers}")
    if np.sum(good) == 0:
        raise RuntimeError("All starting positions invalid under prior/likelihood.")
    return pos

pos = _ensure_valid_starts(pos)

# ================================================================
# run MCMC
# ================================================================
sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(DESI_x_BAO, DESI_y_BAO, y_err_BAO, DESI_x_ratio, DESI_y_ratio, y_err_ratio),
    backend=backend_1
)

sampler.run_mcmc(pos, nsteps, progress=True)
total_steps = backend_1.iteration

print("\nTotal steps stored in ORIGINAL backend:", total_steps)
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# ================================================================
# burn-in from tau -> trimmed backend
# ================================================================
thin = 1
full_chain = backend_1.get_chain()

tau_list = []
for j, name in enumerate(param_names):
    try:
        tj = integrated_time(full_chain[:, :, j], tol=50, quiet=True)
        tj = np.atleast_1d(tj).astype(float)
        tau_list.append(float(tj[0]) if np.all(np.isfinite(tj)) else np.nan)
    except Exception:
        tau_list.append(np.nan)

tau_full = np.array(tau_list)
print("\nAutocorrelation times (FULL chain, per parameter):")
for name, tj in zip(param_names, tau_full):
    print(f"  {name:18s}: {tj}")

tau_finite = tau_full[np.isfinite(tau_full)]
if tau_finite.size == 0:
    discard = max(0, min(total_steps // 5, total_steps // 2, total_steps - 1))
    print("\nWARNING: All tau estimates invalid — using fallback discard.")
else:
    discard = int(5 * np.max(tau_finite))
    discard = max(0, min(discard, total_steps // 2, total_steps - 1))

print(f"\nUsing discard (burn-in) = {discard}")

write_trimmed_backend(
    orig_path=orig_path,
    trim_path=trim_path,
    discard=discard,
    thin=thin,
    nwalkers=nwalkers,
    ndim=ndim
)

trim_backend = HDFBackend(trim_path)

trim_chain = trim_backend.get_chain()
try:
    tau_trim = integrated_time(trim_chain, tol=50, quiet=True)
    print("\nAutocorrelation times (TRIMMED chain):", tau_trim)
except Exception as e:
    print("\nWARNING: tau estimate from TRIMMED chain not reliable.")
    print("Reason:", e)

#%%
# ================================================================
# posterior analysis
# ================================================================
samples = trim_backend.get_chain(flat=True)
logp_samples = trim_backend.get_log_prob(flat=True)

if samples.size == 0:
    raise RuntimeError("Trimmed backend contains no samples.")

# unified posterior arrays
p_list = [unpack_theta(s) for s in samples]

H0_s = np.array([p["H0"] for p in p_list])
OmegaM_s = np.array([p["OmegaM"] for p in p_list])
OmegaR_s = Omega_r_of_H0(H0_s)
OmegaDE0_s = np.array([p["OmegaDE0"] for p in p_list])
OmegaEDE0_s = np.array([p["OmegaEDE0"] for p in p_list])

print("\n==== 68% Bayesian credible intervals (TRIMMED chain) ====\n")

for j, name in enumerate(param_names):
    med, lo, hi = cred68(samples[:, j])
    fmt = "{:.3f}"
    if name == "H0":
        fmt = "{:.2f}"
    print_cred(name, med, lo, hi, fmt=fmt)

# derived parameters
oR_med, oR_lo, oR_hi = cred68(OmegaR_s[np.isfinite(OmegaR_s)])
oDE_med, oDE_lo, oDE_hi = cred68(OmegaDE0_s[np.isfinite(OmegaDE0_s)])
oEDE_med, oEDE_lo, oEDE_hi = cred68(OmegaEDE0_s[np.isfinite(OmegaEDE0_s)])

print_cred("Omega_r", oR_med, oR_lo, oR_hi, fmt="{:.6e}")
print_cred("Omega_DE0", oDE_med, oDE_lo, oDE_hi, fmt="{:.3f}")
print_cred("Omega_EDE0", oEDE_med, oEDE_lo, oEDE_hi, fmt="{:.6e}")

# ================================================================
# best chain sample and true MLE
# ================================================================
chain3 = trim_backend.get_chain()
lp3 = trim_backend.get_log_prob()

imax = np.unravel_index(np.nanargmax(lp3), lp3.shape)
theta_best_chain = chain3[imax]

def chi2_parts(theta):
    m_bao = model_BAO(theta, DESI_x_BAO)
    m_rat = model_ratio(theta, DESI_x_ratio)
    if np.any(~np.isfinite(m_bao)) or np.any(~np.isfinite(m_rat)):
        return np.inf, np.inf, np.inf
    chi2_bao = np.sum(((DESI_y_BAO - m_bao) / y_err_BAO) ** 2)
    chi2_rat = np.sum(((DESI_y_ratio - m_rat) / y_err_ratio) ** 2)
    return chi2_bao + chi2_rat, chi2_bao, chi2_rat

def nll_joint(theta):
    if not np.isfinite(log_prior(theta)):
        return 1e30
    chi2_tot, _, _ = chi2_parts(theta)
    if not np.isfinite(chi2_tot):
        return 1e30
    return 0.5 * chi2_tot

res = minimize(
    nll_joint,
    x0=theta_best_chain,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 4000}
)

theta_mle = res.x
if not np.isfinite(log_prior(theta_mle)) or (not res.success):
    print("\nWARNING: optimiser failed or left prior bounds — using best chain sample as MLE.")
    theta_mle = theta_best_chain

p_mle = unpack_theta(theta_mle)
p_best = unpack_theta(theta_best_chain)

H0_si_mle = p_mle["H0"] / H0_to_SI
rho_c0_si = 3.0 * H0_si_mle**2 / (8.0 * np.pi * G)   # kg m^-3

chi2_tot_mle, chi2_bao_mle, chi2_rat_mle = chi2_parts(theta_mle)
chi2_tot_best, chi2_bao_best, chi2_rat_best = chi2_parts(theta_best_chain)

ndata = len(DESI_x_BAO) + len(DESI_x_ratio)
dof = ndata - ndim
if dof > 0:
    print(f"chi2/dof(best) = {chi2_tot_best / dof:.3f}")
    print(f"chi2/dof(MLE)  = {chi2_tot_mle / dof:.3f}")
else:
    print("WARNING: dof <= 0; reduced chi2 not meaningful.")

rd_best = r_d_integral(theta_best_chain)
rd_mle = r_d_integral(theta_mle)

print("\n==== BEST CHAIN SAMPLE (max log_prob) ====\n")
print("theta_best_chain =", theta_best_chain)
print(f"chi2_total(best) = {chi2_tot_best:.3f}  (BAO {chi2_bao_best:.3f} + ratio {chi2_rat_best:.3f})")
print(f"r_d(best) [Mpc]  = {rd_best:.3f}")

print("\n==== TRUE JOINT MLE (post-optimisation; DESI-only) ====\n")
for name, val in zip(param_names, theta_mle):
    if name == "H0":
        print(f"{name} = {val:.2f}")
    else:
        print(f"{name} = {val:.6g}")
print(f"Omega_r(H0)    = {Omega_r_of_H0(p_mle['H0']):.6e}")
print(f"Omega_DE0      = {p_mle['OmegaDE0']:.6e}")
print(f"Omega_EDE0     = {p_mle['OmegaEDE0']:.6e}")
print(f"r_d(MLE) [Mpc] = {rd_mle:.3f}")
print(f"chi2_total(MLE)= {chi2_tot_mle:.3f}  (BAO {chi2_bao_mle:.3f} + ratio {chi2_rat_mle:.3f})")
print("Optimiser success:", res.success, "| message:", res.message)

# ================================================================
# curves
# ================================================================
bao_best_chain = model_BAO(theta_best_chain, z_plot)
bao_mle = model_BAO(theta_mle, z_plot)
rat_best_chain = model_ratio(theta_best_chain, z_plot)
rat_mle = model_ratio(theta_mle, z_plot)

# posterior predictive bands
rng = np.random.default_rng(123)
ndraw = min(600, samples.shape[0])
idx = rng.choice(samples.shape[0], size=ndraw, replace=False)

nz = z_plot.size
bao_draws = np.empty((ndraw, nz), dtype=np.float32)
rat_draws = np.empty((ndraw, nz), dtype=np.float32)

for k, i in enumerate(idx):
    bao_draws[k] = model_BAO(samples[i], z_plot).astype(np.float32)
    rat_draws[k] = model_ratio(samples[i], z_plot).astype(np.float32)

bao_p16, bao_p50, bao_p84 = np.nanpercentile(bao_draws, [16.0, 50.0, 84.0], axis=0)
rat_p16, rat_p50, rat_p84 = np.nanpercentile(rat_draws, [16.0, 50.0, 84.0], axis=0)
bao_p2p5, bao_p97p5 = np.nanpercentile(bao_draws, [2.5, 97.5], axis=0)
rat_p2p5, rat_p97p5 = np.nanpercentile(rat_draws, [2.5, 97.5], axis=0)

theta_med = np.median(samples, axis=0)
bao_med = model_BAO(theta_med, z_plot)
rat_med = model_ratio(theta_med, z_plot)

# ================================================================
# plot styling
# ================================================================
TITLE_FS = 26
LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 18
LINE_W = 3
ERR_MS = 6

#%%
# ================================================================
# Plot 1: BAO observable
# ================================================================
plt.figure(figsize=fig_size)
plt.errorbar(
    DESI_x_BAO, DESI_y_BAO, yerr=y_err_BAO, fmt='sr',
    label="DESI BAO data", markersize=ERR_MS, color='k'
)
plt.title(f'BAO observable, MCMC fit for {MODEL_CHOICE}', fontsize=TITLE_FS)
plt.plot(z_plot, bao_mle, label="TRUE joint MLE (optimised)", color='k', linewidth=LINE_W)
plt.plot(
    z_plot, bao_best_chain, alpha=0.7, label="Best chain sample (max log_prob)",
    color='red', linestyle=":", linewidth=LINE_W
)
plt.plot(
    z_plot, bao_med, linestyle="--", label="Model at median params",
    color=MED_COLOR, alpha=0.7, linewidth=LINE_W
)
plt.fill_between(
    z_plot, bao_p2p5, bao_p97p5, alpha=0.18,
    label="95% (outer) credible band", color=BAND_COLOR
)
plt.fill_between(
    z_plot, bao_p16, bao_p84, alpha=0.35,
    label="68% (inner) credible band", color=BAND_COLOR
)
plt.xlabel("Redshift z", fontsize=LABEL_FS)
plt.ylabel(r"$\dfrac{D_V(z)}{r_d}\,z^{-2/3}$", fontsize=LABEL_FS)
plt.ylim(15, 24)
plt.xticks(fontsize=TICK_FS)
plt.yticks(fontsize=TICK_FS)
plt.legend(fontsize=LEGEND_FS)
plt.tight_layout()
plt.show()

# ================================================================
# Plot 2: LOS ratio observable
# ================================================================
plt.figure(figsize=fig_size)
plt.errorbar(
    DESI_x_ratio, DESI_y_ratio, yerr=y_err_ratio, fmt='sr',
    label="DESI LOS-ratio data", markersize=ERR_MS, color='k'
)
plt.title(f'LOS-ratio observable, MCMC fit for {MODEL_CHOICE}', fontsize=TITLE_FS)
plt.plot(z_plot, rat_mle, label="TRUE joint MLE (optimised)", color='k', linewidth=LINE_W)
plt.plot(
    z_plot, rat_best_chain, alpha=0.7, label="Best chain sample (max log_prob)",
    color='red', linestyle=":", linewidth=LINE_W
)
plt.plot(
    z_plot, rat_med, linestyle="--", label="Model at median params",
    color=MED_COLOR, alpha=0.7, linewidth=LINE_W
)
plt.fill_between(
    z_plot, rat_p2p5, rat_p97p5, alpha=0.18,
    label="95% (outer) credible band", color=BAND_COLOR
)
plt.fill_between(
    z_plot, rat_p16, rat_p84, alpha=0.35,
    label="68% (inner) credible band", color=BAND_COLOR
)
plt.xlabel("Redshift z", fontsize=LABEL_FS)
plt.ylabel(r"$\dfrac{D_M(z)}{z\,D_H(z)}$", fontsize=LABEL_FS)
plt.ylim(0.9, 2.0)
plt.xticks(fontsize=TICK_FS)
plt.yticks(fontsize=TICK_FS)
plt.legend(fontsize=LEGEND_FS)
plt.tight_layout()
plt.show()

# ================================================================
# Plot 3: rho-density contributions (scaled by present critical density)
# ================================================================
z_evol = np.logspace(-5, 10, 2000)
a_evol = 1.0 / (1.0 + z_evol)

# Total E^2
E2_evol = E2_model(z_evol, theta_mle)

# Present-day radiation density parameter from H0
Omega_r0_mle = Omega_r_of_H0(p_mle["H0"])

# Individual density contributions in units of rho_c,0
rho_r_tilde = Omega_r0_mle * (1.0 + z_evol) ** 4
rho_M_tilde = p_mle["OmegaM"] * (1.0 + z_evol) ** 3

if USE_DDE:
    rho_DE_tilde = p_mle["OmegaDE0"] * f_CPL_of_z(z_evol, p_mle["w0"], p_mle["w1"])
else:
    rho_DE_tilde = np.full_like(z_evol, p_mle["OmegaDE0"], dtype=float)

if USE_EDE and (p_mle["OmegaEDE0"] > 0.0):
    rho_EDE_tilde = OmegaEDE_of_a(a_evol, p_mle["OmegaEDE0"], p_mle["log10_a_c"], p_mle["w_f"])
else:
    rho_EDE_tilde = np.zeros_like(z_evol)

rho_r_si = rho_r_tilde * rho_c0_si
rho_M_si = rho_M_tilde * rho_c0_si
rho_DE_si = rho_DE_tilde * rho_c0_si
rho_EDE_si = rho_EDE_tilde * rho_c0_si

plt.figure(figsize=fig_size)
plt.plot(z_evol, rho_r_si, color='crimson', linewidth=LINE_W, label=r'$\rho_r$')
plt.plot(z_evol, rho_M_si, color='goldenrod', linewidth=LINE_W, label=r'$\rho_M$')
plt.plot(z_evol, rho_DE_si, color='royalblue', linewidth=LINE_W, label=r'$\rho_{DE}$')
if np.any(rho_EDE_tilde > 0):
    plt.plot(z_evol, rho_EDE_si, color='forestgreen', linewidth=LINE_W, label=r'$\rho_{EDE}$')

plt.title(f'Energy-density contributions for {MODEL_CHOICE}', fontsize=TITLE_FS)
plt.xlabel("Redshift z", fontsize=LABEL_FS)
plt.ylabel(r"Density contribution [$kgm^{-3}$]", fontsize=LABEL_FS)
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=TICK_FS)
plt.yticks(fontsize=TICK_FS)
plt.legend(fontsize=LEGEND_FS)
plt.tight_layout()
plt.show()

# ================================================================
# Plot 4: Omega_i(z) fractional density parameters
# ================================================================
Omega_r_z = rho_r_tilde / E2_evol
Omega_M_z = rho_M_tilde / E2_evol
Omega_DE_z = rho_DE_tilde / E2_evol
Omega_EDE_z = rho_EDE_tilde / E2_evol

plt.figure(figsize=fig_size)
plt.plot(z_evol, Omega_r_z, color='crimson', linewidth=LINE_W, label=r'$\Omega_r(z)$')
plt.plot(z_evol, Omega_M_z, color='goldenrod', linewidth=LINE_W, label=r'$\Omega_M(z)$')
plt.plot(z_evol, Omega_DE_z, color='royalblue', linewidth=LINE_W, label=r'$\Omega_{DE}(z)$')
if np.any(Omega_EDE_z > 0):
    plt.plot(z_evol, Omega_EDE_z, color='forestgreen', linewidth=LINE_W, label=r'$\Omega_{EDE}(z)$')

plt.title(f'Fractional density parameters for {MODEL_CHOICE}', fontsize=TITLE_FS)
plt.xlabel("Redshift z", fontsize=LABEL_FS)
plt.ylabel(r"$\Omega_i(z) = \rho_i(z)/\rho_c(z)$", fontsize=LABEL_FS)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-12, 2)
plt.xticks(fontsize=TICK_FS)
plt.yticks(fontsize=TICK_FS)
plt.legend(fontsize=LEGEND_FS)
plt.tight_layout()
plt.show()

# ================================================================
# Plot 5: diagnostics - parameters
# ================================================================
chain_trim = trim_backend.get_chain()

fig, axes = plt.subplots(len(param_names), 1, figsize=(24, 14), sharex=True)
if len(param_names) == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    m = np.mean(chain_trim[:, :, i], axis=1)
    s = np.std(chain_trim[:, :, i], axis=1)
    ax.plot(m, color='forestgreen')
    ax.fill_between(np.arange(len(m)), m - s, m + s, alpha=0.25, facecolor='mediumseagreen')
    ax.set_ylabel(param_names[i])

axes[-1].set_xlabel("step (trimmed)")
plt.suptitle(f'Diagnostic MCMC variable plots: {MODEL_CHOICE}', fontsize=20)
plt.tight_layout()
plt.show()

# ================================================================
# Plot 6: diagnostics - log posterior
# ================================================================
logp_trim = trim_backend.get_log_prob()
lp_med = np.median(logp_trim, axis=1)
lp_p16 = np.percentile(logp_trim, 16, axis=1)
lp_p84 = np.percentile(logp_trim, 84, axis=1)

plt.figure(figsize=(20, 8))
plt.title(f'Diagnostic MCMC likelihood plot: {MODEL_CHOICE}')
plt.plot(lp_med, color='forestgreen')
plt.fill_between(
    np.arange(len(lp_med)), lp_p16, lp_p84,
    alpha=0.25, facecolor='mediumseagreen'
)
plt.ylabel("log posterior (median across walkers)")
plt.xlabel("step (trimmed)")
plt.tight_layout()
plt.show()
#%%