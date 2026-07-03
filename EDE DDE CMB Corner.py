# -*- coding: utf-8 -*-
"""
Corner plot from an emcee HDFBackend (trimmed chain).

- Plots ONLY parameters stored in the backend (MCMC vars).
- Smooth filled contours.
- Optional extra discard/thin (even if already trimmed).
"""

import numpy as np
import corner
import matplotlib.pyplot as plt
from emcee.backends import HDFBackend

# ================================================================
# Load backend
# ================================================================
trim_path = r"C:\Users\hazmc\Downloads\DESI_MCMC_joint_chain_DDE_Planck.h5"
backend = HDFBackend(trim_path)

# How many steps exist?
nsteps = backend.iteration
if nsteps == 0:
    raise RuntimeError("Backend contains 0 steps.")

print("Backend steps:", nsteps)

discard = 0   # extra discard on top of trimming (keep 0 unless you need more)
thin    = 1   # thin>1 can make contours cleaner if chain is huge

# Pull flattened samples: shape (Nsamples, ndim)
samples = backend.get_chain(discard=discard, thin=thin, flat=True)
if samples.size == 0:
    raise RuntimeError("No samples after discard/thin. Reduce discard or thin.")

ndim = samples.shape[1]
print(f"Loaded samples: {samples.shape[0]} x {ndim}")

log_prob = backend.get_log_prob(discard=discard, thin=thin, flat=True)

weights = np.exp(log_prob - np.max(log_prob))  # stabilised

# Sort by probability (descending)
idx = np.argsort(weights)[::-1]

weights_sorted = weights[idx]
samples_sorted = samples[idx]

# Cumulative probability
cum_prob = np.cumsum(weights_sorted)
cum_prob /= cum_prob[-1]

# Keep top 95%
mask = cum_prob <= 0.95

samples_95 = samples_sorted[mask]



# ================================================================
# Labels based on model (inferred from filename)
# ================================================================
if "LCDM" in trim_path and "EDE" not in trim_path:
    labels = [
        r"$H_0 [km/s/Mpc]$",
        r"$\Omega_{M,0}$"
    ]

elif "DDE" in trim_path and "EDE" not in trim_path:
    labels = [
        r"$w_0$",
        r"$w_1$",
        r"$H_0 [km/s/Mpc]$",
        r"$\Omega_{M,0}$"
    ]

elif "LCDM_EDE" in trim_path:
    labels = [
        r"$H_0 [km/s/Mpc]$",
        r"$\Omega_{M,0}$",
        r"$\log_{10}\Omega_{\rm EDE,0}$",
        r"$\log_{10}a_c$",
        r"$w_f$"
    ]

elif "DDE_EDE" in trim_path:
    labels = [
        r"$w_0$",
        r"$w_1$",
        r"$H_0 [km/s/Mpc]$",
        r"$\Omega_{M,0}$",
        r"$\log_{10}\Omega_{\rm EDE,0}$",
        r"$\log_{10}a_c$",
        r"$w_f$"
    ]

else:
    raise ValueError("Could not infer model from filename. Set labels manually.")

# ================================================================
# Corner plot styling (smooth contours like your old ones)
# ================================================================
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})

fig = plt.figure(figsize=(14, 14))

fig = corner.corner(
    samples_95,
    labels=labels,
    fig=fig,
    range=[
        (-2, 1),
        (-4, 2),
        (60, 75),
        (0.25, 0.4)
    ],
    bins=50,                 # 45–60 tends to look good
    smooth=1.2,              # 2D smoothing (try 1.0–2.0)
    smooth1d=1.0,            # 1D smoothing
    plot_contours=True,
    fill_contours=True,
    levels=(0.68, 0.95),     # keep clean: 68% + 95% like most papers
    show_titles=False,
    title_fmt=".3f",
    title_kwargs={"fontsize": 14},
    label_kwargs={"fontsize": 25},
    color="mediumseagreen", 
    contour_kwargs={"linewidths": 1.5},
    hist_kwargs={"linewidth": 1.2},
)

plt.tight_layout()
plt.show()