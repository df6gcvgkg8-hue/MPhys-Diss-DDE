import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from astropy.cosmology import FlatLambdaCDM

c = 299792.458  # km/s

cosmologies = [
    (67.0, 0.27),
    (67.5, 0.315),
    (70.0, 0.30),
    (73.0, 0.30),
    (75.0, 0.35),
]

z = np.linspace(1e-5, 2.5, 2000)

plt.figure(figsize=(10, 6))

for H0, OmegaM in cosmologies:
    OmegaL = 1.0 - OmegaM
    cosmo = FlatLambdaCDM(H0=H0, Om0=OmegaM)

    # Astropy result
    D_M_astropy = cosmo.comoving_distance(z).value

    # Manual integration
    E = np.sqrt(OmegaM * (1 + z)**3 + OmegaL)
    H = H0 * E
    D_M_manual = cumulative_trapezoid(c / H, z, initial=0)

    # Differences
    abs_diff = np.abs(D_M_manual - D_M_astropy)
    frac_diff = abs_diff / D_M_astropy

    # Remove low-z instability
    mask = z > 0.01

    mean_frac_diff = np.mean(frac_diff[mask])

    print(f"H0 = {H0:.1f}, Omega_m = {OmegaM:.3f}")
    print(f"  Mean fractional difference = {mean_frac_diff*100:.6e} %")
    print()

    # --- Astropy (solid grey) ---
    plt.plot(
        z, D_M_astropy,
        color='grey',
        alpha=1.0,
        linewidth=2
    )

    # --- Manual (dashed coloured) ---
    plt.plot(
        z, D_M_manual,
        '--',
        linewidth=2,
        label=(
            rf"$H_0={H0:.1f}$, $\Omega_m={OmegaM:.3f}$ : "
            rf"$\Delta={mean_frac_diff*100:.2e}\%$"
        )
    )

plt.xlabel("Redshift $z$")
plt.ylabel(r"Comoving Distance $D_M$ [Mpc]")
plt.title("Astropy (grey) vs Manual Integration (coloured dashed)")

plt.legend(fontsize=11)  # increased from 8 → 11

plt.grid(True)
plt.tight_layout()
plt.show()