# BAO Cosmology Fitting Pipeline (DESI Data)

## Overview

This project implements a Bayesian inference pipeline for fitting Baryon Acoustic Oscillation (BAO) measurements from DESI data within standard ΛCDM and dynamical dark energy cosmological models using the CPL (Chevallier–Polarski–Linder) parameterisation (w₀–wₐ).

The pipeline performs self-consistent cosmological calculations, including distance measures and the sound horizon, and uses Markov Chain Monte Carlo (MCMC) methods to infer cosmological parameters from observational BAO data.

---

## Scientific Motivation

Baryon Acoustic Oscillations provide a standard ruler for probing the expansion history of the Universe. By comparing theoretical predictions with observational data, we can constrain key cosmological parameters such as:

- Matter density (Ωₘ)
- Hubble constant (H₀)
- Dark energy equation of state (w₀, wₐ)

This project explores how different cosmological models affect inferred expansion histories and constraints on dark energy.

---

## Models Implemented

### ΛCDM
- Flat ΛCDM baseline model
- Optional extensions for curvature (if included)

### CPL Dynamical Dark Energy
The equation of state evolves as:

w(a) = w₀ + wₐ(1 − a)

This allows time-varying dark energy behaviour.

---

## Methodology

### Bayesian Framework
The pipeline performs parameter inference using Bayes’ theorem:

- Likelihood: Gaussian likelihood from BAO measurements
- Priors: Physically motivated parameter bounds
- Posterior: Sampled via MCMC

### Numerical Computations
- Hubble parameter H(z) computed from cosmological model
- Comoving distance evaluated via numerical integration
- Sound horizon r_d calculated self-consistently
- Key observable:
  
  D_V(z) / r_d

---

## Data

The pipeline uses Baryon Acoustic Oscillation measurements from the DESI survey:

- Redshift-binned BAO distance constraints
- Covariance-aware likelihood construction

---

## Implementation

### Core Technologies
- Python
- NumPy
- SciPy
- emcee (MCMC sampling)
- Matplotlib

### Pipeline Structure

The project is modular and consists of:

- **Cosmology module**
  - H(z) functions
  - Distance calculations
  - Sound horizon computation

- **Likelihood module**
  - BAO observable predictions
  - Comparison with DESI data
  - Log-likelihood evaluation

- **Sampler module**
  - MCMC implementation using `emcee`
  - Parameter exploration and convergence tracking

- **Analysis module**
  - Posterior processing
  - Corner plots
  - Trace plots
  - Best-fit model extraction

---

## Outputs

The pipeline generates:

- Posterior distributions for cosmological parameters
- Corner plots of parameter constraints
- MCMC trace plots for convergence diagnostics
- BAO distance-redshift fits
- Derived constraints on:
  - H₀
  - Ωₘ
  - w₀
  - wₐ

---

## Key Features

- Fully self-consistent cosmological distance calculations
- Flexible model framework (ΛCDM and CPL dark energy)
- Bayesian inference using MCMC sampling
- Covariance-aware BAO likelihood
- Reproducible scientific workflow



## Future Work

- Inclusion of Planck CMB likelihoods
- Joint BAO + Type Ia Supernova constraints
- Bayesian evidence calculation for model comparison
- Performance optimisation for large-scale sampling

---

## References

- Abazajian, Kevork et al. (2019). “CMB-S4 Science Case, Reference Design, and Project Plan”. arXiv e-prints, arXiv:1907.04473.
- Abdul Karim, M. et al. (2025). “DESI DR2 results. II. Measurements of baryon acoustic oscillations and cosmological constraints”. Physical Review D, 112(8), 083515.
- Adame, A. G. et al. (2025). “DESI 2024 VI: cosmological constraints from the measurements of baryon acoustic oscillations”. JCAP 2025(2), 021.
- Chevallier, Michel and David Polarski (2001). “Accelerating Universes with Scaling Dark Matter”. International Journal of Modern Physics D, 10(2), 213–224.
- DESI Collaboration et al. (2016). “The DESI Experiment Part I: Science, Targeting, and Survey Design”. arXiv e-prints, arXiv:1611.00036.
- Di Valentino, Eleonora et al. (2021). “In the realm of the Hubble tension—a review of solutions”. Classical and Quantum Gravity, 38(15), 153001.
- du Mas des Bourboux, Hélion et al. (2017). “Baryon acoustic oscillations from the complete SDSS-III Lyα quasar cross-correlation function at z = 2.4”. Astronomy & Astrophysics, 608, A130.
- Einstein, Albert (1915). “Die Feldgleichungen der Gravitation”. Sitzungsberichte der Königlich Preussischen Akademie der Wissenschaften.
- Eisenstein, Daniel J. and Wayne Hu (1998). “Baryonic Features in the Matter Transfer Function”. The Astrophysical Journal, 496(2), 605–614.
- Foreman-Mackey, Daniel et al. (2013). “emcee: The MCMC Hammer”. Publications of the Astronomical Society of the Pacific, 125(925), 306–312.
- Friedmann, A. (1922). “Über die Krümmung des Raumes”. Zeitschrift für Physik, 10, 377–386.
- Goodman, Jonathan and Jonathan Weare (2010). “Ensemble samplers with affine invariance”. Communications in Applied Mathematics and Computational Science, 5(1), 65–80.
- Hastings, W. K. (1970). “Monte Carlo Sampling Methods Using Markov Chains and Their Applications”. Biometrika, 57(1), 97–109.
- Hubble, Edwin (1929). “A Relation between Distance and Radial Velocity among Extra-Galactic Nebulae”. Proceedings of the National Academy of Sciences, 15(3), 168–173.
- Koopmans, Leon V. E. et al. (2015). “The Cosmic Dawn and Epoch of Reionisation with the Square Kilometre Array”. Proceedings of Science AASKA14, 001.
- Lemaître, Georges (1927). “Un Univers homogène de masse constante et de rayon croissant rendant compte de la vitesse radiale des nébuleuses extra-galactiques”. Annales de la Société Scientifique de Bruxelles, 47, 49–59.
- Liddle, Andrew R. (2007). “Information criteria for astrophysical model selection”. Monthly Notices of the Royal Astronomical Society, 377(1), L74–L78.
- Linder, Eric V. (2003). “Exploring the Expansion History of the Universe”. Physical Review Letters, 90(9), 091301.
- Perlmutter, Saul et al. (1999). “Measurements of Ω and Λ from 42 High-Redshift Supernovae”. The Astrophysical Journal, 517(2), 565–586.
- Planck Collaboration et al. (2020). “Planck 2018 results. VI. Cosmological parameters”. Astronomy & Astrophysics, 641, A6.
- Poulin, Vivian et al. (2019). “Early Dark Energy Can Resolve the Hubble Tension”. Physical Review Letters, 122(22), 221301.
- Riess, Adam G. et al. (2022). “A Comprehensive Measurement of the Hubble Constant with 1 km s⁻¹ Mpc⁻¹ Uncertainty from the Hubble Space Telescope and the SH0ES Team”. The Astrophysical Journal Letters, 934(1), L7.
- Robertson, H. P. (1935). “Kinematics and World-Structure”. Proceedings of the National Academy of Sciences, 82, 284.
- Scolnic, D. et al. (2022). “The Pantheon+ Analysis: The Full Data Set and Light-curve Release”. The Astrophysical Journal, 938(2), 113.

---

## Author

Harry McClelland
MSc Astrophysics graduate specialising in cosmology, dark energy modelling, and statistical analysis of large-scale structure surveys.

---

## Notes

This project is intended for research and educational purposes in observational cosmology and Bayesian parameter inference.
