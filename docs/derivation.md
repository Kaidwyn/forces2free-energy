# From the partition function to the heat capacity: the harmonic derivation

These notes go with `src/forces2free/thermo.py`. Work through them on paper first, then compare.

## 1. One quantum harmonic oscillator

An oscillator of frequency $\nu$ has the energy levels

$$E_n = \left(n + \tfrac{1}{2}\right) h\nu, \qquad n = 0, 1, 2, \dots$$

With $x = \dfrac{h\nu}{k_B T}$, the partition function is a geometric series:

$$q = \sum_{n=0}^{\infty} e^{-E_n / k_B T} = e^{-x/2} \sum_{n=0}^{\infty} e^{-n x} = \frac{e^{-x/2}}{1 - e^{-x}}$$

Every thermodynamic quantity follows from it:

| Quantity | Definition | Result |
|---|---|---|
| Helmholtz free energy | $F = -k_B T \ln q$ | $F = \dfrac{h\nu}{2} + k_B T \ln\left(1 - e^{-x}\right)$ |
| Internal energy | $U = -\dfrac{\partial \ln q}{\partial \beta}$, $\beta = \dfrac{1}{k_B T}$ | $U = \dfrac{h\nu}{2} + \dfrac{h\nu}{e^{x} - 1}$ |
| Entropy | $S = \dfrac{U - F}{T}$ | $S = k_B\left[\dfrac{x}{e^{x} - 1} - \ln\left(1 - e^{-x}\right)\right]$ |
| Heat capacity at constant volume | $C_V = \left(\dfrac{\partial U}{\partial T}\right)_V$ | $C_V = k_B \dfrac{x^2 e^{x}}{\left(e^{x} - 1\right)^2}$ |

$\dfrac{1}{e^x - 1}$ is the Bose–Einstein occupation: the mean number of phonons in the mode.
To get $C_V$, differentiate $U$ using $\dfrac{dx}{dT} = -\dfrac{x}{T}$ and $\dfrac{h\nu}{T} = k_B x$.

## 2. A crystal is many independent oscillators

The harmonic approximation expands the energy to second order around the equilibrium positions:

$$E = E_0 + \frac{1}{2} \sum_{ij} \Phi_{ij} u_i u_j, \qquad F_i = -\frac{\partial E}{\partial u_i} = -\sum_j \Phi_{ij} u_j$$

So the force constants are $\Phi_{ij} \approx -\Delta F_i / \Delta u_j$: displace one atom by 0.01 Å and see how the
forces on the other atoms change. That is what phonopy does, with the forces supplied by the ML potential.
The force constants give the dynamical matrix, whose eigenvalues give the phonon frequencies $\nu_{\mathbf{q}j}$ of
each branch $j$ at each wave vector $\mathbf{q}$.

Each normal mode is an independent oscillator. The total partition function is the product of the mode
partition functions, so $\ln Q$, $F$, $U$, $S$ and $C_V$ are sums over modes. On a q-point mesh, where $w_\mathbf{q}$ is
the weight of each irreducible q-point:

$$\text{value per primitive cell} = \frac{1}{\sum_\mathbf{q} w_\mathbf{q}} \sum_\mathbf{q} w_\mathbf{q} \sum_j f\left(\nu_{\mathbf{q}j}\right)$$

## 3. Units: where mistakes happen

1. phonopy gives the ordinary frequency $\nu$ in THz, not the angular frequency $\omega = 2\pi\nu$. The energy of one
   phonon is $h\nu = \hbar\omega$. Using $\nu$ in place of $\omega$ puts energies off by a factor $2\pi$.
2. The sum above is a value per primitive cell. Multiplying by the Avogadro constant $N_A$ gives a value per mole of
   primitive cells, which is what phonopy reports.
3. JANAF reports values per mole of formula units, so divide by the number of formula units $Z$ in the primitive cell.
   The primitive cell of silicon holds two Si atoms, so $Z = 2$; forgetting this doubles the entropy and heat capacity.

## 4. Limits that test the code

- **High temperature** ($x \to 0$): $C_V \to k_B$ per mode. Each atom has three modes, so per mole of atoms
  $C_V \to 3R \approx 24.94$ J/(K·mol), the Dulong–Petit law.
- **Low temperature**: only low-frequency acoustic modes are excited, and the Debye model gives $C_V \propto T^3$.
- **Absolute zero**: $S = 0$ and $C_V = 0$, but $F = U = \sum \dfrac{h\nu}{2} \ne 0$: the zero-point energy.
- **Thermodynamic consistency**: $S = -\left(\dfrac{\partial F}{\partial T}\right)_V$,
  $C_V = T\left(\dfrac{\partial S}{\partial T}\right)_V$, $U = F + TS$.

`tests/test_thermo.py` checks each of these.

## 5. Why the code is written with $e^{-x}$

At low temperature or high frequency $x$ can reach thousands and $e^{x}$ overflows; at high temperature $x$ is tiny
and $1 - e^{-x}$ loses precision. The code therefore writes every formula with $e^{-x}$ only and computes
$1 - e^{-x}$ with `expm1`:

$$n = \frac{e^{-x}}{1 - e^{-x}}, \qquad C_V = k_B \frac{x^2 e^{-x}}{\left(1 - e^{-x}\right)^2}$$

## 6. Why the acoustic modes at $\Gamma$ are left out

At $\Gamma$ ($\mathbf{q} = 0$) the three acoustic modes are rigid translations of the whole crystal: their frequency
is exactly zero and they are not vibrations. Numerically they come out as positive or negative numbers of order
$10^{-6}$ THz. A tiny positive frequency put into the entropy formula gives $-\ln(1 - e^{-x}) \approx -\ln x$,
which is large, so the result would carry an error set by numerical noise. They are therefore set to zero and
excluded before summing.

## 7. $C_V$ is not $C_P$

In the harmonic approximation the volume is fixed, so it gives $C_V$; experiment (JANAF) measures $C_P$ at 1 bar.
The difference is

$$C_P - C_V = \alpha_V^2 B V_m T$$

where $\alpha_V$ is the volumetric thermal expansion coefficient, $B$ the bulk modulus and $V_m$ the molar volume.
Silicon expands very little, so this term is under 1 % at 1000 K; for softer crystals such as Al or NaCl it reaches
about 10 % near 800 K. Stage 3 obtains $C_P$ with the quasi-harmonic approximation, i.e. phonons at several volumes.

Also, in the harmonic approximation $C_V$ can never exceed $3R$, while the measured $C_P$ of silicon clearly exceeds
$3R$ at high temperature. The excess comes mostly from anharmonicity (interactions between phonons). That is a limit
of the harmonic approximation, not an error of the ML potential.

## 8. Where each formula lives in the code

| Formula | Code |
|---|---|
| The four single-mode formulas of section 1 | `thermo.mode_functions` |
| The weighted q-point sum of section 2 and the unit conversion of section 3 | `thermo.harmonic_thermo` |
| The Dulong–Petit limit $3nR$ | `thermo.dulong_petit` |
| Zeroing the acoustic modes at $\Gamma$ (section 6) | `phonons.mesh_frequencies` |
| $Z$ of section 3 | `phonons.formula_units` |

## 9. Exercises

1. Without looking at these notes, derive $q$, $F$, $U$, $S$ and $C_V$ from the energy levels.
2. Check that $S = -\partial F / \partial T$ and $C_V = T\, \partial S / \partial T$.
3. By hand: for a mode with $\nu = 10$ THz at 300 K, what are $x$ and $C_V / k_B$ roughly?
   (Answer: $x \approx 1.60$, $C_V / k_B \approx 0.81$.)
4. Explain why the silicon results must be divided by 2, while those of NaCl (a primitive cell of two atoms but one
   formula unit) must not.
5. Explain why the too-low phonon frequencies of MACE-MP-0 make both the low-temperature heat capacity and the
   entropy too large.
