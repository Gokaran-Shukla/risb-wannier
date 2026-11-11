# risb-wannier

**Rotationally invariant slave-boson (RISB) / multi-orbital Gutzwiller solver on Wannier90 Hamiltonians,
with approximate charge self-consistency and multi-core parallelism. Pure Python (NumPy + SciPy).**

[![tests](https://github.com/Gokaran-Shukla/risb-wannier/actions/workflows/tests.yml/badge.svg)](https://github.com/Gokaran-Shukla/risb-wannier/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

`risb-wannier` solves the RISB saddle-point equations (embedding formulation of Lanatà *et al.*,
PRX **5**, 011008 (2015)) for multi-orbital Hubbard models whose one-body part is read directly from
`wannier90_hr.dat`. It runs one-shot DFT+RISB, or DFT+RISB with a linearised Hartree+xc charge
self-consistency loop inside the Wannier space. It is meant for VASP/QE → Wannier90 users who cannot
use ComRISB's Wien2k/FlapwMBPT-bound workflow.

## Capabilities

| | |
|---|---|
| **Input** | any `wannier90_hr.dat` (VASP, QE, …); paramagnetic (spinless) or spinful (SOC/magnetic, `block` or `interleaved` spin order) |
| **Correlated shells** | any number of shells, any orbitals; equivalent sites share R and Λ through classes (ComRISB's `impurity_problem_equivalence`) |
| **Interactions** | rotationally invariant Kanamori (U, J, U′); Slater–Condon for full p/d shells (F0, F2, F4 or U, J, F4/F2), with orbital-order permutation; arbitrary U_ijkl tensor from a `.npy` file |
| **Double counting** | none, fixed, or FLL with the DFT occupancy, a given occupancy, or self-consistent |
| **Parameter structure** | symmetry-adapted R/Λ bases: `auto` detection from the data (keeps symmetry-allowed off-diagonal elements), `diagonal`, `full`, or explicit label matrices; real or complex; crystal-field or Wannier basis |
| **Solver** | exact-diagonalisation embedding (sparse Lanczos) in the N = n (and Sz = 0) sector; `scipy.optimize.root` (hybr, lm, broyden, krylov, …) with a custom finite-difference Jacobian; interaction-ramp continuation from the exact U = 0 root with bisection; Jacobian-reusing quasi-Newton |
| **Charge self-consistency** | linearised Hartree+xc feedback on correlated and ligand shells (`V_H`, `V_orb`), Anderson mixing combined with self-consistent DC, restartable |
| **Diagnostics** | full-matrix ansatz residual, Hellmann–Feynman-consistent free energy at finite smearing, lowest embedding levels, ⟨S²⟩, double occupancies |
| **Output** | μ, E_kin, E_loc, E_total, F_total; per class R, Λ, Z = R†R, physical and qp occupations, V_dc, ⟨H_int⟩; qp and bare bands with plot; restart file |
| **Parallelism** | Jacobian columns, inequivalent embedding problems and k-point chunks over cores (`n_procs`); serial and parallel agree to ~1e-10 |
| **Validation** | 33 benchmark checks against exact results (atomic multiplets, U = 0, analytic Gutzwiller, Brinkman–Rice, basis invariance, independent Hartree solver, Hellmann–Feynman), run in CI |

## Installation

```bash
git clone https://github.com/Gokaran-Shukla/risb-wannier.git
cd risb-wannier
pip install -e ".[plot]"          # numpy, scipy, matplotlib
python -m pytest tests -q         # 33 benchmark checks, about 1 minute
```

## Quick start

```bash
cp examples/CsV3Sb5_input.json input.json        # edit orbitals, fermi_energy, U, J
risb-wannier input.json                           # or: python risb_wannier.py input.json
risb-wannier input.json --nproc 40                # parallel
risb-wannier input.json --restart risb_out/risb_restart.npz
```

A minimal input:

```json
{
  "hr_file": "wannier90_hr.dat",
  "spin_mode": "paramagnetic",
  "kmesh": [18, 18, 10],
  "temperature": 0.005,
  "fermi_energy": 5.0,
  "impurities": [{"name": "V1", "orbitals": [0, 1, 2, 3, 4], "class": "V"}],
  "classes": {"V": {"interaction": {"type": "kanamori", "U": 4.0, "J": 0.7},
                    "dc": {"type": "fll", "n": "dft"}, "structure": "auto"}}
}
```

The full template for CsV₃Sb₅ (three V sites in two classes, Sb-p ligand feedback) is
[`examples/CsV3Sb5_input.json`](examples/CsV3Sb5_input.json); its `_note` keys explain every choice.

## Parallel execution

Set `"n_procs": 40` in the JSON, or pass `--nproc 40` (`auto` uses all cores). It uses
fork-based multiprocessing on Linux. There are three levels:

| level | what runs in parallel | when it is used |
|---|---|---|
| Jacobian columns | each finite-difference column is an independent residual evaluation (27 for two V classes with `auto`, up to 75 with `full`) | when residual time × columns > `parallel_min_seconds` (0.5 s) |
| Embedding problems | inequivalent classes (e.g. Va and Vb) are diagonalized concurrently in every residual | when the embedding sector dimension ≥ `parallel_min_embedding_dim` (2000; a d shell is 63,504) |
| k-points | the H_qp(k) diagonalization is split into threaded chunks | with ≥ 64 k-points per thread |

Notes:
- Workers run with BLAS limited to 1 thread. Also `export OMP_NUM_THREADS=1` (and `MKL_NUM_THREADS=1`) in your job script so the processes don't oversubscribe the cores.
- The finite-difference Jacobian is where most of the run time goes: during the interaction ramp, at the first outer step, and whenever the quasi-Newton solver needs a fresh Jacobian. With ≥ 27 cores it costs about one residual evaluation instead of 27.
- Memory: each worker holds its own embedding Hamiltonian and Lanczos workspace, about 0.1–0.2 GB for a d shell. Budget about 8 GB for 40 workers.
- The sparse Lanczos solve of each single embedding problem itself stays on one core.
- Parallel and serial runs agree to numerical precision; only the warm-start vectors of the iterative eigensolver differ.
- Measured on 2 cores (two-class d-shell model, feedback + self-consistent DC): 1.7× per residual, 1.9× per Jacobian, 1.8× for the full solve (1000 s → 558 s).

## Equations solved

Lanatà et al., PRX 5, 011008 (2015) embedding formulation. At the saddle point it
is equivalent to Lechermann et al., PRB 76, 155102 (2007).

| quantity | definition |
|---|---|
| qp Hamiltonian | H_qp(k) = 𝓡 ε(k) 𝓡† + Λ, with ε(k) = H₀(k) − E_loc and 𝓡 block-diagonal (R on each correlated shell, 1 elsewhere) |
| qp density | Δ_ab = ⟨f_a† f_b⟩ = [(1/N_k) Σ_k ρ_k]ᵀ, with ρ_k = U_k f(e_k) U_k† |
| hybridisation | D = A^{-1/2} K\*, where A = Δ(1−Δ) and K = (1/N_k) Σ_k [ρ_k 𝓡 ε_k] (shell block) |
| bath levels | λᶜ = −Λ − Qᵀ, where Q = ∂Tr[(M+M†)A^{1/2}]/∂Δᵀ and M = D Rᵀ (Daleckii–Krein derivative) |
| embedding | H_emb = H_loc[c] + Σ (D_{aα} c_α† b_a + h.c.) + Σ λᶜ_{ab} b_b b_a†, with its ground state taken in the N = n sector (Sz = 0 is also imposed when paramagnetic) |
| root conditions | F₁ = ⟨b_b b_a†⟩ − Δ_ab = 0, F₂ = ⟨c_α† b_a⟩ − [Rᵀ A^{1/2}]_{αa} = 0 |

The unknowns (R, Λ) are expanded in a symmetry-adapted matrix basis, built
ComRISB-style from integer structure matrices. They are then solved with
`scipy.optimize.root` (`hybr` by default). The chemical potential is fixed at
every step by the electron count.

Imposing the half-filled embedding sector is what removes the spurious
"empty impurity" R ≈ 0 root at U = 0.

After convergence the code also evaluates the **full-matrix** residuals F₁ and F₂,
not just their projection onto the parameter basis. A non-zero value means the
R/Λ structure is more restrictive than the symmetry. The solution is then not a
stationary point (Hellmann–Feynman fails and the energy is too high), and the code
prints a warning. The result is reported as `ansatz_residual`.

## Approximate charge self-consistency (`charge_feedback`)

Real CSC DFT+RISB feeds the correlated density back into the DFT potential.
Inside the fixed Wannier space, that response is linearized around the DFT
density n⁰ (the U=0 local density of your hr.dat) by adding a local Hartree+xc
functional to the RISB functional:

E_H = Σ_s [ ½ V_H,s (δN_s)² + ½ V_orb,s Tr(δñ_s δñ_s) ],  with δn_s = n_s − n⁰_s, δN_s = Tr δn_s and δñ_s its traceless part.

Making this stationary gives a one-body correction that is iterated to self-consistency:

H₀(k) → H₀(k) + Σ_s δV_s,  δV_s = V_H,s δN_s 𝟙 + V_orb,s δñ_sᵀ

- **Correlated shells** use the physical embedding density. Set this with `classes.X.feedback = {V_H, V_orb}`.
- **Uncorrelated ligand shells** (e.g. Sb-p) use the lattice density. Set these with `charge_feedback.ligands = [{name, orbitals, V_H, V_orb}]`.
- **Outer loop:** a single Anderson (Pulay) mixer acts on the combined vector [δn ; N_dc], the feedback densities together with the self-consistent FLL double-counting occupations. Updating N_dc separately corrupts the Anderson model (in testing it caused runaway charge transfer). The step is capped and the history is reset if the residual grows.
- **Reported energy:** `E_total` is the full functional E_RISB[H₀+δV] − Tr(δV n) + E_H. `F_total` = E_total − T S_qp is exactly stationary at finite smearing (tested to 10⁻⁸ by Hellmann–Feynman).
- **Symmetry:** for correlated shells, δn is restricted to the symmetry space of the R/Λ ansatz.

**Choosing V_H.** V_H is the screened Hartree+xc response of the shell's KS
level to its own occupation, ∂ε_s/∂N_s. This is exactly the quantity computed by
the Cococcioni–de Gironcoli linear-response method (PRB 71, 035105, 2005):
U_LR = χ₀⁻¹ − χ⁻¹, from a VASP/QE calculation that perturbs the V-d (or Sb-p)
potential. That is the most defensible input. Otherwise, treat V_H as a
parameter and report how sensitive your results are to it.

**Interplay with double counting.** With self-consistent FLL DC, the net shift
of the d level is (V_H − U + J/2) δN. So V_H = U − J/2 exactly cancels the DC
update, and the result equals one-shot with DC fixed at the DFT occupancy. Large
V_H pins the shell charge at its DFT value.

| `charge_feedback` key | meaning |
|---|---|
| `ligands` | list of uncorrelated shells receiving feedback |
| `mixing` | Anderson mixing parameter (default 0.5) |
| `history` | Anderson history length (default 5; 0 = linear mixing) |
| `tol` | convergence threshold on max\|n_out − n_in\| (default 1e−6) |
| `max_iter` | maximum outer iterations (default 60) |
| `max_step` | cap per outer step on any δn element (default 0.1; ×5 for N_dc) |
| `outer_method` | inner solver used during outer iterations: `qn` (default, Jacobian-reusing quasi-Newton, falls back to hybr) or any `solver.method` |
| `predict_lambda` | shift Λ by the change of the local level (E_loc + δV − V_dc) before each inner solve (default true) |
| `inexact_inner` | solve the inner RISB problem to 1% of the current outer residual, full `tol` for the final solve (default true) |

The restart file stores the converged feedback potential and the DC occupations,
so a restarted run converges in one outer step.

## Input reference (JSON)

| key | meaning |
|---|---|
| `hr_file` | Wannier90 `seedname_hr.dat` |
| `spin_mode` | `paramagnetic`: spinless hr.dat, spin-degenerate (lattice in orbital space, embedding in the Sz=0 sector). `spinful`: SOC or magnetic hr.dat, which requires `spin_ordering` = `block` (all up, then all down) or `interleaved` (up, down per orbital). Check which convention your code wrote. |
| `kmesh`, `kshift` | Γ-centred mesh |
| `temperature` | Fermi smearing (eV). At finite T the solution extremises the qp *free energy*. |
| `n_electrons` or `fermi_energy` | electrons in the Wannier manifold per cell; `fermi_energy` is converted using the U=0 bands |
| `impurities[]` | `orbitals` holds 0-based spatial WF indices of each correlated shell; `class` holds the equivalence label. Members of a class share R and Λ (this is ComRISB's `impurity_problem_equivalence`). |
| `classes.X.interaction` | `kanamori` {U, J, optional Uprime}; `slater` {F:[F0,F2,F4]} or {U, J, F4_over_F2}, for full p or d shells in Wannier90 order (dz², dxz, dyz, dx²−y², dxy), with optional `orbital_order` permutation; `file` (.npy orbital or spin-orbital U_ijkl, H = ½ Σ U_ijkl c†_i c†_j c_l c_k); `none` |
| `classes.X.dc` | `none`; `fixed` {value}; `fll` {U, J (default: interaction values), n: `dft` (U=0 occupancy), a number, or `self_consistent` (outer loop)} |
| `classes.X.basis` | `cf`: rotate the shell to the eigenbasis of its local Hamiltonian (default). `wannier`: keep the WF basis. |
| `classes.X.structure` | `auto` (recommended): the symmetry-allowed pattern is detected from the data. An element (a,b) is kept if it is non-zero in E_loc, the U=0 density matrix, or the hybridization second moment ⟨ε²⟩_loc; diagonal entries are tied only if all three agree. Tolerances are `auto_tol` (2e−4 eV) and `auto_tol_rel` (1e−3). Other options: `diagonal`; `full` (R general, Λ Hermitian); or an explicit integer label matrix. Use `complex: true` for complex parameters (switched on automatically if `auto` finds complex off-diagonal terms). |
| `classes.X.feedback` | `{V_H, V_orb}` for the approximate charge self-consistency (see above) |
| `classes.X.sz_sector` | restrict the embedding to Sz = 0 (paramagnetic only; default true) |
| `solver` | `method` (`hybr`, `lm`, `broyden1`, `krylov`, …, or `linear`), `tol`, `maxfev`, `accept`, `dc_tol`. Continuation: `interaction_ramp` (number of steps, or a list of scale factors), `ramp_first` (small first step from U=0, default 0.02), `ramp_accept` (acceptance at intermediate scales, default 1e−3), `max_bisect` (step halvings, default 6). Jacobian: `fd_step` (default 1e−6, absolute floor), `fd_central` (default false) |
| `kpath` | `points` (fractional), `labels`, `npts`, `reciprocal_lattice` (rows b_i, for correct distances), `ylim` |

## Outputs

These are written to `output_dir`:

- `results.json`: μ, E_kin, E_local, E_total, F_total, `ansatz_residual`, and with feedback a `charge_feedback` block (N_DFT → N per shell, δV, E_H). For each class it also holds R, Λ, Z = R†R, physical and qp occupations, V_dc, ⟨H_int⟩, the three lowest embedding levels (to check for degeneracy) and, in paramagnetic mode, ⟨S²⟩ and per-orbital double occupancy.
- `risb_restart.npz`
- `qp_bands.dat`, `bare_bands.dat` and `bands.png`

## Validation (`tests/test_risb_wannier.py`)

| test | result |
|---|---|
| Kanamori 2-orbital/2e multiplets (U−3J, U−J, U+J) | exact |
| Kanamori O(3) invariance, Slater d² Racah energies, Slater SO(3) invariance | ≤ 1e−10 |
| U = 0 with complex off-diagonal on-site and inter-orbital hopping → R = 1, Λ = E_loc, E = band energy | ≤ 1e−8 |
| Single-band Gutzwiller vs analytic GA (half filling and n = 0.8, four U values): Z, d, E | all printed digits (≤ 1e−7) |
| Brinkman–Rice Z = 1 − (U/U_c)² | 0.84000 vs 0.84000; 0.35993 vs 0.36000 (finite T) |
| Random complex-unitary rotation of the orbital basis, non-diagonal R, U(4)-invariant interaction | ΔE ~ 1e−14 |
| Shared class vs independent classes for equivalent sites | ΔE ~ 1e−15 |
| Spinful (interleaved) vs paramagnetic | ΔE ~ 1e−14 |
| `auto` detects a symmetry-allowed off-diagonal element and equals `full`; a diagonal ansatz is flagged (residual 1.4e−2, F 18 meV too high); `auto` stays diagonal when symmetry forbids mixing | ΔF ~ 1e−14 |
| Feedback with V_H = V_orb = 0 equals one-shot | exact |
| U = 0 feedback (d shell + p ligand) vs an independent Hartree solver | δN to 1e−8 |
| Hellmann–Feynman for the feedback functional: dF/dV_H = ½δN², dF/dV_orb = Tr δñ² | 4e−8 |
| Increasing V_H pins the d charge monotonically to its DFT value | ✓ |
| Feedback: spinful = paramagnetic, Σ δN = 0 | 1e−14 |
| CLI with feedback, ligand and self-consistent FLL; restart converges in 1 outer step | ✓ |
| Jacobian-reusing quasi-Newton solver (`qn`) vs hybr | ΔF ~ 1e−11 |
| Full suite with `n_procs = 2`; two-class d-shell model with feedback + self-consistent DC, serial vs 2 processes | all pass; ΔF = 4e−10, ΔZ = 3e−12, ΔN = 5e−11 |

Timing (2 cores), for d shells with an embedding sector dimension of 63,504. The
two embedding diagonalizations per residual call dominate (~1 s each), not the k-mesh.

- One-shot, 3 equivalent V sites: about 2 minutes.
- Three V sites in 2 classes plus a ligand shell, with V_H/V_orb feedback and self-consistent FLL DC: 11 outer iterations in about 16 minutes. Each outer step needs only 3–8 residual evaluations, because the quasi-Newton solver reuses its Jacobian.

## Things to know before production runs

1. **How the root solve is made robust.** The interaction (and DC) is switched
   on by continuation, U → sU. The ramp starts with a small step, uses a secant
   predictor, and halves the step if one fails. hybr gets its own
   finite-difference Jacobian with an absolute step floor. MINPACK's built-in step
   is purely relative, which gives noise columns for parameters that start at ~0
   and made hybr stall on d-shell problems. RISB has an exact gauge freedom
   (R → uR, Λ → uΛu†), so compare gauge-invariant quantities between runs
   (Z = R†R, R†ΛR, energies, occupations), not R or Λ themselves.
2. **Multiple saddle points exist.** The tests found a metastable root 0.13 eV
   above the true solution in a 2-orbital model. By default the solver ramps the
   interaction from the exact U = 0 root (`interaction_ramp`). For Mott or
   orbital-selective regimes, also scan U up *and* down using `--restart`, and keep
   the lowest-energy solution.
3. **Check `ansatz_residual`.** It should be ≲ 1e−8. Since v1.1, `auto` detects
   symmetry-allowed mixing. For V-d in a D2h (mmm) site, dz² and dx²−y² are both
   Ag and *can* mix. The old purely diagonal `auto` (v1.0) was therefore not exact
   for CsV₃Sb₅; v1.1 keeps that element automatically. If your Wannier functions
   are only approximately symmetric, loosen `auto_tol` rather than letting tiny
   symmetry-breaking noise add parameters.
4. **Double counting is your physics choice.** FLL with the U=0 occupancy is the
   default when `fll` is chosen.
5. **Charge self-consistency is approximate.** The feedback is linear around the
   DFT density and the Wannier orbitals stay fixed. For full CSC, write the
   correlated density back into VASP/QE and re-Wannierize (planned).
6. **No ghost orbitals** (g-RISB is not implemented).
7. **Memory:** the lattice stores a few N_k × n_WF² complex arrays.
8. **Spinful d shell:** the embedding sector is C(20,10) = 184,756, so expect a few times the paramagnetic cost.

## Citing

If you use risb-wannier, please cite this repository (see [`CITATION.cff`](CITATION.cff)) and the method papers:

- N. Lanatà, Y. Yao, C.-Z. Wang, K.-M. Ho, G. Kotliar, Phys. Rev. X **5**, 011008 (2015)
- F. Lechermann, A. Georges, G. Kotliar, O. Parcollet, Phys. Rev. B **76**, 155102 (2007)
- M. Cococcioni, S. de Gironcoli, Phys. Rev. B **71**, 035105 (2005), if you use linear-response V_H

## Licence

MIT, see [`LICENSE`](LICENSE). This is an independent implementation; it contains no ComRISB source code.
