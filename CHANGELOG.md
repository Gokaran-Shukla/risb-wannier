# Changelog

## 1.2.0 (2026-11-11), first public release

- Multi-core parallelism: Jacobian columns, inequivalent embedding problems, threaded k-point chunks (`n_procs` / `--nproc`); serial and parallel agree to ~1e-10

## 1.1.0

- Approximate charge self-consistency: linearised Hartree(+xc) feedback on correlated and ligand shells (`charge_feedback`), self-consistent FLL double counting, combined Anderson mixer
- Symmetry-detecting `auto` R/Lambda structure (keeps symmetry-allowed off-diagonal elements, e.g. dz2/dx2-y2 in D2h)
- Finite-difference Jacobian with absolute step floor; interaction-ramp continuation with bisection; Jacobian-reusing quasi-Newton outer solver

## 1.0.0

- RISB embedding solver (Lanata et al. PRX 2015) on wannier90_hr.dat: Kanamori, Slater and file interactions; FLL / fixed double counting; symmetry-adapted parameter bases; class equivalence; paramagnetic and spinful modes; quasiparticle bands; restart files; CLI
