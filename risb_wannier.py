#!/usr/bin/env python3
r"""
risb_wannier.py
===============

Rotationally-invariant slave-boson (RISB) / multi-orbital Gutzwiller solver
for interacting models built from a Wannier90 tight-binding Hamiltonian
(``seedname_hr.dat``).

Formalism
---------
Lanata, Yao, Wang, Ho, Kotliar, PRX 5, 011008 (2015) embedding formulation of
RISB (equivalent to Lechermann et al., PRB 76, 155102 (2007) at the saddle
point).  Conventions used throughout (all checked numerically by the test
suite):

  * physical operators  P c_alpha^dag P  ->  sum_a R_{a alpha} f_a^dag
  * quasiparticle Hamiltonian
        H_qp(k) = R eps(k) R^dag + Lambda           (block-diag R on correlated
                                                     orbitals, 1 elsewhere)
    with eps(k) = H0(k) - E_loc, E_loc the on-site block of each correlated
    shell (it is moved into the local/embedding Hamiltonian).
  * Delta_ab = <f_a^dag f_b>_qp   (per correlated shell, qp indices)
  * Embedding Hamiltonian on 2n modes (n physical c, n bath b):
        H_emb = H_loc[c] + sum_{a,alpha} (D_{a alpha} c_alpha^dag b_a + h.c.)
                         + sum_{ab} lambda^c_{ab} b_b b_a^dag
    ground state |Phi> taken in the half-filled sector N_c + N_b = n.
  * Saddle-point (stationarity) equations
        D        = A^{-1/2} K^*,   A = Delta(1-Delta),
                   K_{c alpha} = (1/Nk) sum_k [rho_k R eps_k]_{c alpha},
                   rho_k = U_k f(e_k) U_k^dag
        lambda^c = -Lambda - Q^T,  Q = d Tr[(M+M^dag) A^{1/2}] / d Delta^T,
                   M = D R^T     (Daleckii-Krein derivative of the sqrt)
  * Root conditions solved for (R, Lambda):
        F1_ab      = <Phi| b_b b_a^dag |Phi> - Delta_ab              = 0
        F2_{alpha a} = <Phi| c_alpha^dag b_a |Phi> - [R^T A^{1/2}]_{alpha a} = 0
    projected onto a symmetry-adapted matrix basis (ComRISB-style structure
    matrices), solved with scipy.optimize.root (default 'hybr').

Approximate charge self-consistency (optional, "charge_feedback")
------------------------------------------------------------------
The Wannier Hamiltonian is consistent with the DFT local densities n^0.  When
correlations redistribute charge, the DFT Hartree+xc potential would respond.
Within the Wannier subspace this response is linearised by the local
functional
    E_H[n] = sum_s [ 1/2 V_H,s (dN_s)^2 + 1/2 V_orb,s Tr(dn~_s dn~_s) ],
    dn_s = n_s - n_s^0  (local density matrix of shell s),  dN_s = Tr dn_s,
    dn~_s = traceless part,
added to the RISB functional.  Stationarity gives the one-body correction
    H0(k) -> H0(k) + sum_s dV_s,   dV_s = V_H,s dN_s 1 + V_orb,s dn~_s^T,
on correlated shells (n_s = physical embedding density) and, optionally, on
uncorrelated "ligand" shells (n_s = lattice density).  An outer loop
(Anderson mixing on dn) iterates this to self-consistency, together with a
self-consistent FLL double counting if requested.  The reported E_total is
the value of the full functional, E_RISB[H0+dV] - Tr(dV n) + E_H, which is
stationary (dE/dV_H = 1/2 dN^2: tested).

Usage
-----
    python risb_wannier.py input.json

See examples/CsV3Sb5_input.json and README.md for the input format.
"""
from __future__ import annotations

import argparse
import atexit
import itertools
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import sys
import time

import numpy as np
from scipy import optimize, sparse
from scipy.sparse.linalg import eigsh

__version__ = "1.2.0"

# =============================================================================
# 1. Wannier90 I/O and lattice
# =============================================================================


def read_hr(fname):
    """Read a Wannier90 seedname_hr.dat file.

    Returns
    -------
    rvec : (nrpts, 3) int     lattice vectors R (in units of a1,a2,a3)
    deg  : (nrpts,) int       Wigner-Seitz degeneracies
    hr   : (nrpts, nw, nw)    complex H_mn(R) in eV
    """
    with open(fname) as f:
        f.readline()  # header / date line
        nw = int(f.readline().split()[0])
        nr = int(f.readline().split()[0])
        deg = []
        while len(deg) < nr:
            deg += [int(x) for x in f.readline().split()]
        raw = np.array(f.read().split(), dtype=float)
    if raw.size != nr * nw * nw * 7:
        raise ValueError(
            f"{fname}: expected {nr*nw*nw} hopping lines, found {raw.size/7:.1f}")
    raw = raw.reshape(nr, nw * nw, 7)
    rvec = raw[:, 0, :3].astype(int)
    if not np.all(raw[:, :, :3].astype(int) == rvec[:, None, :]):
        raise ValueError(f"{fname}: R vector changes inside a block")
    m = raw[:, :, 3].astype(int) - 1
    n = raw[:, :, 4].astype(int) - 1
    ir = np.repeat(np.arange(nr), nw * nw).reshape(nr, nw * nw)
    hr = np.zeros((nr, nw, nw), complex)
    hr[ir, m, n] = raw[:, :, 5] + 1j * raw[:, :, 6]
    return rvec, np.asarray(deg[:nr], int), hr


def write_hr(fname, rvec, deg, hr, header="written by risb_wannier"):
    """Write a Wannier90-format hr.dat (used by the tests and for model input)."""
    nr, nw, _ = hr.shape
    with open(fname, "w") as f:
        f.write(header + "\n")
        f.write(f"{nw:12d}\n{nr:12d}\n")
        for i in range(0, nr, 15):
            f.write("".join(f"{d:5d}" for d in deg[i:i + 15]) + "\n")
        for ir in range(nr):
            for n in range(nw):
                for m in range(nw):
                    h = hr[ir, m, n]
                    f.write(f"{rvec[ir,0]:5d}{rvec[ir,1]:5d}{rvec[ir,2]:5d}"
                            f"{m+1:5d}{n+1:5d}{h.real:20.12f}{h.imag:20.12f}\n")


def monkhorst_pack(mesh, shift=(0, 0, 0)):
    """Uniform Gamma-centred (shift=0) fractional k mesh."""
    axes = [(np.arange(n) + s) / n for n, s in zip(mesh, shift)]
    k = np.array(list(itertools.product(*axes)))
    return k


def hk_from_hr(kpts, rvec, deg, hr):
    """H(k) = sum_R exp(2 pi i k.R) H(R) / deg(R)  (Wannier90 convention)."""
    phase = np.exp(2j * np.pi * kpts @ rvec.T) / deg[None, :]
    nr, nw, _ = hr.shape
    return (phase @ hr.reshape(nr, nw * nw)).reshape(len(kpts), nw, nw)


def fermi(x, T):
    return 0.5 * (1.0 - np.tanh(0.5 * x / T))


# =============================================================================
# 2. Interaction tensors   H_int = 1/2 sum U_ijkl c_i^+ c_j^+ c_l c_k
# =============================================================================


def _wigner3j(j1, j2, j3, m1, m2, m3):
    if m1 + m2 + m3 != 0 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    f = math.factorial
    pre = (-1) ** (j1 - j2 - m3) * math.sqrt(
        f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1)
        * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3))
    s = 0.0
    for t in range(0, j1 + j2 + j3 + 1):
        dens = [t, j3 - j2 + t + m1, j3 - j1 + t - m2,
                j1 + j2 - j3 - t, j1 - t - m1, j2 - t + m2]
        if min(dens) < 0:
            continue
        s += (-1) ** t / np.prod([f(d) for d in dens])
    return pre * s


def _gaunt(l1, m1, l2, m2, l3, m3):
    """Integral of Y_l1m1 Y_l2m2 Y_l3m3 over the sphere."""
    return (math.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * math.pi))
            * _wigner3j(l1, l2, l3, 0, 0, 0) * _wigner3j(l1, l2, l3, m1, m2, m3))


def real_harmonics_wannier90(l):
    """Columns = Wannier90 real orbitals expanded in complex Y_lm (m=-l..l).

    l=1 order: pz, px, py
    l=2 order: dz2, dxz, dyz, dx2-y2, dxy    (Wannier90 default)
    Condon-Shortley phase; all real orbitals are the positive cubic harmonics.
    """
    s = 1 / math.sqrt(2)
    T = np.zeros((2 * l + 1, 2 * l + 1), complex)
    ix = lambda m: m + l
    if l == 1:
        T[ix(0), 0] = 1
        T[ix(-1), 1], T[ix(1), 1] = s, -s
        T[ix(-1), 2], T[ix(1), 2] = 1j * s, 1j * s
    elif l == 2:
        T[ix(0), 0] = 1
        T[ix(-1), 1], T[ix(1), 1] = s, -s
        T[ix(-1), 2], T[ix(1), 2] = 1j * s, 1j * s
        T[ix(-2), 3], T[ix(2), 3] = s, s
        T[ix(-2), 4], T[ix(2), 4] = 1j * s, -1j * s
    else:
        raise NotImplementedError("Slater tensor implemented for l=1,2")
    return T


def rotate_tensor(U, T):
    """Basis change c_i = sum_a T_ia c'_a:  U'_abcd = T*_ia T*_jb T_kc T_ld U_ijkl."""
    Tc = T.conj()
    return np.einsum("ia,jb,kc,ld,ijkl->abcd", Tc, Tc, T, T, U, optimize=True)


def slater_orbital_tensor(l, F):
    """Orbital U_{m1 m2 m3 m4} for Slater integrals F = [F0, F2, F4, ...],
    returned in the Wannier90 real-orbital basis."""
    nm = 2 * l + 1
    U = np.zeros((nm,) * 4, complex)
    ms = range(-l, l + 1)
    for ik, k in enumerate(range(0, 2 * l + 1, 2)):
        if ik >= len(F) or F[ik] == 0:
            continue
        for m1, m2, m3, m4 in itertools.product(ms, repeat=4):
            a = 0.0
            for q in range(-k, k + 1):
                g1 = (-1) ** m1 * _gaunt(l, -m1, k, q, l, m3)
                if g1 == 0:
                    continue
                g2 = (-1) ** (m2 + q) * _gaunt(l, -m2, k, -q, l, m4)
                a += g1 * g2
            U[m1 + l, m2 + l, m3 + l, m4 + l] += F[ik] * 4 * math.pi / (2 * k + 1) * a
    U = rotate_tensor(U, real_harmonics_wannier90(l))
    if np.abs(U.imag).max() > 1e-10:
        raise RuntimeError("Slater tensor not real in real-harmonic basis")
    return U.real


def kanamori_orbital_tensor(norb, U, J, Up=None):
    """Orbital Kanamori tensor (U' = U - 2J unless given):
       U_mmmm = U, U_mm'mm' = U', U_mm'm'm = J (exchange), U_mmm'm' = J (pair hop)."""
    Up = U - 2 * J if Up is None else Up
    T = np.zeros((norb,) * 4)
    for m in range(norb):
        T[m, m, m, m] = U
        for mp in range(norb):
            if mp != m:
                T[m, mp, m, mp] = Up
                T[m, mp, mp, m] = J
                T[m, m, mp, mp] = J
    return T


def spin_orbital_tensor(Uorb):
    """Spin-orbital tensor (sigma-major index i = sigma*norb + m), spin-independent."""
    norb = Uorb.shape[0]
    n = 2 * norb
    U = np.zeros((n,) * 4, complex)
    for s1, s2 in itertools.product(range(2), repeat=2):
        o1, o2 = s1 * norb, s2 * norb
        U[o1:o1 + norb, o2:o2 + norb, o1:o1 + norb, o2:o2 + norb] = Uorb
    return U


def build_interaction(spec, norb):
    """Spin-orbital U tensor (Wannier orbital basis, sigma-major) from a spec dict."""
    typ = spec.get("type", "kanamori").lower()
    if typ == "kanamori":
        Uorb = kanamori_orbital_tensor(norb, spec["U"], spec["J"], spec.get("Uprime"))
    elif typ == "slater":
        l = (norb - 1) // 2
        if 2 * l + 1 != norb:
            raise ValueError("Slater interaction needs a full l-shell (3 or 5 orbitals)")
        if "F" in spec:
            F = list(spec["F"])
        else:  # from (U, J) with standard ratio F4/F2
            U, J = spec["U"], spec["J"]
            if l == 2:
                r = spec.get("F4_over_F2", 0.625)
                F2 = 14 * J / (1 + r)
                F = [U, F2, r * F2]
            else:
                F = [U, 5 * J]
        Uorb = slater_orbital_tensor(l, F)
        perm = spec.get("orbital_order")  # optional permutation of W90 order
        if perm is not None:
            Uorb = Uorb[np.ix_(perm, perm, perm, perm)]
    elif typ == "file":
        Uorb = np.load(spec["file"])
        if Uorb.shape == (2 * norb,) * 4:
            return Uorb.astype(complex)
    elif typ == "none":
        Uorb = np.zeros((norb,) * 4)
    else:
        raise ValueError(f"unknown interaction type {typ}")
    return spin_orbital_tensor(Uorb)


def dc_average_UJ(spec):
    """(U_avg, J_avg) for FLL double counting, unless overridden in dc spec."""
    typ = spec.get("type", "kanamori").lower()
    if typ == "slater" and "F" in spec:
        F = spec["F"]
        return F[0], (F[1] + (F[2] if len(F) > 2 else 0)) / 14.0
    return spec.get("U", 0.0), spec.get("J", 0.0)


# =============================================================================
# 3. Fock space of the embedding problem (2n modes, particle-number sector)
# =============================================================================


def _apply_string(states, ops):
    """Apply an operator string (rightmost first) to an array of bit states.

    ops : sequence of (dagger: bool, mode: int) in *application order*.
    Returns (valid mask, new states, sign)."""
    s = states.copy()
    sign = np.ones(len(s), np.int8)
    valid = np.ones(len(s), bool)
    for dag, p in ops:
        bit = np.int64(1) << np.int64(p)
        occ = (s & bit) != 0
        valid &= ~occ if dag else occ
        par = np.bitwise_count(s & (bit - 1)) & 1
        sign = sign * (1 - 2 * par.astype(np.int8))
        s = s ^ bit
    return valid, s, sign


class FockSector:
    """Basis of 2n-mode fermion states with fixed N (and optionally Sz=0)."""

    def __init__(self, nmodes, nparticles, spins=None):
        self.nmodes = nmodes
        combos = itertools.combinations(range(nmodes), nparticles)
        if spins is None:
            st = [sum(1 << p for p in c) for c in combos]
        else:
            spins = np.asarray(spins)
            st = [sum(1 << p for p in c) for c in combos if spins[list(c)].sum() == 0]
        self.states = np.array(sorted(st), dtype=np.int64)
        self.dim = len(self.states)

    def action(self, ops):
        """(rows, cols, sign) of an operator string restricted to the sector."""
        valid, s, sign = _apply_string(self.states, ops)
        cols = np.nonzero(valid)[0]
        s = s[cols]
        rows = np.searchsorted(self.states, s)
        rows = np.minimum(rows, self.dim - 1)
        ok = self.states[rows] == s
        return rows[ok], cols[ok], sign[cols][ok]


class Embedding:
    """Embedding impurity problem for one correlated shell of n spin-orbitals."""

    def __init__(self, Uso, spins=None, verbose=False):
        n = Uso.shape[0]
        self.n = n
        t0 = time.time()
        self.fs = FockSector(2 * n, n, None if spins is None else
                             np.concatenate([spins, spins]))
        dim = self.fs.dim
        # cached one-body actions d_p^+ d_q
        self.ob = {}
        for p in range(2 * n):
            for q in range(2 * n):
                self.ob[p, q] = self.fs.action([(False, q), (True, p)])
        self.Hint = self._two_body(Uso)
        self.phi = None
        self.spins = spins
        self.scale = 1.0          # interaction scale (continuation in U)
        if verbose:
            print(f"    embedding: n={n}, sector dim={dim}, "
                  f"H_int nnz={self.Hint.nnz}, built in {time.time()-t0:.1f}s")

    def _two_body(self, U, tol=1e-12):
        n, dim = self.n, self.fs.dim
        # antisymmetrised W_{ij,kl} for i<j, k<l : c_i^+ c_j^+ c_l c_k
        W = 0.5 * (U - U.transpose(1, 0, 2, 3) - U.transpose(0, 1, 3, 2)
                   + U.transpose(1, 0, 3, 2))
        rows, cols, vals = [], [], []
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for k, l in pairs:
            blk = W[:, :, k, l]
            if np.abs(blk).max() < tol:
                continue
            valid, s1, sg1 = _apply_string(self.fs.states, [(False, k), (False, l)])
            src = np.nonzero(valid)[0]
            s1, sg1 = s1[src], sg1[src]
            for i, j in pairs:
                w = blk[i, j]
                if abs(w) < tol:
                    continue
                v2, s2, sg2 = _apply_string(s1, [(True, j), (True, i)])
                s2 = s2[v2]
                r = np.searchsorted(self.fs.states, s2)
                rows.append(r)
                cols.append(src[v2])
                vals.append(w * (sg1[v2] * sg2[v2]))
        if not rows:
            return sparse.csr_matrix((dim, dim), dtype=complex)
        H = sparse.coo_matrix((np.concatenate(vals),
                               (np.concatenate(rows), np.concatenate(cols))),
                              shape=(dim, dim)).tocsr()
        return H

    def one_body(self, t, tol=1e-14):
        dim = self.fs.dim
        rows, cols, vals = [], [], []
        for (p, q), (r, c, sg) in self.ob.items():
            if abs(t[p, q]) > tol:
                rows.append(r)
                cols.append(c)
                vals.append(t[p, q] * sg)
        if not rows:
            return sparse.csr_matrix((dim, dim), dtype=complex)
        return sparse.coo_matrix((np.concatenate(vals),
                                  (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(dim, dim)).tocsr()

    def operator(self, terms):
        """Sparse matrix of sum_c coeff * (operator string)."""
        dim = self.fs.dim
        rows, cols, vals = [], [], []
        for coeff, ops in terms:
            r, c, sg = self.fs.action(ops)
            rows.append(r); cols.append(c); vals.append(coeff * sg)
        return sparse.coo_matrix((np.concatenate(vals),
                                  (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(dim, dim)).tocsr()

    def hamiltonian(self, h_phys, D, lam_c):
        n = self.n
        t = np.zeros((2 * n, 2 * n), complex)
        t[:n, :n] = h_phys                 # c^+ E_loc c
        t[:n, n:] = D.T                    # D_{a alpha} c_alpha^+ b_a
        t[n:, :n] = D.conj()               # h.c.
        t[n:, n:] = -lam_c                 # lambda^c_ab b_b b_a^+ = Tr - b_a^+ lam b_b
        H = self.scale * self.Hint + self.one_body(t)
        return H, np.trace(lam_c).real

    def ground_state(self, H, nstates=1):
        dim = H.shape[0]
        if dim <= 1500:
            w, v = np.linalg.eigh(H.toarray())
            return w[:nstates], v[:, :nstates]
        v0 = self.phi if (self.phi is not None and len(self.phi) == dim) else None
        w, v = eigsh(H, k=max(nstates, 1), which="SA", v0=v0, tol=1e-12,
                     ncv=max(20, 2 * nstates + 10))
        o = np.argsort(w)
        return w[o], v[:, o]

    def solve(self, h_phys, D, lam_c):
        H, const = self.hamiltonian(h_phys, D, lam_c)
        w, v = self.ground_state(H)
        self.phi = v[:, 0]
        return w[0] + const, self.density_matrix(self.phi)

    def density_matrix(self, phi):
        """Gamma_pq = <phi| d_p^+ d_q |phi> over all 2n modes."""
        n2 = 2 * self.n
        G = np.zeros((n2, n2), complex)
        pc = phi.conj()
        for (p, q), (r, c, sg) in self.ob.items():
            G[p, q] = np.dot(pc[r], sg * phi[c])
        return G

    def expect(self, M, phi=None):
        phi = self.phi if phi is None else phi
        return np.vdot(phi, M @ phi)


# =============================================================================
# 4. Matrix-function helpers
# =============================================================================


def _eig_clip(Delta, eps):
    d, V = np.linalg.eigh(0.5 * (Delta + Delta.conj().T))
    return np.clip(d, eps, 1 - eps), V


def sqrt_A(Delta, eps=1e-10):
    """A^{1/2}, A^{-1/2} for A = Delta(1-Delta)."""
    d, V = _eig_clip(Delta, eps)
    s = np.sqrt(d * (1 - d))
    return (V * s) @ V.conj().T, (V / s) @ V.conj().T


def dtrace_sqrtA(Delta, Hm, eps=1e-10):
    """Q such that  delta Tr[Hm A^{1/2}] = Tr[Q delta Delta]  (Hm Hermitian)."""
    d, V = _eig_clip(Delta, eps)
    s = np.sqrt(d * (1 - d))
    ds = (1 - 2 * d) / (2 * s)
    dd = d[:, None] - d[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(np.abs(dd) > 1e-9, (s[:, None] - s[None, :]) / dd,
                     0.5 * (ds[:, None] + ds[None, :]))
    Ht = V.conj().T @ Hm @ V
    return V @ (F * Ht) @ V.conj().T


# =============================================================================
# 5. Parameter bases (ComRISB-style structure matrices)
# =============================================================================


def structure_bases(S, complex_=False):
    """Orthonormal R basis and Hermitian Lambda basis from an integer label
    matrix S (0 = zero, equal positive labels tied)."""
    S = np.asarray(S, int)
    labels = sorted(set(S[S > 0].ravel()))
    Rb, Lb = [], []
    for k in labels:
        E = (S == k).astype(complex)
        Rb.append(E / np.linalg.norm(E))
        if complex_:
            Rb.append(1j * E / np.linalg.norm(E))
        Es = E + E.T
        Es[np.diag_indices_from(Es)] = np.diag(E)
        if np.abs(Es).sum() > 0:
            Lb.append(Es / np.linalg.norm(Es))
        if complex_:
            Up = np.triu(E, 1)
            if np.abs(Up).sum() > 0:
                A = 1j * (Up - Up.T)
                Lb.append(A / np.linalg.norm(A))
    # Gram-Schmidt in the real vector space of matrices (parameters are real)
    def orth(B):
        out = []
        for b in B:
            v = b.copy()
            for o in out:
                v = v - np.vdot(o, v).real * o
            nv = np.linalg.norm(v)
            if nv > 1e-10:
                out.append(v / nv)
        return out
    return orth(Rb), orth(Lb)


def auto_structure_sym(mats, tol_abs=2e-4, tol_rel=1e-3):
    """Symmetry-adapted structure from data, in the working (cf) basis.

    mats : list of local Hermitian matrices that share the site symmetry
           (E_loc, U=0 density matrix, hybridisation second moment, ...).
    A matrix element (a,b) is kept iff it is non-zero (above tolerance) in at
    least one of them: elements forbidden by symmetry vanish in all of them.
    Diagonal elements are tied iff all matrices agree on them (degenerate
    irrep partners), so accidental degeneracies of E_loc alone are not tied.

    Returns (S_Lambda, S_R, needs_complex):
      S_Lambda : symmetric label matrix for Lambda (and lambda^c, Delta)
      S_R      : label matrix for R (independent labels for (a,b) and (b,a))
    """
    n = mats[0].shape[0]
    scale = [max(np.abs(M).max(), 1e-12) for M in mats]
    allowed = np.zeros((n, n), bool)
    cplx = False
    for M, sc in zip(mats, scale):
        tol = max(tol_abs, tol_rel * sc)
        allowed |= np.abs(M) > tol
        off = M - np.diag(np.diag(M))
        cplx |= bool(np.any(np.abs(off.imag) > tol))
    np.fill_diagonal(allowed, True)
    # diagonal ties
    SL = np.zeros((n, n), int)
    lab = 0
    rep_ = []
    for i in range(n):
        for (j, lj) in rep_:
            same = all(abs(M[i, i] - M[j, j]) <= max(tol_abs, tol_rel * sc)
                       for M, sc in zip(mats, scale))
            # partners of one irrep must also have identical couplings pattern
            same &= np.array_equal(np.sort(allowed[i]), np.sort(allowed[j]))
            if same:
                SL[i, i] = lj
                break
        else:
            lab += 1
            rep_.append((i, lab))
            SL[i, i] = lab
    SR = SL.copy()
    labR = lab
    for a in range(n):
        for b in range(a + 1, n):
            if allowed[a, b]:
                lab += 1
                SL[a, b] = SL[b, a] = lab
                labR += 1
                SR[a, b] = labR
                labR += 1
                SR[b, a] = labR
    return SL, SR, cplx


def auto_structure(Eloc, mode, tol=1e-4):
    """Diagonal labels in the working (crystal-field) basis, tying levels that
    are degenerate within tol (eV)."""
    e = np.real(np.diag(Eloc))
    n = len(e)
    S = np.zeros((n, n), int)
    lab = 0
    rep = []
    for i in range(n):
        for j, (ej, lj) in enumerate(rep):
            if abs(e[i] - ej) < tol:
                S[i, i] = lj
                break
        else:
            lab += 1
            rep.append((e[i], lab))
            S[i, i] = lab
    return S


# =============================================================================
# 5b. Parallel workers (fork-based; Linux/macOS)
# =============================================================================
#
# Three levels of parallelism, all controlled by "n_procs":
#   (1) columns of the finite-difference Jacobian -> one residual evaluation per
#       worker (fork per Jacobian, so every worker sees the current state);
#   (2) inequivalent embedding problems (classes) solved concurrently inside a
#       residual evaluation (persistent pool forked after set-up; it only uses
#       the static parts of the embedding objects, everything else is passed);
#   (3) k-point chunks of the lattice diagonalisation (threads; LAPACK releases
#       the GIL).
# Workers run with BLAS limited to one thread to avoid oversubscription.

_WORKER = None          # solver instance seen by forked workers
_IN_WORKER = False


def _limit_blas():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _jac_worker_init():
    global _IN_WORKER
    _IN_WORKER = True
    _limit_blas()
    if _WORKER is not None:
        _WORKER.v = False


def _jac_column(args):
    j, x, hj, F0, central = args
    s = _WORKER
    e = np.zeros_like(x)
    e[j] = hj
    if central:
        return j, (s.residual(x + e) - s.residual(x - e)) / (2 * hj)
    return j, (s.residual(x + e) - F0) / hj


def _emb_worker_init():
    global _IN_WORKER
    _IN_WORKER = True
    _limit_blas()


def _emb_solve(args):
    cn, hphys, D, lam_c, scale, phi0 = args
    emb = _WORKER.classes[cn]["emb"]
    emb.scale = scale
    emb.phi = phi0
    E0, G = emb.solve(hphys, D, lam_c)
    return cn, E0, G, emb.phi


def _fork_context():
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


# =============================================================================
# 6. The RISB solver
# =============================================================================


class RISB:
    """RISB/Gutzwiller solver on a Wannier90 lattice."""

    def __init__(self, cfg, hr=None, verbose=True):
        self.cfg = cfg
        self.v = verbose
        self.T = float(cfg.get("temperature", 0.005))
        mode = cfg.get("spin_mode", "paramagnetic").lower()
        if mode not in ("paramagnetic", "spinful"):
            raise ValueError("spin_mode must be 'paramagnetic' or 'spinful'")
        self.mode = mode
        self.para = mode == "paramagnetic"
        self.deg = 2.0 if self.para else 1.0

        # ---------------- lattice
        if hr is None:
            hr = read_hr(cfg["hr_file"])
        rvec, wsdeg, H_R = hr
        nw = H_R.shape[1]
        if not self.para:
            order = cfg.get("spin_ordering")
            if order not in ("block", "interleaved"):
                raise ValueError("spinful mode needs spin_ordering = 'block' "
                                 "(all up, then all down) or 'interleaved' (up,down per orbital)")
            if nw % 2:
                raise ValueError("spinful hr.dat must have an even number of WFs")
            no = nw // 2
            if order == "interleaved":
                perm = np.array([2 * m + s for s in range(2) for m in range(no)])
                H_R = H_R[:, perm][:, :, perm]
        self.rvec, self.wsdeg, self.H_R = rvec, wsdeg, H_R
        self.nl = nw
        self.nspatial = nw if self.para else nw // 2
        self.kpts = monkhorst_pack(cfg.get("kmesh", [8, 8, 8]), cfg.get("kshift", [0, 0, 0]))
        self.nk = len(self.kpts)
        H0 = hk_from_hr(self.kpts, rvec, wsdeg, H_R)
        herm = np.abs(H0 - H0.conj().transpose(0, 2, 1)).max()
        if herm > 1e-6:
            print(f"WARNING: H(k) non-Hermitian by {herm:.2e}; symmetrising")
        H0 = 0.5 * (H0 + H0.conj().transpose(0, 2, 1))

        # ---------------- impurities and classes
        self.imps = []
        for spec in cfg["impurities"]:
            orbs = list(spec["orbitals"])
            idx = orbs if self.para else orbs + [o + self.nspatial for o in orbs]
            self.imps.append(dict(name=spec.get("name", f"imp{len(self.imps)}"),
                                  cls=spec.get("class", spec.get("name")),
                                  norb=len(orbs), idx=np.array(idx)))
        allidx = np.concatenate([im["idx"] for im in self.imps])
        if len(set(allidx)) != len(allidx):
            raise ValueError("correlated shells overlap")

        # local levels in Wannier basis and working-basis rotations
        Hloc_w = H0.mean(axis=0)
        Tfull = np.eye(nw, dtype=complex)
        classes_cfg = cfg["classes"]
        for im in self.imps:
            ccfg = classes_cfg[im["cls"]]
            E = Hloc_w[np.ix_(im["idx"], im["idx"])]
            basis = ccfg.get("basis", "cf").lower()
            if basis == "cf":
                e, V = np.linalg.eigh(E)
                for j in range(V.shape[1]):     # fix phases: largest comp. real>0
                    a = np.argmax(np.abs(V[:, j]))
                    V[:, j] *= np.exp(-1j * np.angle(V[a, j]))
                im["T"] = V
            elif basis == "wannier":
                im["T"] = np.eye(len(im["idx"]), dtype=complex)
            else:
                raise ValueError("basis must be 'cf' or 'wannier'")
            Tfull[np.ix_(im["idx"], im["idx"])] = im["T"]
        self.Tfull = Tfull
        H0w = Tfull.conj().T[None] @ H0 @ Tfull[None]
        self.Eloc_full = np.zeros((nw, nw), complex)
        for im in self.imps:
            sl = np.ix_(im["idx"], im["idx"])
            im["Eloc"] = H0w.mean(axis=0)[sl]
            self.Eloc_full[sl] = im["Eloc"]
        self.eps = H0w - self.Eloc_full[None]
        self.H0w = H0w

        # electron count
        e0 = np.linalg.eigvalsh(H0)
        if "n_electrons" in cfg:
            self.N = float(cfg["n_electrons"])
        elif "fermi_energy" in cfg:
            self.N = self.deg * fermi(e0 - cfg["fermi_energy"], self.T).sum() / self.nk
        else:
            raise ValueError("give n_electrons or fermi_energy")
        self.mu0 = self._find_mu(e0, self.N)

        # non-interacting local occupation (for DC 'dft' option)
        e, U = np.linalg.eigh(H0w)
        P0 = np.einsum("kan,kn,kbn->ab", U, fermi(e - self.mu0, self.T), U.conj()) / self.nk
        self.n0_full = P0.T            # n0_ab = <c_a^+ c_b> at U=0 (working basis, per spin if PM)

        # classes
        self.classes = {}
        for cname, ccfg in classes_cfg.items():
            members = [i for i, im in enumerate(self.imps) if im["cls"] == cname]
            if not members:
                continue
            ref = self.imps[members[0]]
            norb = ref["norb"]
            for i in members[1:]:
                if self.imps[i]["norb"] != norb:
                    raise ValueError(f"class {cname}: members have different shells")
                de = np.abs(np.sort(np.diag(self.imps[i]["Eloc"]).real)
                            - np.sort(np.diag(ref["Eloc"]).real)).max()
                if de > 1e-3:
                    print(f"WARNING: class {cname}: local levels of "
                          f"{self.imps[i]['name']} differ from {ref['name']} by {de:.2e} eV")
            nso = 2 * norb
            nlat = norb if self.para else nso
            # interaction in working basis
            Uw = build_interaction(ccfg["interaction"], norb)
            Tso = np.kron(np.eye(2), ref["T"]) if self.para else ref["T"]
            Uw = rotate_tensor(Uw, Tso)
            # double counting
            dc = ccfg.get("dc", {"type": "none"})
            n0 = np.trace(P0[np.ix_(ref["idx"], ref["idx"])]).real * self.deg
            # structure
            Eloc_lat = np.mean([self.imps[i]["Eloc"] for i in members], axis=0)
            st = ccfg.get("structure", "auto")
            cplx = bool(ccfg.get("complex", False))
            if isinstance(st, str):
                if st == "auto":
                    S = auto_structure(Eloc_lat, self.mode)
                elif st == "full":
                    # Lambda: symmetric labels; R: independent labels (see _bases)
                    S = np.zeros((nlat, nlat), int)
                    iu = np.triu_indices(nlat)
                    S[iu] = np.arange(1, len(iu[0]) + 1)
                    S = S + np.triu(S, 1).T
                elif st == "diagonal":
                    S = np.diag(np.arange(1, nlat + 1))
                else:
                    raise ValueError(f"structure {st}")
            else:
                S = np.array(st, int)
            SR = S
            if isinstance(st, str) and st == "auto":
                idx0 = ref["idx"]
                n0b = np.mean([self.n0_full[np.ix_(self.imps[i]["idx"], self.imps[i]["idx"])]
                               for i in members], axis=0)
                ep = self.eps[:, idx0, :]
                M2 = np.einsum("kab,kbc->ac", ep, self.eps[:, :, idx0]) / self.nk
                S, SR, need_c = auto_structure_sym(
                    [Eloc_lat, n0b, M2],
                    tol_abs=float(ccfg.get("auto_tol", 2e-4)),
                    tol_rel=float(ccfg.get("auto_tol_rel", 1e-3)))
                if need_c and not cplx:
                    cplx = True
                    if self.v:
                        print(f"  class {cname}: complex off-diagonal elements detected -> complex parameters")
                noff = int((np.triu(S, 1) > 0).sum())
                if self.v and noff:
                    print(f"  class {cname}: auto structure keeps {noff} symmetry-allowed "
                          f"off-diagonal element(s)")
            elif isinstance(st, str) and st == "full":
                SR = np.arange(1, nlat * nlat + 1).reshape(nlat, nlat)
            Rb, _ = structure_bases(SR, cplx)
            _, Lb = structure_bases(S, cplx)
            spins = (np.array([1] * norb + [-1] * norb)
                     if (self.para and ccfg.get("sz_sector", True)) else None)
            if self.v:
                print(f"  class {cname}: members={[self.imps[i]['name'] for i in members]}, "
                      f"{norb} orbitals, {len(Rb)} R + {len(Lb)} Lambda parameters")
            emb = Embedding(Uw, spins=spins, verbose=self.v)
            self.classes[cname] = dict(members=members, norb=norb, nso=nso, nlat=nlat,
                                       U=Uw, dc=dc, n0=n0, S=S, Rb=Rb, Lb=Lb,
                                       emb=emb, Eloc=Eloc_lat, Vdc=0.0,
                                       dcUJ=dc_average_UJ(ccfg["interaction"]))
            self._set_dc(cname, None)
        self.cnames = list(self.classes)
        self.scale = 1.0
        self._init_feedback()
        self._init_parallel()

    # ------------------------------------------------------------- parallelism
    def _init_parallel(self):
        n = self.cfg.get("n_procs", 1)
        if n in ("auto", 0, None):
            n = os.cpu_count() or 1
        self.nproc = max(1, int(n))
        self._emb_pool = None
        if self.nproc > 1 and _fork_context() is None:
            print("WARNING: 'fork' start method unavailable; running serially")
            self.nproc = 1
        big = max(c["emb"].fs.dim for c in self.classes.values()) >= int(
            self.cfg.get("parallel_min_embedding_dim", 2000))
        if self.nproc > 1 and len(self.cnames) > 1 and big and \
                self.cfg.get("parallel_classes", True) and not _IN_WORKER:
            global _WORKER
            _WORKER = self
            ctx = _fork_context()
            self._emb_pool = ctx.Pool(min(self.nproc, len(self.cnames)), initializer=_emb_worker_init)
            atexit.register(self.close)
        if self.v and self.nproc > 1:
            print(f"  parallel: {self.nproc} processes (Jacobian columns"
                  + (f", {len(self.cnames)} embedding problems concurrently" if self._emb_pool else "")
                  + ", threaded k-chunks)")

    def close(self):
        p = getattr(self, "_emb_pool", None)
        if p is not None:
            p.terminate()
            p.join()
            self._emb_pool = None

    def _kthreads(self):
        return 1 if (_IN_WORKER or self.nproc <= 1) else min(self.nproc, max(1, self.nk // 64))

    def _eigh_k(self, Hq):
        nt = self._kthreads()
        if nt == 1:
            return np.linalg.eigh(Hq)
        chunks = np.array_split(np.arange(len(Hq)), nt)
        with ThreadPoolExecutor(nt) as ex:
            res = list(ex.map(lambda c: np.linalg.eigh(Hq[c]), chunks))
        return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])

    # ------------------------------------------------------------- charge feedback
    def _init_feedback(self):
        """Set up the linearised Hartree(+xc) feedback (see module docstring)."""
        fcfg = self.cfg.get("charge_feedback", {})
        self.eps_dft = self.eps.copy()
        for c in self.classes.values():
            c["Eloc_dft"] = c["Eloc"].copy()
        for im in self.imps:
            im["Eloc_dft"] = im["Eloc"].copy()
        self.dV_full = np.zeros((self.nl, self.nl), complex)
        self.fb = []           # list of feedback shells
        for cn, c in self.classes.items():
            fb = self.cfg["classes"][cn].get("feedback")
            if not fb:
                continue
            ref = self.imps[c["members"][0]]["idx"]
            n0 = np.mean([self.n0_full[np.ix_(self.imps[i]["idx"], self.imps[i]["idx"])]
                          for i in c["members"]], axis=0)
            self.fb.append(dict(kind="class", name=cn, VH=float(fb.get("V_H", 0.0)),
                                Vorb=float(fb.get("V_orb", 0.0)), n0=n0,
                                nsites=len(c["members"]), dim=c["nlat"]))
        for j, lg in enumerate(fcfg.get("ligands", [])):
            orbs = list(lg["orbitals"])
            idx = np.array(orbs if self.para else orbs + [o + self.nspatial for o in orbs])
            used = set(np.concatenate([im["idx"] for im in self.imps]).tolist())
            if used & set(idx.tolist()):
                raise ValueError(f"ligand shell {lg.get('name', j)} overlaps a correlated shell")
            self.fb.append(dict(kind="ligand", name=lg.get("name", f"lig{j}"), idx=idx,
                                VH=float(lg.get("V_H", 0.0)), Vorb=float(lg.get("V_orb", 0.0)),
                                n0=self.n0_full[np.ix_(idx, idx)], nsites=1, dim=len(idx)))
        self.csc = len(self.fb) > 0
        self.fb_hist = []
        self._outer_rprev = np.inf
        self.fb_vin = np.zeros(sum(2 * f["dim"] ** 2 for f in self.fb))
        self.fb_iter = 0

    def _fb_pack(self, dns):
        return np.concatenate([np.concatenate([d.real.ravel(), d.imag.ravel()]) for d in dns]) \
            if dns else np.zeros(0)

    def _fb_unpack(self, v):
        out, i = [], 0
        for f in self.fb:
            m = f["dim"] ** 2
            d = (v[i:i + m] + 1j * v[i + m:i + 2 * m]).reshape(f["dim"], f["dim"])
            out.append(0.5 * (d + d.conj().T))
            i += 2 * m
        return out

    def _fb_potential(self, f, dn):
        """dV = V_H dN 1 + V_orb dn~^T  and  E_H for one shell (per site)."""
        dim = f["dim"]
        dn = self._fb_project(f, dn)
        dN = self.deg * np.trace(dn).real
        dnt = dn - np.trace(dn) / dim * np.eye(dim)
        h = f["VH"] * dN * np.eye(dim) + f["Vorb"] * dnt.T
        EH = 0.5 * f["VH"] * dN ** 2 + 0.5 * f["Vorb"] * self.deg * np.einsum("ab,ba->", dnt, dnt).real
        return h, EH, dN

    def _fb_project(self, f, dn):
        """Correlated shells: restrict dn to the symmetry/ansatz space spanned by the
        Lambda basis, so that E_H and dV are conjugate within the RISB ansatz (the
        functional stays exactly stationary even for a restricted structure)."""
        if f["kind"] != "class":
            return dn
        Lb = self.classes[f["name"]]["Lb"]
        return sum(np.vdot(b, dn).real * b for b in Lb) if Lb else 0 * dn

    def _fb_apply(self, v):
        """Insert the feedback potential for input density deviations v."""
        dns = self._fb_unpack(v)
        self.dV_full = np.zeros((self.nl, self.nl), complex)
        self.eps = self.eps_dft.copy()
        for c in self.classes.values():
            c["Eloc"] = c["Eloc_dft"].copy()
        for im in self.imps:
            im["Eloc"] = im["Eloc_dft"].copy()
        self.fb_EH, self.fb_h = 0.0, []
        for f, dn in zip(self.fb, dns):
            h, EH, _ = self._fb_potential(f, dn)
            self.fb_h.append(h)
            self.fb_EH += f["nsites"] * EH
            if f["kind"] == "class":
                c = self.classes[f["name"]]
                c["Eloc"] = c["Eloc_dft"] + h
                for i in c["members"]:
                    im = self.imps[i]
                    im["Eloc"] = im["Eloc_dft"] + h
                    self.dV_full[np.ix_(im["idx"], im["idx"])] = h
            else:
                sl = np.ix_(f["idx"], f["idx"])
                self.dV_full[sl] = h
                self.eps[:, f["idx"][:, None], f["idx"][None, :]] += h[None]
        self.Eloc_full = np.zeros((self.nl, self.nl), complex)
        for im in self.imps:
            self.Eloc_full[np.ix_(im["idx"], im["idx"])] = im["Eloc"]

    def _fb_densities(self):
        """Current local density matrices (lattice-space convention, per spin if PM)."""
        out = []
        P = self.last["lattice"]["Pmean"]
        for f in self.fb:
            if f["kind"] == "class":
                c = self.classes[f["name"]]
                G = self.last[f["name"]]["Gamma"]
                no = c["norb"]
                if self.para:
                    n = 0.5 * (G[:no, :no] + G[no:2 * no, no:2 * no])
                else:
                    n = G[:c["nso"], :c["nso"]]
            else:
                n = P[np.ix_(f["idx"], f["idx"])].T
            out.append(n - f["n0"])
        return out

    def _dc_sc_classes(self):
        return [cn for cn, c in self.classes.items()
                if c["dc"].get("type") == "fll" and c["dc"].get("n") == "self_consistent"]

    def _outer_update(self, fcfg):
        """One outer step: Anderson (Pulay) mixing of the combined vector
        [feedback density deviations dn ; self-consistent DC occupations N_dc].

        Mixing both together is essential: updating N_dc directly while
        Anderson-mixing dn changes the map between iterations and corrupts the
        secant model (observed: runaway charge transfer).  Safeguards: the step
        is capped (max_step for dn entries, 5*max_step for N_dc) and the history
        is reset when the residual grows by more than a factor 2.
        Returns (max|r| of the dn part, max|r| of the N_dc part)."""
        dcs = self._dc_sc_classes()
        lev_old = {cn: c["Eloc"] - c["Vdc"] * np.eye(c["nlat"]) for cn, c in self.classes.items()}
        vin_fb = self.fb_vin
        vout_fb = self._fb_pack(self._fb_densities()) if self.csc else np.zeros(0)
        vin_dc = np.array([self.classes[cn]["Ndc"] for cn in dcs])
        vout_dc = np.array([np.trace(self.last[cn]["Gamma"][:self.classes[cn]["nso"],
                                                             :self.classes[cn]["nso"]]).real
                            for cn in dcs])
        vin = np.concatenate([vin_fb, vin_dc])
        r = np.concatenate([vout_fb, vout_dc]) - vin
        nfb = len(vin_fb)
        a = float(fcfg.get("mixing", 0.5))
        m = int(fcfg.get("history", 5))
        cap = np.concatenate([np.full(nfb, float(fcfg.get("max_step", 0.1))),
                              np.full(len(vin_dc), 5 * float(fcfg.get("max_step", 0.1)))])
        rn = float(np.abs(r).max()) if len(r) else 0.0
        if self.fb_hist and rn > 2.0 * self._outer_rprev:
            self.fb_hist = []                        # Anderson restart
        self._outer_rprev = rn
        self.fb_hist.append((vin.copy(), r.copy()))
        self.fb_hist = self.fb_hist[-(m + 1):]
        if len(self.fb_hist) < 2 or m == 0:
            vnew = vin + a * r
        else:
            dV = np.array([self.fb_hist[i + 1][0] - self.fb_hist[i][0]
                           for i in range(len(self.fb_hist) - 1)]).T
            dR = np.array([self.fb_hist[i + 1][1] - self.fb_hist[i][1]
                           for i in range(len(self.fb_hist) - 1)]).T
            gam = np.linalg.lstsq(dR, r, rcond=None)[0]
            vnew = vin + a * r - (dV + a * dR) @ gam
        step = vnew - vin
        if len(step):
            lim = np.max(np.abs(step) / cap)
            if lim > 1.0:
                vnew = vin + step / lim
        if self.csc:
            self.fb_vin = vnew[:nfb]
            self._fb_apply(self.fb_vin)
        for j, cn in enumerate(dcs):
            self._set_dc(cn, float(vnew[nfb + j]))
        rfb = float(np.abs(r[:nfb]).max()) if nfb else 0.0
        rdc = float(np.abs(r[nfb:]).max()) if len(dcs) else 0.0
        # predictor: Lambda follows the change of the local level E_loc + dV - V_dc
        dRL = {cn: (np.zeros((c["nlat"], c["nlat"])),
                    (c["Eloc"] - c["Vdc"] * np.eye(c["nlat"])) - lev_old[cn])
               for cn, c in self.classes.items()}
        self._outer_dx = self.pack(dRL)
        return rfb, rdc

    def _fb_energy_correction(self):
        """E_H - Tr(dV n) at the *current* (converged) input potential."""
        if not self.csc:
            return 0.0, 0.0
        ns = self._fb_densities()
        corr = 0.0
        for f, h, dn in zip(self.fb, self.fb_h, ns):
            n = dn + f["n0"]          # Tr(h n): h lies in the ansatz space, so projection is implicit
            corr -= f["nsites"] * self.deg * np.einsum("ab,ab->", h, n).real
        return self.fb_EH, corr

    # ------------------------------------------------------------- helpers
    def _bases(self, S, cplx, full):
        if full:
            n = S.shape[0]
            SR = np.arange(1, n * n + 1).reshape(n, n)
            Rb, _ = structure_bases(SR, cplx)
            _, Lb = structure_bases(S, cplx)
        else:
            Rb, Lb = structure_bases(S, cplx)
        return Rb, Lb

    def to_so(self, M):
        return np.kron(np.eye(2), M) if self.para else M

    def _set_dc(self, cname, n_override):
        c = self.classes[cname]
        dc = c["dc"]
        typ = dc.get("type", "none").lower()
        if typ == "none":
            c["Vdc"] = 0.0
        elif typ == "fixed":
            c["Vdc"] = float(dc["value"])
        elif typ == "fll":
            U = dc.get("U", c["dcUJ"][0])
            J = dc.get("J", c["dcUJ"][1])
            nd = dc.get("n", "dft")
            if n_override is not None:
                N = n_override
            elif nd == "dft" or nd == "self_consistent":
                N = c["n0"]
            else:
                N = float(nd)
            c["Ndc"] = N
            c["Vdc"] = U * (N - 0.5) - J * (N / 2 - 0.5)
        else:
            raise ValueError(f"dc type {typ}")

    def _find_mu(self, e, N):
        f = lambda mu: self.deg * fermi(e - mu, self.T).sum() / self.nk - N
        lo, hi = e.min() - 20 * self.T - 1, e.max() + 20 * self.T + 1
        return optimize.brentq(f, lo, hi, xtol=1e-13)

    def unpack(self, x):
        out, i = {}, 0
        for cn in self.cnames:
            c = self.classes[cn]
            nr, nlb = len(c["Rb"]), len(c["Lb"])
            R = sum(x[i + j] * c["Rb"][j] for j in range(nr)); i += nr
            L = sum(x[i + j] * c["Lb"][j] for j in range(nlb)); i += nlb
            out[cn] = (R, L)
        return out

    def pack(self, RL):
        x = []
        for cn in self.cnames:
            c = self.classes[cn]
            R, L = RL[cn]
            x += [np.vdot(b, R).real for b in c["Rb"]]
            x += [np.vdot(b, L).real for b in c["Lb"]]
        return np.array(x)

    def initial_guess(self):
        RL = {}
        for cn, c in self.classes.items():
            RL[cn] = (np.eye(c["nlat"], dtype=complex),
                      c["Eloc"] - self.scale * c["Vdc"] * np.eye(c["nlat"]))
        return self.pack(RL)

    # ------------------------------------------------------------- lattice
    def lattice(self, RL, kin=False):
        nl = self.nl
        Rfull = np.eye(nl, dtype=complex)
        Lfull = np.zeros((nl, nl), complex)
        for cn, (R, L) in RL.items():
            for i in self.classes[cn]["members"]:
                sl = np.ix_(self.imps[i]["idx"], self.imps[i]["idx"])
                Rfull[sl] = R
                Lfull[sl] = L
        RE = Rfull[None] @ self.eps
        Hq = RE @ Rfull.conj().T[None] + Lfull[None]
        e, U = self._eigh_k(Hq)
        mu = self._find_mu(e, self.N)
        f = fermi(e - mu, self.T)
        P = (U * f[:, None, :]) @ U.conj().transpose(0, 2, 1)   # rho_k
        out = {"mu": mu}
        for i, im in enumerate(self.imps):
            idx = im["idx"]
            Delta = P[:, idx][:, :, idx].mean(axis=0).T
            K = (P[:, idx, :] @ RE[:, :, idx]).mean(axis=0)
            out[i] = (Delta, K)
        if kin:
            out["Ekin"] = self.deg * np.einsum("kab,kba->", P, Hq - Lfull[None]).real / self.nk
            out["e"], out["f"] = e, f
            out["Pmean"] = P.mean(axis=0)
        return out

    # ------------------------------------------------------------- residual
    def residual(self, x, store=False):
        _t0 = time.perf_counter()
        F = self._residual(x, store)
        dt = time.perf_counter() - _t0
        self._t_res = dt if getattr(self, "_t_res", None) is None else 0.8 * self._t_res + 0.2 * dt
        return F

    def _residual(self, x, store=False):
        RL = self.unpack(x)
        lat = self.lattice(RL, kin=store)
        F = []
        info = {"lattice": lat}
        pre = {}
        for cn in self.cnames:
            c = self.classes[cn]
            R_l, L_l = RL[cn]
            Delta = np.mean([lat[i][0] for i in c["members"]], axis=0)
            K = np.mean([lat[i][1] for i in c["members"]], axis=0)
            R, L = self.to_so(R_l), self.to_so(L_l)
            Dso, Kso = self.to_so(Delta), self.to_so(K)
            n = c["nso"]
            Ah, Aih = sqrt_A(Dso)
            D = Aih @ Kso.conj()
            M = D @ R.T
            Q = dtrace_sqrtA(Dso, M + M.conj().T)
            lam_c = -L - Q.T
            lam_c = 0.5 * (lam_c + lam_c.conj().T)
            hphys = self.to_so(c["Eloc"]) - self.scale * c["Vdc"] * np.eye(n)
            pre[cn] = (R, L, Dso, Ah, D, lam_c, hphys)
        sols = {}
        if self._emb_pool is not None and not _IN_WORKER:
            jobs = [(cn, pre[cn][6], pre[cn][4], pre[cn][5], self.scale, self.classes[cn]["emb"].phi)
                    for cn in self.cnames]
            for cn, E0, G, phi in self._emb_pool.map(_emb_solve, jobs, chunksize=1):
                self.classes[cn]["emb"].phi = phi          # keep the warm start in the parent
                sols[cn] = (E0, G)
        else:
            for cn in self.cnames:
                R, L, Dso, Ah, D, lam_c, hphys = pre[cn]
                sols[cn] = self.classes[cn]["emb"].solve(hphys, D, lam_c)
        for cn in self.cnames:
            c = self.classes[cn]
            n = c["nso"]
            R, L, Dso, Ah, D, lam_c, hphys = pre[cn]
            E0, G = sols[cn]
            F1 = np.eye(n) - G[n:, n:] - Dso
            F2 = G[:n, n:] - R.T @ Ah
            for b in c["Rb"]:
                F.append(np.vdot(self.to_so(b), F2).real / (np.sqrt(2) if self.para else 1))
            for b in c["Lb"]:
                F.append(np.vdot(self.to_so(b), F1).real / (np.sqrt(2) if self.para else 1))
            if store:
                info[cn] = dict(R=R, Lambda=L, Delta=Dso, D=D, lam_c=lam_c, Gamma=G,
                                E0=E0, F1=F1, F2=F2, hphys=hphys)
        F = np.array(F)
        self._fcache = (np.array(x, float).copy(), F.copy())
        if store:
            self.last = info
        if self.v and not store:
            self._ncall = getattr(self, "_ncall", 0) + 1
            print(f"    call {self._ncall:4d}  |F| = {np.linalg.norm(F):.3e}  mu = {lat['mu']:+.6f}",
                  flush=True)
        return F

    # ------------------------------------------------------------- solve
    def solve(self, x0=None):
        scfg = self.cfg.get("solver", {})
        method = scfg.get("method", "hybr")
        tol = float(scfg.get("tol", 1e-9))
        if x0 is None:
            self._set_scale(0.0)          # exact root of the U=0 problem
            x = self.initial_guess()
        else:
            x = np.asarray(x0, float)
        dc_sc = any(c["dc"].get("type") == "fll" and c["dc"].get("n") == "self_consistent"
                    for c in self.classes.values())
        fcfg = self.cfg.get("charge_feedback", {})
        n_outer = 1
        if dc_sc:
            n_outer = int(scfg.get("dc_max_outer", 60))
        if self.csc:
            n_outer = max(n_outer, int(fcfg.get("max_iter", 60)))
            fb_tol = float(fcfg.get("tol", 1e-6))
        # Continuation in the interaction strength: solve at U*s for an
        # increasing sequence of s, starting from the exact U=0 root. This
        # selects the saddle point adiabatically connected to the
        # non-interacting state and avoids metastable roots.
        nramp = scfg.get("interaction_ramp", 5 if x0 is None else 1)
        ramp = (list(np.linspace(0, 1, int(nramp) + 1)[1:]) if np.isscalar(nramp)
                else [float(v) for v in nramp])
        rfb = changed = 1.0
        for outer in range(n_outer):
            t0 = time.time()
            if outer == 0:
                x, sol = self._continuation(x, ramp, method, tol, scfg)
            else:
                self._set_scale(1.0)
                self._ncall = 0
                if fcfg.get("predict_lambda", True) and getattr(self, "_outer_dx", None) is not None:
                    x = x + self._outer_dx
                tol_in = tol
                if fcfg.get("inexact_inner", True):
                    tol_in = max(tol, min(1e-5, 1e-2 * max(rfb, changed)))
                sol = self._root(x, fcfg.get("outer_method", "qn"), tol_in, scfg)
                x = sol.x
            Fn = np.linalg.norm(self.residual(x, store=True))
            if self.v:
                print(f"  root solve: success={sol.success}  |F|={Fn:.3e}  "
                      f"({time.time()-t0:.1f}s)  {getattr(sol,'message','')}")
            if not (dc_sc or self.csc):
                break
            rfb, changed = self._outer_update(fcfg)
            self.fb_iter = outer + 1
            if self.v:
                msg = f"  outer {outer:3d}:"
                if dc_sc:
                    msg += f"  |N_dc,out - N_dc,in| = {changed:.2e}"
                if self.csc:
                    dNs = [self.deg * np.trace(d).real for d in self._fb_unpack(self.fb_vin)]
                    msg += (f"  |n_out - n_in| = {rfb:.2e}  dN_in = "
                            + ", ".join(f"{f['name']}:{x:+.5f}" for f, x in zip(self.fb, dNs)))
                print(msg, flush=True)
            if changed < float(scfg.get("dc_tol", fb_tol if self.csc else 1e-4)) and \
                    (not self.csc or rfb < fb_tol):
                # final RISB solve at the converged (mixed) potential / DC, full tolerance
                if fcfg.get("predict_lambda", True) and getattr(self, "_outer_dx", None) is not None:
                    x = x + self._outer_dx
                sol = self._root(x, fcfg.get("outer_method", "qn"), tol, scfg)
                x = sol.x
                Fn = np.linalg.norm(self.residual(x, store=True))
                break
        else:
            if dc_sc or self.csc:
                print(f"WARNING: outer (DC / charge-feedback) loop not converged in {n_outer} steps")
        self.x = x
        self.success = bool(Fn < float(scfg.get("accept", 1e-6)))
        if not self.success:
            print(f"WARNING: residual {Fn:.2e} above acceptance threshold")
        return x

    def _continuation(self, x, ramp, method, tol, scfg):
        """Adaptive continuation in the interaction scale s (U -> sU, V_dc -> sV_dc).

        * The Jacobian is ill-conditioned at the exact U=0 root, so a first
          small step (solver.ramp_first, default 0.02) is inserted when starting
          from s=0; from there on Newton-type steps converge quickly.
        * Starting points are predicted by secant extrapolation in s.
        * A failed step is retried from the last converged point with half the
          step (up to solver.max_bisect times); intermediate scales use the looser
          acceptance solver.ramp_accept, only s=1 must meet solver.accept.
        """
        accept = float(scfg.get("accept", 1e-6))
        ramp_accept = max(accept, float(scfg.get("ramp_accept", 1e-3)))
        max_bisect = int(scfg.get("max_bisect", 6))
        targets = list(ramp)
        s0 = float(scfg.get("ramp_first", 0.02))
        if self.scale == 0.0 and targets and targets[0] > s0 > 0:
            targets = [s0] + targets
        s_good, x_good = self.scale, x.copy()
        s_prev, x_prev = None, None
        sol = optimize.OptimizeResult(x=x_good, success=True, fun=None,
                                      message="no continuation needed")
        for target in targets:
            step = target - s_good
            nb = 0
            while s_good < target - 1e-14:
                sc = min(target, s_good + step)
                self._set_scale(sc)
                self._ncall = 0
                if self.v:
                    print(f"  interaction scale {sc:.4f}", flush=True)
                xs = x_good
                if s_prev is not None and s_good > s_prev:      # secant predictor
                    xs = x_good + (x_good - x_prev) * (sc - s_good) / (s_good - s_prev)
                sol = self._root(xs, method, tol, scfg)
                Fn = (np.linalg.norm(sol.fun) if getattr(sol, "fun", None) is not None
                      else np.linalg.norm(self.residual(sol.x)))
                thr = accept if sc >= 1.0 - 1e-12 else ramp_accept
                if Fn >= thr and xs is not x_good:            # predictor overshot: retry plain
                    sol2 = self._root(x_good, method, tol, scfg)
                    F2 = np.linalg.norm(sol2.fun)
                    if F2 < Fn:
                        sol, Fn = sol2, F2
                if Fn < thr:
                    s_prev, x_prev = s_good, x_good
                    s_good, x_good = sc, sol.x.copy()
                    if nb:
                        step = min(2 * step, target - s_good)
                else:
                    nb += 1
                    if nb > max_bisect:
                        print(f"WARNING: continuation stalled at scale {sc:.4f} "
                              f"(|F|={Fn:.2e}); last converged scale {s_good:.4f}")
                        return sol.x, sol
                    step *= 0.5
                    if self.v:
                        print(f"  step failed (|F|={Fn:.1e}); halving to {step:.4f}", flush=True)
        self._set_scale(s_good)
        return x_good, sol

    def _set_scale(self, sc):
        self.scale = float(sc)
        for c in self.classes.values():
            c["emb"].scale = self.scale

    def _fd_jacobian(self, x):
        """Finite-difference Jacobian with an absolute step floor.

        MINPACK's internal difference step is purely relative (sqrt(eps)*|x_j|):
        parameters that sit at ~1e-17 (e.g. off-diagonal or zero Lambda entries
        of the U=0 starting point) then get steps ~1e-25 and pure-noise columns,
        which stalls hybr.  Here h_j = max(h, h*|x_j|), h = solver.fd_step."""
        h = float(self.cfg.get("solver", {}).get("fd_step", 1e-6))
        central = self.cfg.get("solver", {}).get("fd_central", False)
        x = np.asarray(x, float)
        c = getattr(self, "_fcache", None)
        F0 = c[1] if (c is not None and c[0].shape == x.shape and np.array_equal(c[0], x)) \
            else self.residual(x)
        J = np.empty((len(F0), len(x)))
        hs = [max(h, h * abs(x[j])) for j in range(len(x))]
        worth = (getattr(self, "_t_res", 1.0) or 1.0) * len(x) > float(
            self.cfg.get("parallel_min_seconds", 0.5))
        if self.nproc > 1 and not _IN_WORKER and len(x) > 1 and worth:
            global _WORKER
            _WORKER = self
            ctx = _fork_context()
            with ctx.Pool(min(self.nproc, len(x)), initializer=_jac_worker_init) as pool:
                for j, col in pool.imap_unordered(
                        _jac_column, [(j, x, hs[j], F0, central) for j in range(len(x))], chunksize=1):
                    J[:, j] = col
            return J
        for j in range(len(x)):
            e = np.zeros_like(x)
            e[j] = hs[j]
            if central:
                J[:, j] = (self.residual(x + e) - self.residual(x - e)) / (2 * hs[j])
            else:
                J[:, j] = (self.residual(x + e) - F0) / hs[j]
        return J

    def _qn_solve(self, x, tol, scfg, maxit=None):
        """Quasi-Newton root solve that keeps its Jacobian between calls.

        Newton steps with the finite-difference Jacobian, Broyden ('good')
        rank-1 updates and backtracking; the Jacobian is recomputed only when a
        step fails.  Because the Jacobian is cached on the solver, consecutive
        solves (outer charge-feedback/DC iterations, continuation steps) usually
        need only a few residual evaluations instead of a fresh n+1 for the
        Jacobian each time.  Converged when |F| < tol."""
        maxit = int(scfg.get("qn_maxit", 60)) if maxit is None else maxit
        x = np.asarray(x, float).copy()
        F = self.residual(x)
        nf = 1
        J = getattr(self, "_J_cache", None)
        fresh = False
        if J is None or J.shape != (len(F), len(x)):
            J = self._fd_jacobian(x); nf += len(x); fresh = True
        fails = 0
        for it in range(maxit):
            Fn0 = np.linalg.norm(F)
            if Fn0 < tol:
                break
            dx = -np.linalg.lstsq(J, F, rcond=None)[0]
            lam, ok = 1.0, False
            while lam >= 1.0 / 32:
                xn = x + lam * dx
                Fn = self.residual(xn); nf += 1
                if np.linalg.norm(Fn) < (1 - 1e-4 * lam) * Fn0:
                    ok = True
                    break
                lam *= 0.5
            if not ok:
                if fresh or fails >= 2:
                    break                       # genuine failure: let the caller fall back
                J = self._fd_jacobian(x); nf += len(x); fresh = True; fails += 1
                continue
            sx, yF = xn - x, Fn - F
            J = J + np.outer(yF - J @ sx, sx) / (sx @ sx)
            x, F, fresh = xn, Fn, False
        self._J_cache = J
        return optimize.OptimizeResult(x=x, fun=F, success=bool(np.linalg.norm(F) < tol), nfev=nf,
                                       message=f"quasi-Newton: {it} iterations, {nf} evaluations")

    def _root(self, x, method, tol, scfg):
        if method == "linear":
            return self._linear_mixing(x, tol, scfg)
        if method == "qn":
            sol = self._qn_solve(x, tol, scfg)
            if not sol.success:                # robust fallback
                if self.v:
                    print(f"  quasi-Newton did not converge (|F|={np.linalg.norm(sol.fun):.1e}); "
                          "falling back to hybr", flush=True)
                self._J_cache = None
                sol = self._root(sol.x, "hybr", tol, scfg)
            return sol
        opts = dict(scfg.get("options", {}))
        jac = self._fd_jacobian if method in ("hybr", "lm") else None
        if method == "hybr":
            opts.setdefault("xtol", tol)
            opts.setdefault("maxfev", int(scfg.get("maxfev", 2000)))
        elif method == "lm":
            opts.setdefault("xtol", tol)
            opts.setdefault("maxiter", int(scfg.get("maxfev", 2000)))
        else:
            opts.setdefault("fatol", tol)
            opts.setdefault("maxiter", int(scfg.get("maxiter", 500)))
        return optimize.root(self.residual, x, method=method, jac=jac, options=opts)

    def _linear_mixing(self, x, tol, scfg):
        """Simple damped fixed-point on the root function (fallback)."""
        alpha = float(scfg.get("mixing", 0.3))
        for it in range(int(scfg.get("maxiter", 500))):
            F = self.residual(x)
            if np.linalg.norm(F) < tol:
                break
            x = x + alpha * F
        return optimize.OptimizeResult(x=x, success=np.linalg.norm(F) < tol,
                                       message=f"linear mixing, {it} iterations")

    # ------------------------------------------------------------- results
    def results(self):
        info = self.last
        lat = info["lattice"]
        res = {"mu": lat["mu"], "mu_U0": self.mu0, "N": self.N,
               "E_kin": lat["Ekin"], "success": self.success,
               "residual_norm": float(np.linalg.norm(self.residual(self.x, store=True))),
               "classes": {}}
        Eloc_tot = 0.0
        for cn, c in self.classes.items():
            d = info[cn]
            n = c["nso"]
            emb = c["emb"]
            G = d["Gamma"]
            R = d["R"]
            Z = R.conj().T @ R
            Eint = emb.expect(emb.Hint).real
            E1 = np.einsum("ab,ab->", d["hphys"], G[:n, :n]).real
            # check degeneracy of the embedding ground state
            H, _ = emb.hamiltonian(d["hphys"], d["D"], d["lam_c"])
            w, _ = emb.ground_state(H, nstates=min(3, emb.fs.dim))
            obs = {}
            if self.para:
                no = c["norb"]
                docc = [emb.expect(emb.operator([(1.0, [(False, m + no), (True, m + no),
                                                        (False, m), (True, m)])])).real
                        for m in range(no)]
                S2 = self._S2(emb, no)
                obs = {"double_occupancy": docc, "S2": S2}
            nm = len(c["members"])
            Eloc_tot += nm * (E1 + Eint)
            res["classes"][cn] = {
                "members": [self.imps[i]["name"] for i in c["members"]],
                "working_basis_levels_eV": np.real(np.diag(c["Eloc"])).tolist(),
                "V_dc": c["Vdc"],
                "Z_diag": np.real(np.diag(Z)).tolist(),
                "Z_eigenvalues": np.linalg.eigvalsh(Z).tolist(),
                "R": _cm(d["R"]), "Lambda": _cm(d["Lambda"]),
                "occupation_physical": np.real(np.diag(G[:n, :n])).tolist(),
                "occupation_qp": np.real(np.diag(d["Delta"])).tolist(),
                "n_total": float(np.trace(G[:n, :n]).real),
                "E_interaction": Eint, "E_onebody_local": E1,
                "embedding_lowest_levels": w.tolist(),
                "embedding_dim": emb.fs.dim,
                **obs,
            }
        # residual of the FULL matrix equations (not only their projection on the
        # parameter basis). Non-zero => the R/Lambda structure is more restrictive
        # than the symmetry; the solution is then not stationary/variational.
        viol = max(max(np.abs(info[cn]["F1"]).max(), np.abs(info[cn]["F2"]).max())
                   for cn in self.cnames)
        res["ansatz_residual"] = float(viol)
        if viol > float(self.cfg.get("solver", {}).get("ansatz_warn", 1e-5)):
            print(f"WARNING: full-matrix residual outside the R/Lambda ansatz = {viol:.2e}. "
                  "The structure is more restrictive than the symmetry; the solution is not "
                  "a stationary point. Use structure 'auto' (symmetry-detecting) or 'full'.")
        res["E_local_total"] = Eloc_tot
        res["E_total"] = res["E_kin"] + Eloc_tot
        f = np.clip(lat["f"], 1e-300, 1.0)
        fc = np.clip(1.0 - lat["f"], 1e-300, 1.0)
        S_qp = -self.deg * np.sum(lat["f"] * np.log(f) + (1 - lat["f"]) * np.log(fc)) / self.nk
        res["S_qp"] = float(S_qp)
        if self.csc:
            EH, corr = self._fb_energy_correction()
            res["E_total_without_feedback_correction"] = res["E_total"]
            res["E_total"] = res["E_total"] + corr + EH
            dns = self._fb_densities()
            res["charge_feedback"] = {
                "outer_iterations": self.fb_iter,
                "E_H": EH, "minus_Tr_dV_n": corr,
                "shells": {f["name"]: {
                    "kind": f["kind"], "V_H": f["VH"], "V_orb": f["Vorb"],
                    "N_dft_per_site": float(self.deg * np.trace(f["n0"]).real),
                    "N_per_site": float(self.deg * np.trace(dn + f["n0"]).real),
                    "dN_per_site": float(self.deg * np.trace(dn).real),
                    "dV": _cm(h)} for f, dn, h in zip(self.fb, dns, self.fb_h)},
            }
        # free energy: the quantity that is exactly stationary at finite smearing
        res["F_total"] = res["E_total"] - self.T * S_qp
        return res

    def _S2(self, emb, no):
        terms = []
        # S^2 = Sz^2 + (S+S- + S-S+)/2, S+ = sum_m c^+_{m up} c_{m dn}
        up = lambda m: m
        dn = lambda m: m + no
        for m in range(no):
            for mp in range(no):
                for s1, s2 in itertools.product((up, dn), repeat=2):
                    sg = (1 if s1 is up else -1) * (1 if s2 is up else -1)
                    terms.append((0.25 * sg, [(False, s2(mp)), (True, s2(mp)),
                                              (False, s1(m)), (True, s1(m))]))
                # S+ S- : c+_{m up} c_{m dn} c+_{m' dn} c_{m' up}
                terms.append((0.5, [(False, up(mp)), (True, dn(mp)), (False, dn(m)), (True, up(m))]))
                terms.append((0.5, [(False, dn(mp)), (True, up(mp)), (False, up(m)), (True, dn(m))]))
        return emb.expect(emb.operator(terms)).real

    # ------------------------------------------------------------- bands
    def bands(self, path):
        """Quasiparticle and bare bands along a k path (fractional coords)."""
        pts = np.array(path["points"], float)
        nseg = int(path.get("npts", 60))
        ks, dist = [], [0.0]
        for a, b in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0, 1, nseg, endpoint=False):
                ks.append(a + t * (b - a))
        ks.append(pts[-1])
        ks = np.array(ks)
        Hk = hk_from_hr(ks, self.rvec, self.wsdeg, self.H_R)
        Hk = 0.5 * (Hk + Hk.conj().transpose(0, 2, 1))
        Hw = self.Tfull.conj().T[None] @ Hk @ self.Tfull[None] + self.dV_full[None]
        eps = Hw - self.Eloc_full[None]
        RL = self.unpack(self.x)
        Rfull = np.eye(self.nl, dtype=complex)
        Lfull = np.zeros((self.nl, self.nl), complex)
        for cn, (R, L) in RL.items():
            for i in self.classes[cn]["members"]:
                sl = np.ix_(self.imps[i]["idx"], self.imps[i]["idx"])
                Rfull[sl], Lfull[sl] = R, L
        Hq = Rfull[None] @ eps @ Rfull.conj().T[None] + Lfull[None]
        eq = np.linalg.eigvalsh(Hq) - self.last["lattice"]["mu"]
        e0 = np.linalg.eigvalsh(Hk) - self.mu0
        # cartesian-free distance (fractional; scale with reciprocal lattice if given)
        B = np.array(path.get("reciprocal_lattice", np.eye(3)), float)
        dk = np.linalg.norm(np.diff(ks @ B, axis=0), axis=1)
        dist = np.concatenate([[0], np.cumsum(dk)])
        ticks = dist[::nseg]
        return dist, eq, e0, ticks


def _cm(M):
    M = np.asarray(M)
    if np.abs(M.imag).max() < 1e-12:
        return np.round(M.real, 10).tolist()
    return {"re": np.round(M.real, 10).tolist(), "im": np.round(M.imag, 10).tolist()}


# =============================================================================
# 7. Command line driver
# =============================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description="RISB solver on a Wannier90 hr.dat")
    ap.add_argument("input", help="JSON input file")
    ap.add_argument("--restart", help="npz from a previous run (initial guess)")
    ap.add_argument("--nproc", help="number of processes (overrides n_procs; 'auto' = all cores)")
    args = ap.parse_args(argv)
    with open(args.input) as f:
        cfg = json.load(f)
    if args.nproc is not None:
        cfg["n_procs"] = args.nproc if args.nproc == "auto" else int(args.nproc)
    outdir = cfg.get("output_dir", "risb_out")
    os.makedirs(outdir, exist_ok=True)
    print(f"risb_wannier {__version__}  —  {time.ctime()}")
    t0 = time.time()
    solver = RISB(cfg)
    print(f"  {solver.nl} lattice orbitals, {solver.nk} k-points, N = {solver.N:.6f}, "
          f"mu(U=0) = {solver.mu0:.6f} eV, T = {solver.T} eV")
    x0 = None
    if args.restart:
        rs = np.load(args.restart)
        x0 = rs["x"]
        if solver.csc and "fb_vin" in rs and len(rs["fb_vin"]) == len(solver.fb_vin):
            solver.fb_vin = rs["fb_vin"]
            solver._fb_apply(solver.fb_vin)
            print("  restart: charge-feedback potential restored")
        if "Ndc" in rs:
            for cn, nd in zip(rs["Ndc_names"], rs["Ndc"]):
                c = solver.classes.get(str(cn))
                if c is not None and c["dc"].get("n") == "self_consistent" and np.isfinite(nd):
                    solver._set_dc(str(cn), float(nd))
            print("  restart: self-consistent double-counting occupations restored")
    x = solver.solve(x0)
    res = solver.results()
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez(os.path.join(outdir, "risb_restart.npz"), x=x, fb_vin=solver.fb_vin,
             Ndc_names=np.array(solver.cnames),
             Ndc=np.array([solver.classes[c].get("Ndc", np.nan) for c in solver.cnames]))
    for cn, r in res["classes"].items():
        print(f"\n  class {cn}  ({', '.join(r['members'])})")
        print("    level (eV)     Z        n_phys (per spin-orbital)")
        lv = r["working_basis_levels_eV"]
        nlv = len(lv)
        for j in range(nlv):
            print(f"    {lv[j]:+9.4f}  {r['Z_diag'][j]:8.5f}  {r['occupation_physical'][j]:8.5f}")
        print(f"    n_total = {r['n_total']:.5f}   V_dc = {r['V_dc']:.4f} eV")
        if "S2" in r:
            print(f"    <S^2> = {r['S2']:.4f}")
    if "charge_feedback" in res:
        print("\n  charge feedback (per site):  shell   N_dft   ->   N      dN")
        for nm, sh in res["charge_feedback"]["shells"].items():
            print(f"    {nm:>10s}  {sh['N_dft_per_site']:8.5f} -> {sh['N_per_site']:8.5f}  "
                  f"{sh['dN_per_site']:+.5f}")
    print(f"\n  mu = {res['mu']:.6f} eV   E_total = {res['E_total']:.8f} eV/cell")
    if "kpath" in cfg:
        dist, eq, e0, ticks = solver.bands(cfg["kpath"])
        np.savetxt(os.path.join(outdir, "qp_bands.dat"), np.column_stack([dist, eq]),
                   header="k_dist  E_qp - mu (eV)")
        np.savetxt(os.path.join(outdir, "bare_bands.dat"), np.column_stack([dist, e0]),
                   header="k_dist  E_U0 - mu_U0 (eV)")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(dist, e0, color="0.7", lw=0.8)
            ax.plot(dist, eq, color="C3", lw=1.0)
            ax.plot([], [], color="0.7", label="U = 0")
            ax.plot([], [], color="C3", label="RISB quasiparticle")
            for t in ticks:
                ax.axvline(t, color="k", lw=0.4)
            ax.axhline(0, color="k", ls="--", lw=0.5)
            labels = cfg["kpath"].get("labels")
            ax.set_xticks(ticks)
            if labels:
                ax.set_xticklabels(labels)
            ax.set_xlim(dist[0], dist[-1])
            ax.set_ylim(*cfg["kpath"].get("ylim", [-2, 2]))
            ax.set_ylabel(r"$E-\mu$ (eV)")
            ax.legend(frameon=False, loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "bands.png"), dpi=200)
        except ImportError:
            pass
    print(f"\n  done in {time.time()-t0:.1f}s; output in {outdir}/")
    solver.close()
    return res


if __name__ == "__main__":
    main()
