#!/usr/bin/env python3
"""Benchmark / regression tests for risb_wannier.py.

Run:  python test_risb_wannier.py          (or: python -m pytest -q test_risb_wannier.py)
"""
import itertools
import os
import tempfile

import numpy as np
from scipy import optimize

import risb_wannier as rw

TMP = tempfile.mkdtemp(prefix="risb_test_")


# ----------------------------------------------------------------------------- helpers
def atomic_spectrum(Uso, N, h1=None):
    n = Uso.shape[0]

    class _A:  # minimal object to reuse Embedding._two_body on an n-mode space
        pass
    a = _A()
    a.n, a.fs = n, rw.FockSector(n, N)
    H = rw.Embedding._two_body(a, Uso).toarray()
    if h1 is not None:
        for p, q in itertools.product(range(n), repeat=2):
            r, c, sg = a.fs.action([(False, q), (True, p)])
            H[r, c] += h1[p, q] * sg
    return np.linalg.eigvalsh(H)


def model_hr(hops, nw, fname):
    """hops: dict {(R1,R2,R3): (nw x nw) matrix}; adds Hermitian partners."""
    H = {}
    for R, M in hops.items():
        M = np.asarray(M, complex)
        H[R] = H.get(R, 0) + M
        mR = tuple(-r for r in R)
        if mR != R:
            H[mR] = H.get(mR, 0) + M.conj().T
    keys = sorted(H)
    rvec = np.array(keys)
    hr = np.array([H[k] for k in keys])
    deg = np.ones(len(keys), int)
    rw.write_hr(fname, rvec, deg, hr)
    return rvec, deg, hr


def two_orbital_hops(t1=-1.0, t2=-0.5, cf=0.4, tab=0.0, tabc=0.0):
    """Square lattice, two orbitals; optional inter-orbital NN hopping tab and
    complex inter-orbital term tabc."""
    on = np.array([[-cf / 2, 0], [0, cf / 2]])
    hx = np.array([[t1, tab + 1j * tabc], [tab - 1j * tabc, t2]])
    hy = np.array([[t1, tab - 1j * tabc], [tab + 1j * tabc, t2]])
    return {(0, 0, 0): on, (1, 0, 0): hx, (0, 1, 0): hy}


def run(cfg, hr, verbose=False):
    s = rw.RISB(cfg, hr=hr, verbose=verbose)
    s.solve()
    return s, s.results()


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    assert ok, name


# ----------------------------------------------------------------------------- tests
def test_kanamori_atomic():
    for U, J in [(4.0, 0.7), (3.0, 0.0), (5.0, 1.2)]:
        Uso = rw.spin_orbital_tensor(rw.kanamori_orbital_tensor(2, U, J))
        e = atomic_spectrum(Uso, 2)
        ref = np.sort([U - 3 * J] * 3 + [U - J] * 2 + [U + J])
        check(f"Kanamori 2orb/2e multiplets U={U} J={J}", np.allclose(e, ref, atol=1e-12))


def test_kanamori_O_invariance():
    rng = np.random.default_rng(1)
    Uorb = rw.kanamori_orbital_tensor(3, 4.0, 0.6)
    O, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Ur = rw.rotate_tensor(Uorb.astype(complex), O)
    check("Kanamori (U'=U-2J) invariant under real O(3) orbital rotation",
          np.allclose(Ur, Uorb, atol=1e-12))


def test_slater_d2():
    F0, F2, F4 = 5.0, 7.0, 4.375
    Uso = rw.spin_orbital_tensor(rw.slater_orbital_tensor(2, [F0, F2, F4]))
    e = atomic_spectrum(Uso, 2)
    A, B, C = F0 - F4 / 9, F2 / 49 - 5 * F4 / 441, 35 * F4 / 441
    ref = np.sort([A - 8 * B] * 21 + [A - 3 * B + 2 * C] * 5 + [A + 7 * B] * 9
                  + [A + 4 * B + 2 * C] * 9 + [A + 14 * B + 7 * C])
    check("Slater d^2 term energies (3F,1D,3P,1G,1S) vs Racah formulas",
          np.allclose(e, ref, atol=1e-10), f"max err {np.abs(e-ref).max():.1e}")
    # SO(3) invariance: d-orbital rotation matrix from a random rotation
    rng = np.random.default_rng(2)
    O, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    O *= np.sign(np.linalg.det(O))
    # rotate real d-orbitals by evaluating them on points: build 5x5 rep numerically
    pts = rng.normal(size=(40, 3))
    def dfun(p):
        x, y, z = p.T
        r2 = x * x + y * y + z * z
        return np.stack([(3 * z * z - r2) / np.sqrt(3), 2 * x * z, 2 * y * z, x * x - y * y, 2 * x * y], 1)
    D = np.linalg.lstsq(dfun(pts), dfun(pts @ O.T), rcond=None)[0]
    Uorb = rw.slater_orbital_tensor(2, [F0, F2, F4])
    check("Slater d tensor invariant under SO(3) rotation of real d orbitals",
          np.allclose(rw.rotate_tensor(Uorb.astype(complex), D), Uorb, atol=1e-10))


def test_hr_roundtrip():
    hops = two_orbital_hops(tab=0.2, tabc=0.15)
    f = os.path.join(TMP, "rt_hr.dat")
    rvec, deg, hr = model_hr(hops, 2, f)
    r2, d2, h2 = rw.read_hr(f)
    k = rw.monkhorst_pack([7, 5, 1])
    check("hr.dat write/read round trip", np.allclose(rw.hk_from_hr(k, rvec, deg, hr),
                                                      rw.hk_from_hr(k, r2, d2, h2), atol=1e-11))


def base_cfg(norb=2, U=0.0, J=0.0, n=2.0, mesh=(24, 24, 1), basis="wannier",
             structure="full", cplx=False, T=0.01, itype="kanamori"):
    return {
        "spin_mode": "paramagnetic", "kmesh": list(mesh), "temperature": T,
        "n_electrons": n,
        "impurities": [{"name": "A", "orbitals": list(range(norb)), "class": "A"}],
        "classes": {"A": {"interaction": {"type": itype, "U": U, "J": J},
                          "basis": basis, "structure": structure, "complex": cplx}},
        "solver": {"method": "hybr", "tol": 1e-11},
    }


def test_U0_identity():
    hops = two_orbital_hops(tab=0.25, tabc=0.2)
    hops[(0, 0, 0)] = np.array([[-0.3, 0.1 + 0.05j], [0.1 - 0.05j, 0.3]])  # complex on-site mixing
    hr = model_hr(hops, 2, os.path.join(TMP, "u0_hr.dat"))
    for basis, st, cplx in [("wannier", "full", True), ("cf", "full", True)]:
        cfg = base_cfg(U=0.0, n=1.6, basis=basis, structure=st, cplx=cplx)
        cfg["classes"]["A"]["interaction"] = {"type": "none"}
        s, r = run(cfg, hr)
        R = s.last["A"]["R"]
        L = s.last["A"]["Lambda"]
        Eloc = s.to_so(s.classes["A"]["Eloc"])
        # exact band energy at U=0
        e = np.linalg.eigvalsh(s.H0w)
        Eband = 2 * (rw.fermi(e - s.mu0, s.T) * e).sum() / s.nk
        Erisb = r["E_total"]
        # gauge freedom R -> uR, Lambda -> u Lambda u^+ : compare gauge-invariant combinations
        Z = R.conj().T @ R
        RLR = R.conj().T @ L @ R
        ok = (np.abs(Z - np.eye(4)).max() < 1e-8 and np.abs(RLR - Eloc).max() < 1e-8
              and abs(Erisb - Eband) < 1e-8 and abs(r["mu"] - s.mu0) < 1e-8)
        check(f"U=0 -> Z=R^+R=1, R^+ Lambda R=E_loc, E=band energy ({basis}, complex hopping)", ok,
              f"|Z-1|={np.abs(Z-np.eye(4)).max():.1e} |R+LR-Eloc|={np.abs(RLR-Eloc).max():.1e} "
              f"dE={abs(Erisb-Eband):.1e}")


def ga_single_band(eps_k, n, U, T):
    """Analytic paramagnetic Gutzwiller approximation (Gutzwiller 1965).
    At finite smearing T the RISB saddle point extremises the quasiparticle
    *free energy*  F(d) = Omega_qp(q(d)) + mu*n + U*d  (Omega = -T sum ln(1+e^{-(q eps-mu)/T})),
    so the reference minimises the same functional. Returns (E, q, d)."""
    ns = n / 2

    def q_of(d):
        return (np.sqrt((ns - d) * (1 - 2 * ns + d)) + np.sqrt(d * (ns - d))) ** 2 / (ns * (1 - ns))

    def parts(d):
        q = q_of(d)
        e = q * eps_k
        mu = optimize.brentq(lambda m: 2 * rw.fermi(e - m, T).mean() - n,
                             e.min() - 1, e.max() + 1, xtol=1e-14)
        x = (e - mu) / T
        omega = -2 * T * np.mean(np.logaddexp(0, -x))
        F = omega + mu * n + U * d
        E = 2 * (rw.fermi(e - mu, T) * e).mean() + U * d
        return F, E, q
    lo, hi = max(0.0, 2 * ns - 1) + 1e-12, ns * ns
    res = optimize.minimize_scalar(lambda d: parts(d)[0], bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-14})
    F, E, q = parts(res.x)
    return E, q, res.x


def test_single_band_gutzwiller():
    t = -1.0
    hops = {(1, 0, 0): [[t]], (0, 1, 0): [[t]]}
    hr = model_hr(hops, 1, os.path.join(TMP, "sq_hr.dat"))
    T = 0.02
    k = rw.monkhorst_pack([96, 96, 1])
    eps_k = np.real(rw.hk_from_hr(k, *hr)[:, 0, 0])
    for n, U in [(1.0, 4.0), (1.0, 8.0), (0.8, 6.0), (0.8, 12.0)]:
        cfg = base_cfg(norb=1, U=U, J=0.0, n=n, mesh=(96, 96, 1), T=T)
        s, r = run(cfg, hr)
        Eref, qref, dref = ga_single_band(eps_k, n, U, T)
        Z = r["classes"]["A"]["Z_diag"][0]
        docc = r["classes"]["A"]["double_occupancy"][0]
        ok = abs(Z - qref) < 2e-6 and abs(r["E_total"] - Eref) < 2e-7 and abs(docc - dref) < 2e-6
        check(f"single-band GA n={n} U={U}", ok,
              f"Z={Z:.6f} (ref {qref:.6f})  d={docc:.6f} (ref {dref:.6f})  "
              f"E={r['E_total']:.8f} (ref {Eref:.8f})")


def test_brinkman_rice():
    """Half filling, T->0 limit: Z = 1 - (U/Uc)^2 with Uc = 8|eps_bar|, eps_bar the
    U=0 kinetic energy per site summed over both spins."""
    t = -1.0
    hops = {(1, 0, 0): [[t]], (0, 1, 0): [[t]]}
    hr = model_hr(hops, 1, os.path.join(TMP, "sq2_hr.dat"))
    T = 0.004
    k = rw.monkhorst_pack([200, 200, 1])
    e = np.real(rw.hk_from_hr(k, *hr)[:, 0, 0])
    eps_bar = 2 * (rw.fermi(e, T) * e).mean()
    Uc = 8 * abs(eps_bar)
    for U in [0.4 * Uc, 0.8 * Uc]:
        cfg = base_cfg(norb=1, U=U, n=1.0, mesh=(200, 200, 1), T=T)
        s, r = run(cfg, hr)
        Z = r["classes"]["A"]["Z_diag"][0]
        ref = 1 - (U / Uc) ** 2
        check(f"Brinkman-Rice U/Uc={U/Uc:.1f}", abs(Z - ref) < 5e-3,
              f"Z={Z:.5f}  analytic(T=0)={ref:.5f}")


def test_basis_invariance():
    """Complex unitary rotation of the orbital basis with a U(4)-invariant
    interaction must leave E, mu, spectrum of Z unchanged."""
    rng = np.random.default_rng(7)
    hops = two_orbital_hops(t1=-1.0, t2=-0.4, cf=0.5, tab=0.15, tabc=0.1)
    hr0 = model_hr(hops, 2, os.path.join(TMP, "inv0_hr.dat"))
    X = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    W, _ = np.linalg.qr(X)
    hops_r = {R: W.conj().T @ np.asarray(M) @ W for R, M in hops.items()}
    hr1 = model_hr(hops_r, 2, os.path.join(TMP, "inv1_hr.dat"))
    out = []
    for hr, basis in [(hr0, "wannier"), (hr1, "wannier"), (hr1, "cf")]:
        cfg = base_cfg(U=3.0, J=0.0, n=1.4, mesh=(20, 20, 1), basis=basis,
                       structure="full", cplx=True)
        s, r = run(cfg, hr)
        out.append((r["E_total"], r["mu"], np.sort(r["classes"]["A"]["Z_eigenvalues"]),
                    r["residual_norm"]))
    for j in (1, 2):
        dE = abs(out[j][0] - out[0][0]); dmu = abs(out[j][1] - out[0][1])
        dZ = np.abs(out[j][2] - out[0][2]).max()
        check(f"complex unitary basis invariance (case {j})", dE < 1e-8 and dmu < 1e-8 and dZ < 1e-7,
              f"dE={dE:.1e} dmu={dmu:.1e} dZ={dZ:.1e} Z={out[0][2][::2]}")


def test_off_diagonal_R_needed():
    """With inter-orbital hopping, a diagonal-R ansatz in a rotated basis is
    variationally worse (higher E) than the full matrix: sanity of the
    structure machinery."""
    hops = two_orbital_hops(t1=-1.0, t2=-0.4, cf=0.3, tab=0.3)
    hr = model_hr(hops, 2, os.path.join(TMP, "od_hr.dat"))
    Es = {}
    for st in ["full", "diagonal"]:
        cfg = base_cfg(U=3.0, J=0.5, n=1.5, mesh=(20, 20, 1), basis="wannier", structure=st)
        s, r = run(cfg, hr)
        Es[st] = r["E_total"]
    check("full-matrix R gives lower energy than diagonal ansatz",
          Es["full"] < Es["diagonal"] - 1e-6, f"E_full={Es['full']:.6f} E_diag={Es['diagonal']:.6f}")


def test_equivalence():
    """Two identical sites in a doubled cell: one shared class == two classes."""
    t = -1.0
    hops = {(0, 0, 0): [[0, t], [t, 0]],
            (1, 0, 0): [[0, 0], [t, 0]], (0, 1, 0): [[t, 0], [0, t]]}
    hr = model_hr(hops, 2, os.path.join(TMP, "eq_hr.dat"))
    res = []
    for shared in (True, False):
        cfg = {"spin_mode": "paramagnetic", "kmesh": [24, 24, 1], "temperature": 0.01,
               "n_electrons": 1.7,
               "impurities": [{"name": "s1", "orbitals": [0], "class": "X" if shared else "X1"},
                              {"name": "s2", "orbitals": [1], "class": "X" if shared else "X2"}],
               "classes": {}, "solver": {"method": "hybr", "tol": 1e-11}}
        for c in (["X"] if shared else ["X1", "X2"]):
            cfg["classes"][c] = {"interaction": {"type": "kanamori", "U": 5.0, "J": 0.0}}
        s, r = run(cfg, hr)
        res.append(r)
    dE = abs(res[0]["E_total"] - res[1]["E_total"])
    Z0 = res[0]["classes"]["X"]["Z_diag"][0]
    Z1, Z2 = res[1]["classes"]["X1"]["Z_diag"][0], res[1]["classes"]["X2"]["Z_diag"][0]
    check("site equivalence (shared class == independent classes)",
          dE < 1e-8 and abs(Z0 - Z1) < 1e-7 and abs(Z0 - Z2) < 1e-7,
          f"dE={dE:.1e} Z={Z0:.6f},{Z1:.6f},{Z2:.6f}")


def test_spinful_vs_paramagnetic():
    hops = two_orbital_hops(t1=-1.0, t2=-0.6, cf=0.3, tab=0.2)
    hr = model_hr(hops, 2, os.path.join(TMP, "sp_hr.dat"))
    cfg = base_cfg(U=3.0, J=0.6, n=1.8, mesh=(16, 16, 1), basis="cf", structure="auto")
    s, rp = run(cfg, hr)
    # spinful hr.dat, interleaved ordering (up,dn per orbital)
    rvec, deg, H = hr
    Hs = np.array([np.kron(h, np.eye(2)) for h in H])
    cfg2 = dict(cfg, spin_mode="spinful", spin_ordering="interleaved")
    s2, rs = run(cfg2, (rvec, deg, Hs))
    dE = abs(rp["E_total"] - rs["E_total"])
    dZ = np.abs(np.sort(rp["classes"]["A"]["Z_diag"]) - np.sort(rs["classes"]["A"]["Z_diag"])).max()
    check("spinful (interleaved, no Sz restriction) == paramagnetic (Sz=0 sector)",
          dE < 1e-8 and dZ < 1e-7, f"dE={dE:.1e} dZ={dZ:.1e} E={rp['E_total']:.8f}")


def test_kanamori_three_orbital_hund():
    """Degenerate 3-orbital (t2g-like) model, 2 electrons: Hund's coupling must
    increase <S^2> and the embedding state must conserve particle number."""
    hops = {(1, 0, 0): np.diag([-1.0, -1.0, -1.0]) * 0.5, (0, 1, 0): np.diag([-1.0, -1.0, -1.0]) * 0.5,
            (0, 0, 1): np.diag([-1.0, -1.0, -1.0]) * 0.5}
    hr = model_hr(hops, 3, os.path.join(TMP, "t2g_hr.dat"))
    S2 = []
    for J in (0.0, 0.6):
        cfg = base_cfg(norb=3, U=4.0, J=J, n=2.0, mesh=(10, 10, 10), basis="cf", structure="auto")
        s, r = run(cfg, hr)
        c = r["classes"]["A"]
        S2.append(c["S2"])
        assert abs(c["n_total"] - 2.0) < 1e-6
        assert r["residual_norm"] < 1e-8
    check("3-orbital Kanamori: Hund's J raises local <S^2>", S2[1] > S2[0] + 0.1,
          f"<S^2>(J=0)={S2[0]:.4f}  <S^2>(J=0.6)={S2[1]:.4f}")


def test_cli_end_to_end():
    import json
    hops = two_orbital_hops(t1=-1.0, t2=-0.5, cf=0.4, tab=0.1)
    f = os.path.join(TMP, "cli_hr.dat")
    model_hr(hops, 2, f)
    cfg = base_cfg(U=2.5, J=0.4, n=1.5, mesh=(16, 16, 1), basis="cf", structure="auto")
    cfg["hr_file"] = f
    cfg["output_dir"] = os.path.join(TMP, "cli_out")
    cfg["classes"]["A"]["dc"] = {"type": "fll", "n": "self_consistent"}
    cfg["kpath"] = {"points": [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]],
                    "labels": ["G", "X", "M", "G"], "npts": 30}
    inp = os.path.join(TMP, "cli.json")
    json.dump(cfg, open(inp, "w"))
    r = rw.main([inp])
    check("CLI end-to-end with self-consistent FLL double counting + bands",
          r["residual_norm"] < 1e-8 and os.path.exists(os.path.join(cfg["output_dir"], "qp_bands.dat")),
          f"V_dc={r['classes']['A']['V_dc']:.4f}  n={r['classes']['A']['n_total']:.4f}")



# ----------------------------------------------------------------------------- charge feedback
def dp_model(ed=0.0, ep=-1.0, tdd=-0.4, tpp=-0.6, tpd=0.7):
    """Square-lattice d-p model: orbital 0 = correlated 'd', orbital 1 = ligand 'p'."""
    on = np.array([[ed, tpd], [tpd, ep]])
    hx = np.array([[tdd, tpd], [-tpd, tpp]])
    hy = np.array([[tdd, tpd], [-tpd, tpp]])
    return {(0, 0, 0): on, (1, 0, 0): hx, (0, 1, 0): hy}


def fb_cfg(U, J=0.0, VH=0.0, Vorb=0.0, VHp=None, n=2.4, T=0.003, mesh=(40, 40, 1), dc=None):
    cfg = {"spin_mode": "paramagnetic", "kmesh": list(mesh), "temperature": T, "n_electrons": n,
           "impurities": [{"name": "d", "orbitals": [0], "class": "d"}],
           "classes": {"d": {"interaction": {"type": "kanamori", "U": U, "J": J},
                             "feedback": {"V_H": VH, "V_orb": Vorb}}},
           "charge_feedback": {"mixing": 0.6, "history": 5, "tol": 1e-10, "max_iter": 200},
           "solver": {"method": "hybr", "tol": 1e-13}}
    if dc is not None:
        cfg["classes"]["d"]["dc"] = dc
    if VHp is not None:
        cfg["charge_feedback"]["ligands"] = [{"name": "p", "orbitals": [1], "V_H": VHp}]
    return cfg


def test_feedback_zero_is_one_shot():
    hr = model_hr(dp_model(), 2, os.path.join(TMP, "dp_hr.dat"))
    _, r0 = run({k: v for k, v in fb_cfg(4.0).items() if k != "charge_feedback"} |
                {"classes": {"d": {"interaction": {"type": "kanamori", "U": 4.0, "J": 0.0}}}}, hr)
    _, r1 = run(fb_cfg(4.0, VH=0.0), hr)
    check("feedback with V_H=V_orb=0 reproduces one-shot", abs(r0["E_total"] - r1["E_total"]) < 1e-11,
          f"dE={abs(r0['E_total']-r1['E_total']):.1e}")


def independent_hartree(hr, VHd, VHp, n, T, mesh, shift_d=0.0):
    """Stand-alone Hartree solve for U=0: H(k) + diag(shift_d + VHd dNd, VHp dNp),
    dN measured from the unshifted (DFT) density."""
    k = rw.monkhorst_pack(mesh)
    H = rw.hk_from_hr(k, *hr)
    def occ(Hk):
        e, V = np.linalg.eigh(Hk)
        mu = optimize.brentq(lambda m: 2 * rw.fermi(e - m, T).sum() / len(k) - n, e.min() - 5, e.max() + 5, xtol=1e-15)
        f = rw.fermi(e - mu, T)
        return 2 * np.einsum("kan,kn,kan->a", V, f, V.conj()).real / len(k)
    n0 = occ(H)
    def resid(dN):
        Hk = H + np.diag([shift_d + VHd * dN[0], VHp * dN[1]])[None]
        return occ(Hk) - n0 - dN
    dN = optimize.fsolve(resid, np.zeros(2), xtol=1e-14)
    return n0, dN


def test_feedback_vs_independent_hartree():
    hr = model_hr(dp_model(), 2, os.path.join(TMP, "dp_hr.dat"))
    mesh, T, n = (40, 40, 1), 0.003, 2.4
    for VHd, VHp in [(1.5, 0.0), (2.0, 1.0)]:
        cfg = fb_cfg(0.0, VH=VHd, VHp=VHp if VHp else None, n=n, T=T, mesh=mesh)
        cfg["classes"]["d"]["interaction"] = {"type": "none"}
        cfg["classes"]["d"]["dc"] = {"type": "fixed", "value": 0.8}   # drives d -> p charge transfer
        s, r = run(cfg, hr)
        n0, dN = independent_hartree(hr, VHd, VHp, n, T, mesh, shift_d=-0.8)
        sh = r["charge_feedback"]["shells"]
        got_d = sh["d"]["dN_per_site"]
        ok = abs(got_d - dN[0]) < 1e-8 and abs(dN[0]) > 1e-2
        msg = f"dN_d={got_d:+.8f} (ref {dN[0]:+.8f})"
        if VHp:
            ok &= abs(sh["p"]["dN_per_site"] - dN[1]) < 1e-8
            msg += f"  dN_p={sh['p']['dN_per_site']:+.8f} (ref {dN[1]:+.8f})"
        check(f"U=0 feedback == independent Hartree solver (V_H,d={VHd}, V_H,p={VHp})", ok, msg)


def test_feedback_hellmann_feynman():
    """Total functional must be stationary: dE/dV_H = 1/2 dN^2 per site."""
    hr = model_hr(dp_model(), 2, os.path.join(TMP, "dp_hr.dat"))
    VH, h = 1.0, 0.05
    _, r = run(fb_cfg(5.0, VH=VH), hr)
    dN = r["charge_feedback"]["shells"]["d"]["dN_per_site"]
    F = [run(fb_cfg(5.0, VH=v), hr)[1]["F_total"] for v in (VH - h, VH + h, VH - h / 2, VH + h / 2)]
    d1, d2 = (F[1] - F[0]) / (2 * h), (F[3] - F[2]) / h
    dFdV = (4 * d2 - d1) / 3                      # Richardson-extrapolated derivative
    check("Hellmann-Feynman for feedback: dF/dV_H = dN^2/2", abs(dFdV - 0.5 * dN ** 2) < 1e-7,
          f"dF/dV_H={dFdV:.9f}  dN^2/2={0.5*dN**2:.9f}  (dN={dN:+.5f})")


def test_feedback_hellmann_feynman_orbital():
    """Two-orbital correlated shell + V_orb: dE/dV_orb = 1/2 deg Tr(dn~^2)."""
    hops = two_orbital_hops(t1=-1.0, t2=-0.4, cf=0.6, tab=0.2)
    hr = model_hr(hops, 2, os.path.join(TMP, "orb_hr.dat"))
    def cfg(Vorb):
        c = base_cfg(U=3.0, J=0.6, n=1.5, mesh=(32, 32, 1), basis="cf", structure="auto", T=0.003)
        c["classes"]["A"]["feedback"] = {"V_H": 0.5, "V_orb": Vorb}
        c["charge_feedback"] = {"mixing": 0.6, "tol": 1e-10, "max_iter": 200}
        c["solver"]["tol"] = 1e-13
        return c
    V, h = 0.8, 0.05
    s, r = run(cfg(V), hr)
    dn = s._fb_densities()[0]
    # symmetry-detecting 'auto' keeps the allowed off-diagonal element -> full dn enters
    dnt = dn - np.trace(dn) / 2 * np.eye(2)
    ref = 0.5 * 2 * np.einsum("ab,ba->", dnt, dnt).real
    assert abs(dn[0, 1]) > 1e-3        # the model really has off-diagonal density (non-trivial test)
    assert r["ansatz_residual"] < 1e-9
    F = [run(cfg(v), hr)[1]["F_total"] for v in (V - h, V + h, V - h / 2, V + h / 2)]
    dEdV = (4 * (F[3] - F[2]) / h - (F[1] - F[0]) / (2 * h)) / 3
    # orbital polarisation must be reduced relative to V_orb = 0
    s0, _ = run(cfg(0.0), hr)
    dn0 = s0._fb_densities()[0]
    pol = lambda d: (d[0, 0] - d[1, 1]).real
    check("Hellmann-Feynman for orbital feedback: dF/dV_orb = Tr(dn~^2)", abs(dEdV - ref) < 1e-7,
          f"dF/dV_orb={dEdV:.9f}  ref={ref:.9f}; orbital polarisation {pol(dn0):+.4f} -> {pol(dn):+.4f}")


def test_feedback_pins_charge():
    hr = model_hr(dp_model(), 2, os.path.join(TMP, "dp_hr.dat"))
    dNs = [run(fb_cfg(6.0, VH=v), hr)[1]["charge_feedback"]["shells"]["d"]["dN_per_site"]
           for v in (0.0, 1.0, 4.0, 20.0, 100.0)]
    mono = all(abs(dNs[i + 1]) < abs(dNs[i]) for i in range(4))
    check("increasing V_H monotonically pins the d charge to its DFT value", mono and abs(dNs[-1]) < 0.1 * abs(dNs[0]),
          "dN_d = " + ", ".join(f"{x:+.5f}" for x in dNs))


def test_feedback_equivalence_and_spinful():
    t = -1.0
    hops = {(0, 0, 0): [[0.3, t], [t, -0.3]], (1, 0, 0): [[0, 0], [t, 0]], (0, 1, 0): [[t, 0], [0, t]]}
    hr = model_hr(hops, 2, os.path.join(TMP, "eqfb_hr.dat"))
    def cfg(shared, spinful=False):
        c = {"spin_mode": "spinful" if spinful else "paramagnetic", "kmesh": [24, 24, 1],
             "temperature": 0.01, "n_electrons": 1.7,
             "impurities": [{"name": "s1", "orbitals": [0], "class": "X" if shared else "X1"},
                            {"name": "s2", "orbitals": [1], "class": "X" if shared else "X1"}],
             "classes": {}, "charge_feedback": {"tol": 1e-11, "max_iter": 200},
             "solver": {"method": "hybr", "tol": 1e-12}}
        if spinful:
            c["spin_ordering"] = "block"
        # sites are NOT equivalent (on-site +-0.3): use separate classes
        c["impurities"][1]["class"] = "X2"
        c["impurities"][0]["class"] = "X1"
        for cn in ("X1", "X2"):
            c["classes"][cn] = {"interaction": {"type": "kanamori", "U": 5.0, "J": 0.0},
                                "feedback": {"V_H": 1.2}}
        return c
    s, rp = run(cfg(False), hr)
    rvec, deg, H = hr
    Hs = np.array([np.kron(np.eye(2), h) for h in H])        # block spin ordering
    s2, rs = run(cfg(False, spinful=True), (rvec, deg, Hs))
    dE = abs(rp["E_total"] - rs["E_total"])
    dN = max(abs(rp["charge_feedback"]["shells"][c]["dN_per_site"] - rs["charge_feedback"]["shells"][c]["dN_per_site"])
             for c in ("X1", "X2"))
    tot = sum(rp["charge_feedback"]["shells"][c]["dN_per_site"] for c in ("X1", "X2"))
    check("feedback: spinful == paramagnetic, charge conserved (sum dN = 0 in a 2-site model)",
          dE < 1e-8 and dN < 1e-8 and abs(tot) < 1e-9,
          f"dE={dE:.1e} d(dN)={dN:.1e} sum dN={tot:.1e}")


def test_feedback_cli_restart():
    import json
    f = os.path.join(TMP, "dpcli_hr.dat")
    model_hr(dp_model(), 2, f)
    cfg = fb_cfg(5.0, J=0.0, VH=1.0, VHp=0.5, mesh=(24, 24, 1),
                 dc={"type": "fll", "n": "self_consistent"})
    cfg["hr_file"] = f
    cfg["output_dir"] = os.path.join(TMP, "dpcli_out")
    cfg["charge_feedback"]["tol"] = 1e-8
    inp = os.path.join(TMP, "dpcli.json")
    json.dump(cfg, open(inp, "w"))
    r1 = rw.main([inp])
    r2 = rw.main([inp, "--restart", os.path.join(cfg["output_dir"], "risb_restart.npz")])
    ok = (abs(r1["E_total"] - r2["E_total"]) < 1e-7 and r2["charge_feedback"]["outer_iterations"] <= 2
          and r1["residual_norm"] < 1e-8)
    check("CLI: feedback + self-consistent FLL DC + ligand shell; restart is immediate", ok,
          f"outer its {r1['charge_feedback']['outer_iterations']} -> {r2['charge_feedback']['outer_iterations']}, "
          f"dE={abs(r1['E_total']-r2['E_total']):.1e}")


def test_auto_structure_symmetry():
    """(i) symmetry-allowed off-diagonal element must be detected and give the
    same solution as 'full'; (ii) a too-restrictive ansatz must be flagged;
    (iii) when symmetry forbids mixing, 'auto' stays diagonal."""
    hops = two_orbital_hops(t1=-1.0, t2=-0.4, cf=0.6, tab=0.2)
    hr = model_hr(hops, 2, os.path.join(TMP, "orb_hr.dat"))
    out = {}
    for st, basis in [("auto", "cf"), ("full", "wannier"), ("diagonal", "cf")]:
        c = base_cfg(U=3.0, J=0.6, n=1.5, mesh=(32, 32, 1), basis=basis, structure=st, T=0.003)
        c["solver"]["tol"] = 1e-13
        s, r = run(c, hr)
        out[st] = (r["F_total"], r["ansatz_residual"], len(s.classes["A"]["Lb"]))
    ok = (abs(out["auto"][0] - out["full"][0]) < 1e-9 and out["auto"][1] < 1e-9
          and out["diagonal"][1] > 1e-3 and out["diagonal"][0] > out["full"][0] + 1e-3)
    check("auto structure detects symmetry-allowed mixing (== full); diagonal ansatz is flagged", ok,
          f"F_auto-F_full={out['auto'][0]-out['full'][0]:.1e}, ansatz residual: auto {out['auto'][1]:.1e}, "
          f"diagonal {out['diagonal'][1]:.1e} (F higher by {out['diagonal'][0]-out['full'][0]:.4f} eV)")
    # symmetry forbids mixing: no inter-orbital hopping (orbital parity)
    hr2 = model_hr(two_orbital_hops(t1=-1.0, t2=-0.4, cf=0.6, tab=0.0), 2, os.path.join(TMP, "orb0_hr.dat"))
    c = base_cfg(U=3.0, J=0.6, n=1.5, mesh=(32, 32, 1), basis="cf", structure="auto", T=0.003)
    s, r = run(c, hr2)
    S = s.classes["A"]["S"]
    check("auto stays diagonal when symmetry forbids mixing", (np.triu(S, 1) == 0).all() and r["ansatz_residual"] < 1e-9,
          f"S={S.tolist()}")


def test_qn_solver_matches_hybr():
    hops = two_orbital_hops(t1=-1.0, t2=-0.4, cf=0.6, tab=0.2)
    hr = model_hr(hops, 2, os.path.join(TMP, "orb_hr.dat"))
    out = {}
    for m in ("hybr", "qn"):
        c = base_cfg(U=3.0, J=0.6, n=1.5, mesh=(24, 24, 1), basis="cf", structure="auto", T=0.005)
        c["solver"] = {"method": m, "tol": 1e-11}
        s, r = run(c, hr)
        out[m] = (r["F_total"], np.sort(r["classes"]["A"]["Z_eigenvalues"]), r["residual_norm"])
    dF = abs(out["qn"][0] - out["hybr"][0]); dZ = np.abs(out["qn"][1] - out["hybr"][1]).max()
    check("quasi-Newton (Jacobian-reusing) solver == hybr", dF < 1e-10 and dZ < 1e-9 and out["qn"][2] < 1e-9,
          f"dF={dF:.1e} dZ={dZ:.1e}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"\nall tests passed in {time.time()-t0:.1f}s")
