import itertools, os, sys
from itertools import product

import numpy, scipy
from opt_einsum import contract as einsum

import pyscf
from pyscf.pbc import tools
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

c = pbcgto.Cell()
c.atom = """
He 1.000000 1.000000 2.000000
He 1.000000 1.000000 4.000000
"""
c.a = numpy.diag([2.0, 2.0, 6.0])
c.basis = "sto3g"
c.verbose = 0
# c.mesh = [10] * 3   
c.mesh = [15] * 3
c.build()

from pyscf.pbc.lib.kpts_helper import get_kconserv
vk = c.make_kpts([4, 4, 4], wrap_around=False)
kconserv3 = get_kconserv(c, vk)
kconserv2 = kconserv3[:, :, 0].T

nk = vk.shape[0]
nq = len(numpy.unique(kconserv2))
assert nk == nq

df = pyscf.pbc.df.FFTDF(c)
gmesh = c.mesh
coord = c.gen_uniform_grids(mesh=gmesh, wrap_around=False)
phik  = pbcdft.numint.eval_ao_kpts(c, coord, kpts=vk, deriv=0)
phik  = numpy.asarray(phik)
ng, nao = phik.shape[1:]

zeta = einsum("kgm,khm,lgn,lhn->gh", phik.conj(), phik, phik.conj(), phik) / nk / nk
assert abs(zeta.imag).max() < 1e-10
zeta = zeta.real

from pyscf.lib.scipy_helper import pivoted_cholesky
chol, perm, rank = pivoted_cholesky(zeta)
mask = perm[:rank]
res = scipy.linalg.lstsq(zeta[mask][:, mask], zeta[mask, :])

z = res[0].T
nip = z.shape[1]
assert z.shape == (ng, nip)
print(f"{z.shape = }, {nip = }")

def get_eri_1(k1, k2, k3, k4):
    vk1, vk2, vk3, vk4 = vk[k1], vk[k2], vk[k3], vk[k4]
    q = kconserv2[k1, k2]
    vq = vk[q]

    from pyscf.pbc.tools import fft, ifft
    t12 = numpy.dot(coord, vq)
    f12 = numpy.exp(-1j * t12)
    z12_g = fft(einsum("gI,g->Ig", z, f12).reshape(-1, ng), gmesh)
    assert z12_g.shape == (nip, ng)

    v12_g  = z12_g * tools.get_coulG(c, k=vq, mesh=gmesh) * (c.vol / ng)
    assert v12_g.shape == (nip, ng)

    t34 = numpy.dot(coord, vq)
    f34 = numpy.exp(1j * t34)
    z34_g = ifft(einsum("gI,g->Ig", z, f34).reshape(-1, ng), gmesh)
    assert z34_g.shape == (nip, ng)
    coul_q = einsum("Ig,Jg->IJ", v12_g, z34_g)

    x1, x2, x3, x4 = pbcdft.numint.eval_ao_kpts(c, coord[mask], kpts=[vk1, vk2, vk3, vk4])

    eri = einsum("IJ,Im,In,Jk,Jl->mnkl", coul_q, x1.conj(), x2, x3, x4.conj())
    return eri.reshape(nao * nao, nao * nao)

def get_eri_2(k1, k2, k3, k4):
    vk1, vk2, vk3, vk4 = vk[k1], vk[k2], vk[k3], vk[k4]
    q = kconserv2[k1, k2]
    vq = vk[q]
    # assert q == kconserv2[k3, k4]

    from pyscf.pbc.tools import fft, ifft
    t12 = numpy.dot(coord, vq)
    f12 = numpy.exp(-1j * t12)

    from pyscf.pbc.df.fft_ao2mo import get_ao_pairs_G
    z12_g  = get_ao_pairs_G(df, [vk1, vk2], vq, compact=False)
    v12_g  = tools.get_coulG(c, k=vq, mesh=gmesh)[:, None] * z12_g
    v12_g *= c.vol / ng / ng

    z34_g = get_ao_pairs_G(df, [-vk3, -vk4], vq, compact=False).conj()

    eri = einsum(
        "gx,gy->xy", v12_g, z34_g
    )
    return eri

for (k1, k2, k3) in itertools.product(range(nk), repeat=3):
    vk1, vk2, vk3 = vk[k1], vk[k2], vk[k3]

    q = kconserv2[k1, k2]
    vq = vk[q]
    k4 = kconserv3[k1, k2, k3]
    vk4 = vk[k4]

    from pyscf.pbc.df.df_ao2mo import _iskconserv
    from pyscf.pbc.df.df_ao2mo import _format_kpts
    vk1234 = numpy.asarray([vk1, vk2, vk3, vk4])
    assert _iskconserv(c, vk1234)

    eri_sol = get_eri_1(k1, k2, k3, k4)
    eri_ref = get_eri_2(k1, k2, k3, k4)

    err1 = abs(eri_sol.real - eri_ref.real).max()
    err2 = abs(eri_sol.imag - eri_ref.imag).max()

    print(f"k1 = {k1:4d}, k2 = {k2:4d}, k3 = {k3:4d}, k4 = {k4:4d}, err = {err1 + err2:6.2e}")
    if not err1 + err2 < 1e-4:
        print("vk = ")
        numpy.savetxt(sys.stdout, numpy.asarray(vk1234), fmt="% 8.4f", delimiter=", ")

        print("vq = ")
        numpy.savetxt(sys.stdout, vq, fmt="% 8.4f", delimiter=", ")
        
        print(f"err1 = {err1:6.2e}, err2 = {err2:6.2e}")
        raise RuntimeError("eri_err too large")
