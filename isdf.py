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
c.mesh = [12] * 3
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
chol, perm, rank = pivoted_cholesky(zeta, tol=1e-20)
mask = perm[:rank]
res = scipy.linalg.lstsq(zeta[mask][:, mask], zeta[mask, :])

z = res[0].T
nip = z.shape[1]
assert z.shape == (ng, nip)

coul = []
for q, vq in enumerate(vk):
    from pyscf.pbc.tools import fft, ifft
    t12 = numpy.dot(coord, vq)
    f12 = numpy.exp(-1j * t12)
    z12_g = fft(einsum("gI,g->Ig", z, f12).reshape(-1, ng), gmesh)
    assert z12_g.shape == (nip, ng)

    v12_g  = z12_g * tools.get_coulG(c, k=vq, mesh=gmesh)
    v12_g *= c.vol / ng / ng
    assert v12_g.shape == (nip, ng)

    t34 = numpy.dot(coord, vq)
    f34 = numpy.exp(-1j * t34)
    z34_g = fft(einsum("gI,g->Ig", z, f34).reshape(-1, ng), gmesh)
    assert z34_g.shape == (nip, ng)

    coul.append(einsum("Im,In->mn", z12_g, z34_g.conj()))

def get_eri_1(k1, k2, k3, k4):
    q = kconserv2[k1, k2]

    x1 = phik[k1][mask]
    x2 = phik[k2][mask]
    x3 = phik[k3][mask]
    x4 = phik[k4][mask]

    eri = einsum("IJ,Im,In,Jk,Jl->mnkl", coul[q], x1.conj(), x2, x3.conj(), x4)
    # ao_pair_r_12 = einsum("gI,Im,In->gmn", z, x1.conj(), x2).reshape(ng, -1)
    # ao_pair_r_34 = einsum("gI,Im,In->gmn", z, x3.conj(), x4).reshape(ng, -1)
    # ao_pair_g_12 = einsum("Ig,Im,In->gmn", z12_g, x1.conj(), x2).reshape(ng, -1)
    # ao_pair_g_34 = einsum("Ig,Im,In->gmn", z34_g, x3.conj(), x4).reshape(ng, -1)
    res = {
        "eri": eri.reshape(nao * nao, nao * nao),
        # "ao_pair_r_12": ao_pair_r_12,
        # "ao_pair_r_34": ao_pair_r_34,
        # "ao_pair_g_12": ao_pair_g_12,
        # "ao_pair_g_34": ao_pair_g_34,
    }
    return res

def get_eri_2(k1, k2, k3, k4):
    vk1, vk2, vk3, vk4 = vk[k1], vk[k2], vk[k3], vk[k4]
    q = kconserv2[k1, k2]
    vq = vk[q]

    from pyscf.pbc.df.fft_ao2mo import get_ao_pairs_G
    z12_g  = get_ao_pairs_G(df, [vk1, vk2], vq, compact=False)
    v12_g  = tools.get_coulG(c, k=vq, mesh=gmesh)[:, None] * z12_g
    v12_g *= c.vol / ng / ng

    z34_g = get_ao_pairs_G(df, [-vk3, -vk4], vq, compact=False).conj()

    eri = einsum(
        "gx,gy->xy", v12_g, z34_g
    )
    # ao_pair_r_12 = einsum("gm,gn->gmn", phik[k1].conj(), phik[k2]).reshape(ng, -1)
    # ao_pair_r_34 = einsum("gm,gn->gmn", phik[k3].conj(), phik[k4]).reshape(ng, -1)
    # ao_pair_g_12 = get_ao_pairs_G(df, [vk1, vk2], vq, compact=False)
    # ao_pair_g_34 = get_ao_pairs_G(df, [vk3, vk4], vq, compact=False)

    res = {
        "eri": eri,
        # "ao_pair_r_12": ao_pair_r_12,
        # "ao_pair_r_34": ao_pair_r_34,
        # "ao_pair_g_12": ao_pair_g_12,
        # "ao_pair_g_34": ao_pair_g_34,
    }

    return res

for (k1, k2, k3) in itertools.product(range(nk), repeat=3):
    vk1, vk2, vk3 = vk[k1], vk[k2], vk[k3]
    k4 = kconserv3[k1, k2, k3]
    vk4 = vk[k4]

    from pyscf.pbc.df.df_ao2mo import _iskconserv
    from pyscf.pbc.df.df_ao2mo import _format_kpts
    vk1234 = numpy.asarray([vk1, vk2, vk3, vk4])
    assert _iskconserv(c, vk1234)

    sol = get_eri_1(k1, k2, k3, k4)
    ref = get_eri_2(k1, k2, k3, k4)

    err_list = []
    # err_list.append(abs(sol["ao_pair_r_12"] - ref["ao_pair_r_12"]).max())
    # err_list.append(abs(sol["ao_pair_r_34"] - ref["ao_pair_r_34"]).max())
    # err_list.append(abs(sol["ao_pair_g_12"] - ref["ao_pair_g_12"]).max())
    # err_list.append(abs(sol["ao_pair_g_34"] - ref["ao_pair_g_34"]).max())
    err_list.append(abs(sol["eri"] - ref["eri"]).max())

    info = ", ".join("% 8.2e" % x for x in err_list)
    print(f"{k1 = :4d}, {k2 = :4d}, {k3 = :4d}, {k4 = :4d}, {info = :s}")

    # numpy.savetxt(sys.stdout, sol["eri"].real, fmt="% 8.2e", header="sol[eri]", delimiter=", ")
    # numpy.savetxt(sys.stdout, ref["eri"].real, fmt="% 8.2e", header="ref[eri]", delimiter=", ")

print("all tests passed")
