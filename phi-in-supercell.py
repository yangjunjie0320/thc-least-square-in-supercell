import os, sys
import numpy, scipy
import scipy.linalg
import pyscf
from pyscf.pbc import gto

tmpdir = pyscf.lib.param.TMPDIR
stdout = sys.stdout

cell = pyscf.pbc.gto.Cell()
cell.atom  = 'He 2.0000 2.0000 2.0000; He 2.0000 2.0000 6.0000'
cell.basis = '321g'
cell.a = numpy.diag([4.0000, 4.0000, 8.0000])
cell.unit = 'bohr'
cell.verbose = 5
cell.stdout = stdout
cell.build()

nao = cell.nao_nr()

kmesh = numpy.asarray([4] * 3)
vk = kpts = cell.make_kpts(kmesh)
nk = len(vk)

from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
vr = rpts = translation_vectors_for_kmesh(cell, kmesh=kmesh, wrap_around=False)
nr = len(vr)

scell = pyscf.pbc.tools.pbc.super_cell(cell.copy(deep=True), kmesh, wrap_around=False)

gmesh = numpy.asarray([5] * 3)
ng = numpy.prod(gmesh)

coord0 = cell.gen_uniform_grids(gmesh, wrap_around=False)
coord1 = coord0[None, :] + vr[:, None]
coord1 = coord1.reshape(nr * ng, 3)

phi0 = scell.pbc_eval_gto('GTOval', coord0)
phi0 = phi0.reshape(ng, nr, nao)

phi = scell.pbc_eval_gto('GTOval', coord1)
zeta1 = numpy.einsum("Im,Jm,In,Jn->IJ", phi, phi, phi, phi, optimize=True)
zeta1 = zeta1.reshape(nr, ng, nr, ng)
print(f"zeta1: {zeta1.shape}")

phi = phi.reshape(nr, ng, nr, nao)
zeta2 = numpy.einsum("rIkm,sJkm,rIln,sJln->rIsJ", phi, phi, phi, phi, optimize=True)
assert abs(zeta1 - zeta2).max() < 1e-8
print(f"zeta2: {zeta2.shape}")

zeta3 = numpy.einsum("Ikm,sJkm,Iln,sJln->IsJ", phi0, phi, phi0, phi, optimize=True)
assert abs(zeta1[0] - zeta3).max() < 1e-8
print(f"zeta3: {zeta3.shape}")

theta = numpy.einsum('kx,rx->kr', vk, vr)
phase = numpy.exp(-1j * theta)

# phi_k_1 = numpy.einsum('grm,kr->kgm', phi0, phase.conj())
# phi_k_2 = numpy.einsum("rgsm,kr,ls->kglm", phi, phase, phase.conj()) / nr
zeta_k = numpy.einsum("rIsJ,kr,ls->kIlJ", zeta1, phase, phase.conj(), optimize=True) / nr

for k1k2 in range(nk * nk):
    k1, k2 = divmod(k1k2, nk)
    if (k1k2 + 1) % (nk * nk // 100) == 0:
        print(f"Progress: {(k1k2 + 1): 5d} / {nk * nk}")

    if k1 != k2:
        err = abs(zeta_k[k1, :, k2, :]).max()

        if err > 1e-8:
            print(f"Error: {k1} {k2} {err}")
            assert 1 == 2

    else:
        phi_k_1 = phi_k_2 = cell.pbc_eval_gto('GTOval', coord0, kpt=vk[k1])

        zeta_k_1 = zeta1[k1, :, k2, :]
        zeta_k_2 = numpy.einsum(
            "Im,Jm,In,Jn->IJ",
            phi_k_1.conj(), phi_k_1, 
            phi_k_1, phi_k_1.conj(),
            optimize=True
        )

print("All tests passed")

