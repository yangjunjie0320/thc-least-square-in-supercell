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

gmesh = numpy.asarray([11] * 3)
ng = numpy.prod(gmesh)

coord0 = cell.gen_uniform_grids(gmesh, wrap_around=False)
coord1 = coord0[None, :] + vr[:, None]
coord1 = coord1.reshape(nr * ng, 3)

phi0 = scell.pbc_eval_gto('GTOval', coord0)
phi0 = phi0.reshape(ng, nr, nao)

phi = scell.pbc_eval_gto('GTOval', coord1)
phi = phi.reshape(nr, ng, nr, nao)

theta = numpy.einsum('kx,rx->kr', vk, vr)
phase = numpy.exp(-1j * theta)

phi_k_1 = numpy.einsum('grm,kr->kgm', phi0, phase.conj())
phi_k_2 = numpy.einsum("rgsm,kr,ls->kglm", phi, phase, phase.conj()) / nr

err_info = []

for k1k2 in range(nk * nk):
    k1, k2 = divmod(k1k2, nk)
    phi2 = phi_k_2[k1, :, k2, :]

    if (k1k2 + 1) % (nk * nk // 10) == 0:

        print(f"Progress: {(k1k2 + 1): 5d} / {nk * nk}")

    if k1 != k2:
        err = abs(phi2).max()
        
        if not abs(phi2).max() < 1e-8:
            print(f"err = {err:6.2e}, k1 = {k1}, k2 = {k2}")
            print(phi2[:10, :])
            assert 1 == 2

    else:
        phi_ = cell.pbc_eval_gto('GTOval', coord0, kpt=vk[k1])
        phi_ = numpy.asarray(phi_)

        err = abs(phi_ - phi2).max()
        if not err < 1e-8:
            print(f"err = {err:6.2e}, k1 = {k1}, k2 = {k2}")
            print(phi_[:10, :])
            print(phi2[:10, :])
            assert 1 == 2
            
        err = abs(phi_k_1[k1, :] - phi2).max()
        if not err < 1e-8:
            print(f"err = {err:6.2e}, k1 = {k1}, k2 = {k2}")
            print(phi_k_1[k1, :10, :])
            print(phi2[:10, :])
            assert 1 == 2

print("All tests passed")

