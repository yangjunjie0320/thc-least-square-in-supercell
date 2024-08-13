import os, sys
import numpy, scipy
import scipy.linalg
import pyscf
from pyscf.pbc import gto

from opt_einsum import contract as einsum

tmpdir = pyscf.lib.param.TMPDIR
stdout = sys.stdout

cell = pyscf.pbc.gto.Cell()
cell.atom  = 'He 2.0000 2.0000 2.0000; He 2.0000 2.0000 6.0000'
cell.basis = '321g'
cell.a = numpy.diag([4.0000, 4.0000, 8.0000])
cell.unit = 'bohr'
cell.verbose = 5
cell.stdout = stdout
cell.ke_cutoff = 100.0
cell.build()

nao = cell.nao_nr()

kmesh = numpy.asarray([4] * 3)
vk = kpts = cell.make_kpts(kmesh)
nk = len(vk)

from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
vr = rpts = translation_vectors_for_kmesh(cell, kmesh=kmesh, wrap_around=False)
nr = len(vr)

scell = pyscf.pbc.tools.pbc.super_cell(cell.copy(deep=True), kmesh, wrap_around=False)

gmesh = numpy.asarray([10] * 3)
ng = numpy.prod(gmesh)

coord0 = cell.gen_uniform_grids(gmesh, wrap_around=False)
coord1 = coord0[None, :] + vr[:, None]
coord1 = coord1.reshape(nr * ng, 3)

phi0 = scell.pbc_eval_gto('GTOval', coord0)
phi0 = phi0.reshape(ng, nr, nao)

phi = scell.pbc_eval_gto('GTOval', coord1)
phi = phi.reshape(nr, ng, nr, nao)

phik = cell.pbc_eval_gto('GTOval', coord0, kpts=vk)
phik = numpy.asarray(phik).reshape(nk, ng, nao)

# rho = einsum("rgsm,rgtn->rgsmtn", phi, phi, optimize=True)
# assert rho.shape == (nr, ng, nr, nao, nr, nao)
# rho1 = rho[0]

rho2 = einsum("grm,gsn->grmsn", phi0, phi0, optimize=True)
assert rho2.shape == (ng, nr, nao, nr, nao)
rho = rho2

# err = abs(rho1 - rho2).max()
# assert err < 1e-10, err

rho_k = einsum("kgm,lgn->gkmln", phik.conj(), phik)
assert rho_k.shape == (ng, nk, nao, nk, nao)

# theta = numpy.dot(vr, vk.T)
# theta = theta.reshape(nr, nk)
# phase = numpy.exp(-1j * theta)
# rho3 = einsum("gkmln,rk,sl->grmsn", rho_k, phase.conj(), phase) / nr / nr
# rho3 = rho3.reshape(ng, nr, nao, nr, nao)

# zeta1 = einsum("grmsn,hrmsn->gh", rho3, rho3)
zeta1 = einsum("kgm,khm,lgn,lhn->gh", phik.conj(), phik, phik.conj(), phik) / nr / nr
assert abs(zeta1.imag).max() < 1e-10
zeta1 = zeta1.real

from pyscf.lib.scipy_helper import pivoted_cholesky
chol, perm, rank = pivoted_cholesky(zeta1)
mask = perm[:rank]

res = scipy.linalg.lstsq(zeta1[mask][:, mask], zeta1[mask, :])
z = res[0].T

rho_sol = einsum("gI,Irm,Isn->grmsn", z, phi0[mask], phi0[mask])
rho_ref = rho
err = abs(rho_sol - rho_ref).max()
assert err < 1e-6, err

rho_k_ref = rho_k
rho_k_sol = einsum("gI,kIm,lIn->gkmln", z, phik[:, mask].conj(), phik[:, mask])
err = abs(rho_k_sol - rho_k_ref).max()
assert err < 1e-6, err
print(err)

print("all test passed!")

