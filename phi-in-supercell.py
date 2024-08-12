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

gmesh = numpy.asarray([10] * 3)
ng = numpy.prod(gmesh)

coord0 = cell.gen_uniform_grids(gmesh, wrap_around=False)
coord1 = coord0[None, :] + vr[:, None]
coord1 = coord1.reshape(nr * ng, 3)

phi0 = scell.pbc_eval_gto('GTOval', coord0)
phi0 = phi0.reshape(ng, nr, nao)

phi = scell.pbc_eval_gto('GTOval', coord1)
zeta1 = numpy.einsum("Im,Jm,In,Jn->IJ", phi, phi, phi, phi, optimize=True)
zeta1 = zeta1.reshape(nr, ng, nr, ng)
rho1 = numpy.einsum("gm,gn->gmn", phi, phi, optimize=True)
rho1 = rho1.reshape(nr, ng, nr, nao, nr, nao)
print(f"zeta1: {zeta1.shape}")
print(f"rho1: {rho1.shape}")

phi = phi.reshape(nr, ng, nr, nao)
rho2 = numpy.einsum("rgsm,rgtn->rgsmtn", phi, phi, optimize=True)
assert rho2.shape == (nr, ng, nr, nao, nr, nao)

rho3 = numpy.einsum("grm,gsn->grmsn", phi0, phi0, optimize=True)
assert rho3.shape == (ng, nr, nao, nr, nao)
rhs = numpy.einsum("grm,grm,hrm,hrm->gh", phi0, phi0, phi0, phi0, optimize=True)

from pyscf.lib.scipy_helper import pivoted_cholesky
chol, perm, rank = pivoted_cholesky(rhs)
perm = perm[:rank]
zeta = rhs[:, perm] @ numpy.linalg.pinv(rhs[perm][:, perm])
print(perm)

rho4 = numpy.einsum("gI,Irm,Isn->grmsn", zeta, phi0[perm], phi0[perm], optimize=True)
assert rho4.shape == (ng, nr, nao, nr, nao)
err = abs(rho3 - rho4).max()
assert err < 1e-10, err
assert 1 == 2

err = abs(rho3 - rho1[0]).max()
assert err < 1e-10

err = abs(rho3 - rho2[0]).max()
assert err < 1e-10
print("All tests passed")
