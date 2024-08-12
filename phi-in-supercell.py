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

gmesh = numpy.asarray([5] * 3)
ng = numpy.prod(gmesh)
coord0 = cell.gen_uniform_grids(gmesh, wrap_around=False)

phik = cell.pbc_eval_gto('GTOval', coord0, kpts=vk)
phik = numpy.asarray(phik)
assert phik.shape == (nk, ng, nao)

rhs = numpy.einsum("kgm,khm->gh", phik.conj(), phik, optimize=True) / cell.vol * 2
rhs = rhs.conj() * rhs
rhs = rhs.real

from pyscf.lib.scipy_helper import pivoted_cholesky
chol, perm, rank = pivoted_cholesky(rhs)
perm = perm[:rank]
zeta = rhs[:, perm] @ numpy.linalg.pinv(rhs[perm][:, perm])
print(perm)

for k1k2 in range
