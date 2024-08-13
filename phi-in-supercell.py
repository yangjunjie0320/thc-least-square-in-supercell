import numpy, scipy
import scipy.linalg
import pyscf
from pyscf.pbc import gto

cell = pyscf.pbc.gto.Cell()
cell.atom  = 'He 2.0000 2.0000 2.0000; He 2.0000 2.0000 6.0000'
cell.basis = '321g'
cell.a = numpy.diag([4.0000, 4.0000, 8.0000])
cell.unit = 'bohr'
cell.build()

nao = cell.nao_nr()

mesh = numpy.asarray([2] * 3)
vk = kpts = cell.make_kpts(mesh)
nk = len(vk)

from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
vr = rpts = translation_vectors_for_kmesh(cell, kmesh=mesh, wrap_around=False)
nr = len(vr)

scell = pyscf.pbc.tools.pbc.super_cell(cell.copy(deep=True), mesh, wrap_around=False)

# ng, 3
gmesh = numpy.asarray([5] * 3)
ng = numpy.prod(gmesh)

coord0 = cell.gen_uniform_grids(gmesh, wrap_around=False)
coord1 = coord0[None, :] + vr[:, None]
coord1 = coord1.reshape(nr * ng, 3)

phi0 = scell.pbc_eval_gto('GTOval', coord0)
phi0 = phi0.reshape(ng, nr, nao)

phi = scell.pbc_eval_gto('GTOval', coord1)
phi = phi.reshape(nr, ng, nr, nao) # , nr)

# for ir in range(nr):
#     for jr in range(nr):
#         # jr = ir
#         # ind = 0
#         vrd = vr[jr] - vr[ir]
#         nn = numpy.linalg.norm(vrd - vr, axis=1)
#         ind = nn.argmin()
#         if not nn[ind] < 1e-8:
#             continue

#         phi1 = phi[ir, :, jr, :]
#         phi2 = phi0[:, ind, :]

#         err = abs(phi1 - phi2).max()
#         assert err < 1e-8, err

theta = numpy.einsum('kx,rx->kr', vk, vr)
phase = numpy.exp(-1j * theta)

# phi_k_1 = numpy.einsum('grm,kr->gkm', phi, phase)
phi_k_2 = numpy.einsum("rgsm,kr,ls->kglm", phi, phase, phase) / nr

for k1 in range(nk):
    for k2 in range(nk):
        phi2 = phi_k_2[k1, :, k2, :]

        if k1 != k2:
            err = abs(phi2).max()
            # print(k1, k2, phi2)
            assert abs(phi2).max() < 1e-8, err

        else:
            phi_ = cell.pbc_eval_gto('GTOval', coord0, kpt=vk[k1])
            phi_ = numpy.asarray(phi_).real

            # print("\nphi")
            # print(phi_[:10])

            # print("\nphi")
            # print(phi2[:10].real)
            err = abs(phi_ - phi2).max()
            assert err < 1e-8, err