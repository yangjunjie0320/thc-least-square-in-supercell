import numpy, scipy
import pyscf
from pyscf.pbc import gto

cell = pyscf.pbc.gto.Cell()
cell.atom  = 'He 2.0000 2.0000 2.0000; He 2.0000 2.0000 6.0000'
cell.basis = 'sto3g'
cell.a = numpy.diag([4.0000, 4.0000, 8.0000])
cell.unit = 'bohr'
cell.build()

nao = cell.nao_nr()

kmesh = [4, 4, 2]
vk = kpts = cell.make_kpts(kmesh)
nk = len(kpts)

for xx in [29, 30]:
    coord = cell.gen_uniform_grids([xx, ] * 3)
    ng = len(coord)
    weigh = numpy.ones(ng) / ng * cell.vol

    k1 = 0
    k2 = 4
    print(f"k1: {k1}, k2: {k2}")
    vk1, vk2 = vk[k1], vk[k2]
    print(f"vk1: {vk1}")
    print(f"vk2: {vk2}")
    xks = cell.pbc_eval_gto('GTOval', coord, kpts=[vk1, vk2])
    xk1, xk2 = numpy.einsum('g,kgm->kgm', numpy.sqrt(weigh), xks)
    assert xk1.shape == (ng, nao)
    assert xk2.shape == (ng, nao)

    # k1 ovlp
    ovlp_sol = numpy.einsum("gm,gn->mn", xk1.conj(), xk1, optimize=True)
    ovlp_ref = cell.pbc_intor('int1e_ovlp', kpts=vk1)
    assert abs(ovlp_sol - ovlp_ref).max() < 1e-4, f"Error: {abs(ovlp_sol - ovlp_ref).max()}, {xx = }"

    # k2 ovlp
    ovlp_sol = numpy.einsum("gm,gn->mn", xk2.conj(), xk2, optimize=True)
    ovlp_ref = cell.pbc_intor('int1e_ovlp', kpts=vk2)
    assert abs(ovlp_sol - ovlp_ref).max() < 1e-4, f"Error: {abs(ovlp_sol - ovlp_ref).max()}, {xx = }"

    ovlp_sol = numpy.einsum("gm,gn->mn", xk1.conj(), xk2, optimize=True)
    ovlp_ref = ovlp_sol * 0.0
    assert abs(ovlp_sol - ovlp_ref).max() < 1e-4, f"Error: {abs(ovlp_sol - ovlp_ref).max()}, {xx = }"

print("All tests passed.")
