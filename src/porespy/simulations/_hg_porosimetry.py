from porespy.filters import capillary_transform
from porespy.generators import borders
from porespy.metrics import pc_map_to_pc_curve
from porespy.simulations import drainage, imbibition
from porespy.tools import Results

__all__ = [
    'hg_porosimetry',
]


def hg_porosimetry(im, steps=25, voxel_size=1.0):
    faces = borders(im.shape, mode='faces')
    pc = capillary_transform(
        im=im,
        sigma=0.465,
        theta=140,
        voxel_size=voxel_size,
    )

    drn = drainage(im=im, pc=pc, inlets=faces, steps=steps)
    imb = imbibition(im=im, pc=pc, outlets=faces, steps=steps)

    res = Results()
    pc, snwp = pc_map_to_pc_curve(
        im=im,
        pc=drn.im_pc,
        seq=drn.im_seq,
        mode='drainage',
        fix_ends=True,
    )
    res.pc_intrusion = pc
    res.snwp_intrusion = snwp

    pc, snwp = pc_map_to_pc_curve(
        im=im,
        pc=imb.im_pc,
        seq=imb.im_seq,
        mode='imbibition',
        fix_ends=True,
    )
    res.pc_extrusion = pc
    res.snwp_extrusion = snwp

    return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import porespy as ps

    i = 50591
    im = ps.generators.blobs([100, 100, 100], porosity=0.4, seed=0)
    mip = hg_porosimetry(im, voxel_size=1e-5, steps=50)

    fig, ax = plt.subplots()
    ax.semilogx(mip.pc_intrusion, mip.snwp_intrusion, 'b.-')
    ax.semilogx(mip.pc_extrusion, mip.snwp_extrusion, 'r.-')
    ax.set_xlim([1_000, 100_000])
    ax.set_ylim([0, 1.05])
