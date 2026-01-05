import numpy as np
from GenericTest import GenericTest

import porespy as ps

ps.visualization.set_mpl_style()


ps.settings.tqdm["disable"] = True


class IBOPTest(GenericTest):
    def setup_class(self):
        self.im2D = ps.generators.blobs(
            shape=[300, 150],
            porosity=0.6,
            seed=0,
            periodic=False,
        )

    def test_ibop_w_and_wo_pc(self):
        # bins must be none to ensure they both use same bins (i.e. all of them)
        r1 = ps.simulations.drainage(im=self.im2D, steps=None)
        assert np.sum(r1.im_seq == -1) == 0
        pc = ps.filters.capillary_transform(im=self.im2D)
        r2 = ps.simulations.drainage(im=self.im2D, pc=pc, steps=None)
        assert np.all(r1.im_seq == r2.im_seq)

    def test_ibop_w_trapping(self):
        inlets = ps.generators.faces(shape=self.im2D.shape, inlet=0)
        r1 = ps.simulations.drainage(im=self.im2D, inlets=inlets, steps=None)
        outlets = ps.generators.faces(shape=self.im2D.shape, outlet=0)
        r2 = ps.simulations.drainage(
            im=self.im2D, inlets=inlets, outlets=outlets, steps=None
        )
        assert np.sum(r1.im_seq == -1) == 0
        assert np.sum(r2.im_seq == -1) == 7170
        temp = ps.filters.fill_invalid_pores(self.im2D)
        r3 = ps.simulations.drainage(im=temp, inlets=inlets, steps=None)
        assert np.sum(r3.im_seq == -1) == 0

    def test_ibop_w_residual(self):
        rs = ps.filters.local_thickness(self.im2D) > 20
        inlets = ps.generators.faces(shape=self.im2D.shape, inlet=0)
        r1 = ps.simulations.drainage(im=self.im2D, inlets=inlets, residual=rs, steps=None)
        # Ensure all residual voxels have a sequence of 0 (invaded before first step)
        assert np.all(r1.im_seq[rs] == 0)


if __name__ == "__main__":
    self = IBOPTest()
    self.run_all()
