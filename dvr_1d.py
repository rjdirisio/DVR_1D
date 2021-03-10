import numpy as np
import numpy.linalg as la


class DVR_1D:
    def __init__(self,
                 grid,
                 potential,
                 mass,
                 res_flie='dvr_results',
                 outputPath="dvr_results/"):
        """
        A class to do a generic 1D DVR based on the Colbert and Miller paper:
        https://aip.scitation.org/doi/10.1063/1.462100
        :param grid: 1D Array. The gridpoints that come from diagonalizing the coordinate operator
        :type np.ndarray
        :param potential: 1D Array. The potential already evaluated at each gridpoint.
        :type np.ndarray
        :param res_flie: What we are calling this DVR, will end up in outputFolder
        :type str
        :param outputFolder: the folder where our results will be stored. Always downstream
        :type str
        """
        self.grid = grid
        self.pot = potential
        self.mass = mass
        self.res_file = res_flie
        self.res_dir = outputPath
        self.dx = self.grid[1] - self.grid[0]

    def get_potential(self):
        """The potential is placed along a diagonal."""
        return np.diag(self.pot)

    def get_kinetic(self):
        """
        Not-so-elegant construction of the kinetic energy matrix.
        This is based on analytic expressions for the C&M Paper
        """
        grid_size = self.grid.shape[0]
        ke = np.zeros((grid_size, grid_size))
        t_coef = 1 / (2 * self.mass * (self.dx ** 2))
        for i in range(1, grid_size):
            for j in range(i):
                ke[i, j] = t_coef * (-1.) ** (i - j) * (2. / ((i - j) ** 2))
        dgVals = t_coef * (np.pi * np.pi / 3.)
        s = ke + ke.T
        np.fill_diagonal(s, dgVals)
        return s

    def digonalize_ham(self):
        V = self.get_potential()
        T = self.get_kinetic()
        energies, wfns = la.eigh(T + V)
        return energies, wfns

    def run(self):
        evals, evecs = self.digonalize_ham()
        if not os.path.isdir(self.res_dir):
            os.makedirs(self.res_dir)

        np.savez(self.res_dir + self.res_file,
                 grid=self.grid,
                 potential=self.pot,
                 energies=evals,
                 wfns=evecs
                 )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from analyze_dvr import *
    from pyvibdmc.simulation_utilities import Constants

    grid = np.arange(-50, 50, 0.1)
    omega = 3000
    mass = 1
    potential = 0.5 * mass * Constants.convert(omega, 'wavenumbers', to_AU=True) ** 2 * grid ** 2
    test = DVR_1D(grid=grid,
                  potential=potential,
                  mass=mass)
    test.run()
    test_dvr = AnalyzeDVR('dvr_results/dvr_results.npz')
    wfns = test_dvr.get_wfns()
    grd = test_dvr.get_grid()
    pot = test_dvr.get_potential()
    plt.plot(grd, wfns[:,0])
    plt.show()
    exp_x = test_dvr.exp_val(grd, wfn=wfns, quanta=0)
    std = test_dvr.std_dev(grd,wfns,0)
    print(exp_x, std)