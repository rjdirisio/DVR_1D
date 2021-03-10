import numpy as np
import os, sys

class AnalyzeDVR:
    def __init__(self,
                 res_file):
        """
        :param res_file: Full path to .npz file that houses DVR results
        """
        if '.npz' not in res_file:
            res_file = res_file + '.npz'
        self.dvr_npz = np.load(res_file)

    def get_wfns(self):
        return self.dvr_npz['wfns']

    def get_grid(self):
        return self.dvr_npz['grid']

    def get_energies(self):
        return self.dvr_npz['energies']

    def get_potential(self):
        return self.dvr_npz['potential']

    @staticmethod
    def exp_val(attr, wfn, quanta):
        return np.dot(wfn[:, quanta], (attr * wfn[:, quanta]))

    @staticmethod
    def std_dev(grd, wfn, quanta):
        av_x = self.exp_val(grd, wfn, quanta)
        av_x2 = self.exp_val(grd ** 2, wfn, quanta)
        return np.sqrt(av_x2 - av_x ** 2)

    @staticmethod
    def calc_re(grd, wfn, quanta):
        idx = np.argmax(wfn[:, quanta])
        return grd[idx]
