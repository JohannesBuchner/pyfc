__author__ = 'ayw'

import unittest
import numpy as np
import pyFC


class FCCase(unittest.TestCase):

    def setUp(self):

        # Standard test case for Lognormal fractal cube
        self.ni = self.nj = self.nk = 64
        self.mu = 1.
        self.sigma = np.sqrt(5.)
        self.kmin = 1.
        self.beta = 5. / 3.

        # Standard test case for Gaussian fractal cube
        self.mu_g = 0.
        self.sigma_g = 200.

        # Fractal cube objects
        self.lfc = pyFC.LogNormalFractalCube(self.ni, self.nj, self.nk, self.kmin, self.mu, self.sigma, self.beta)
        self.gfc = pyFC.GaussianFractalCube(self.ni, self.nj, self.nk, self.kmin, self.mu_g, self.sigma_g, self.beta)

        # Saved, valid fractal cube data

        self.rlc = np.fromfile('test_lfc_1_64x64x64.dbl').reshape((ni, nj, nk))
        #self.rgc = np.fromfile('test_gfc_1_64x64x64.dbl').reshape((ni, nj, nk))

    def test_write_cube(self):
        self.assertTrue(np.all_close())

    def test_read_cube(self):
        #rrc = lfc.read_cube('test_lnfc_1_64x64x64.dbl')
        rrc = np.ones((ni, nj, nk)) # False test
        self.assertTrue(np.all_close(rlc, rrc))


class LFCCase(FCCase):


    def test_lfc_gencube(self):
        self.assertEqual(True, True)


class GFCCase(FCCase):


    def test_lfc_gencube(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()

