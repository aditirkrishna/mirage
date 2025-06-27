import numpy as np
from mirror_linear_regression.geometry import riemannian_mirror_step, spd_riemannian_step, project_to_spd

import unittest

class TestGeometry(unittest.TestCase):
    def test_riemannian_mirror_step(self):
        theta_t = np.array([0.4, 0.3, 0.3])
        grad = np.array([0.1, -0.2, 0.05])
        eta = 0.5
        theta_new = riemannian_mirror_step(grad, theta_t, eta)
        print("Riemannian mirror step result:", theta_new)
        self.assertEqual(theta_new.shape, theta_t.shape)
        self.assertTrue(np.all(theta_new >= 0))
        self.assertTrue(np.isclose(np.sum(theta_new), 1.0))

    def test_spd_riemannian_step(self):
        Sigma_t = np.array([[2.0, 0.5], [0.5, 1.0]])
        grad = np.array([[0.1, 0.0], [0.0, -0.1]])
        eta = 0.3
        Sigma_new = spd_riemannian_step(grad, Sigma_t, eta)
        print("SPD Riemannian step result:\n", Sigma_new)
        self.assertTrue(np.allclose(Sigma_new, Sigma_new.T))
        eigvals = np.linalg.eigvalsh(Sigma_new)
        self.assertTrue(np.all(eigvals > 0))

    def test_project_to_spd(self):
        A = np.array([[1.0, 2.0], [2.0, -1.0]])
        SPD = project_to_spd(A)
        print("Projected SPD matrix:\n", SPD)
        self.assertTrue(np.allclose(SPD, SPD.T))
        eigvals = np.linalg.eigvalsh(SPD)
        self.assertTrue(np.all(eigvals > 0))

if __name__ == "__main__":
    unittest.main()
