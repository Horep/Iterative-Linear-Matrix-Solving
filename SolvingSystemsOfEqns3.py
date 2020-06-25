import numpy as np
from DiagonalDominate import (ConvergencePotential,
                              SimpleDiagonalDominator,
                              DiagonalAbsSum)
from MatrixSolverMethods import Matrix_Solve_Jacobi, Matrix_Solve_Gauss_Seidel

m_list = [
          [86, 8, 17, 3, 2],
          [-1, 2, 15, 6, 1],
          [-20, 5, -80, 7, -150],
          [1, 10, 1, 1, 2],
          [-6, 3, 1, 123, 18]
          ]

A = np.array(m_list)

m_list = [
     [17],
     [3],
     [9],
     [1],
     [2]
     ]

B = np.array(m_list)


A, B = SimpleDiagonalDominator(A, B)
X = Matrix_Solve_Jacobi(A, B, 10**-5)
Y = Matrix_Solve_Gauss_Seidel(A, B, 10**-5)
