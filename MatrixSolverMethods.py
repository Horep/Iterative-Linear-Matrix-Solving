import numpy as np


#  Checks if matrix is diagonally dominant
def dd(X):
    D = np.diag(np.abs(X))
    S = np.sum(np.abs(X), axis=1) - D
    if np.all(D > S):
        return True
    else:
        return False


def Residue(A, B, x):
    return np.amax(abs(A.dot(x) - B))


#  Solves for x in Ax=B via the Jacobi method
def Matrix_Solve_Jacobi(A, B, tol):
    Max_Iter = 1000
    iterations = 0
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D = np.diagflat(np.diag(A))
    D_inv = np.linalg.inv(D)
    P = -D_inv.dot(L+U)
    q = D_inv.dot(B)

    if dd(A) is True:
        print("Matrix A's diagonals dominate, convergence guaranteed.")
    else:
        eigenvalues = np.linalg.eig(P)[0]
        SpecRad = np.amax(abs(eigenvalues))
        if SpecRad < 1:
            print(
                f"Spectral radius of P={SpecRad} < 1, convergence guaranteed."
                 )
        else:
            print(
                f"Spectral radius of P={SpecRad}>1"
                + ", convergence not guaranteed."
                )

    x_n = np.zeros_like(B)
    residue = Residue(A, B, x_n)
    while residue > tol and iterations < Max_Iter:
        x_n = P.dot(x_n) + q
        iterations += 1
        residue = Residue(A, B, x_n)
    if iterations == Max_Iter:
        print("Max iterations reached.")
    else:
        print(f"Finished in {iterations} iterations.")
    return x_n


#  Solves for x in Ax=B via the Jacobi method
def Matrix_Solve_Gauss_Seidel(A, B, tol):
    Max_Iter = 1000
    iterations = 0
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D = np.diagflat(np.diag(A))
    LD_inv = np.linalg.inv(D+L)
    P = -LD_inv.dot(U)
    q = LD_inv.dot(B)

    if dd(A) is True:
        print("Matrix A's diagonals dominate, convergence guaranteed.")
    else:
        eigenvalues = np.linalg.eig(P)[0]
        SpecRad = np.amax(abs(eigenvalues))
        if SpecRad < 1:
            print(
                f"Spectral radius of P={SpecRad} < 1, convergence guaranteed."
                 )
        else:
            print(
                f"Spectral radius of P={SpecRad}>1"
                + ", convergence not guaranteed."
                )

    x_n = np.zeros_like(B)
    residue = Residue(A, B, x_n)
    while residue > tol and iterations < Max_Iter:
        x_n = P.dot(x_n) + q
        iterations += 1
        residue = Residue(A, B, x_n)
    if iterations == Max_Iter:
        print("Max iterations reached.")
    else:
        print(f"Finished in {iterations} iterations.")
    return x_n