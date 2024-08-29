# pyabacus.hsolver

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Union, Callable

from .._core import hsolver

# class diag_comm_info:
#     def __init__(self, rank: int, nproc: int) -> None: ...
    
#     @property
#     def rank(self) -> int: ...
    
#     @property
#     def nproc(self) -> int: ...
    
def dav_subspace(
    mm_op: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    init_v: NDArray[np.complex128],
    dim: int,
    num_eigs: int,
    pre_condition: NDArray[np.float64],
    dav_ndim: int = 2,
    tol: float = 1e-2,
    max_iter: int = 1000,
    need_subspace: bool = False,
    is_occupied: Union[List[bool], None] = None,
    scf_type: bool = False,
    need_mpi: bool = False
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """ A function to diagonalize a matrix using the Davidson-Subspace method.

    Parameters
    ----------
    mm_op : Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
        The operator to be diagonalized, which is a function that takes a matrix as input
        and returns a matrix mv_op(X) = H * X as output.
    init_v : NDArray[np.complex128]
        The initial guess for the eigenvectors.
    dim : int
        The number of basis, i.e. the number of rows/columns in the matrix.
    num_eigs : int
        The number of bands to calculate, i.e. the number of eigenvalues to calculate.
    pre_condition : NDArray[np.float64]
        The preconditioner.
    dav_ndim : int, optional
        The number of vectors in the subspace, by default 2.
    tol : float, optional
        The tolerance for the convergence, by default 1e-2.
    max_iter : int, optional    
        The maximum number of iterations, by default 1000.
    need_subspace : bool, optional
        Whether to use subspace function, by default False.
    is_occupied : List[bool] | None, optional
        The list of occupied bands, by default None. This indicates how many eigenvalues 
        need to be calculated, starting from the smallest eigenvalue. Only the energy levels 
        occupied by electrons (occupied) need to be calculated.
    scf_type : bool, optional
        Indicates whether the calculation is a self-consistent field (SCF) calculation. 
        If True, the initial precision of eigenvalue calculation can be coarse. 
        If False, it indicates a non-self-consistent field (non-SCF) calculation, 
        where high precision in eigenvalue calculation is required from the start.
    need_mpi : bool, optional
        Whether to use MPI, by default False.  
    
    Returns
    -------
    e : NDArray[np.float64]
        The eigenvalues.
    v : NDArray[np.complex128]
        The eigenvectors corresponding to the eigenvalues.
    """
    if not callable(mm_op):
        raise TypeError("mm_op must be a callable object.")
    
    if is_occupied is None:
        is_occupied = [True] * num_eigs
    
    if init_v.ndim != 1 or init_v.dtype != np.complex128:
        init_v = init_v.flatten().astype(np.complex128, order='C')
    
    _diago_obj_dav_subspace = hsolver.diago_dav_subspace(dim, num_eigs, need_mpi)
    _diago_obj_dav_subspace.set_psi(init_v)
    _diago_obj_dav_subspace.init_eigenvalue()
    
    nproc = _diago_obj_dav_subspace.get_nproc()
    assert dav_ndim > 1, "dav_ndim must be greater than 1."
    assert dav_ndim * num_eigs < dim * nproc, "dav_ndim * num_eigs must be less than dim * comm_info.nproc."
   
    _ = _diago_obj_dav_subspace.diag(
        mm_op,
        pre_condition,
        dav_ndim,
        tol,
        max_iter,
        need_subspace,
        is_occupied,
        scf_type
    )
    
    e = _diago_obj_dav_subspace.get_eigenvalue()
    v = _diago_obj_dav_subspace.get_psi()
    
    return e, v
    