from firedrake.petsc import PETSc
from firedrake.ensemble.ensemble_functionspace import (
    EnsembleFunctionSpace, EnsembleDualSpace)
from firedrake.adjoint_utils import EnsembleFunctionMixin
from firedrake.function import Function
from firedrake.norms import norm
from pyop2 import MixedDat

from functools import cached_property
from contextlib import contextmanager

__all__ = ("EnsembleFunction", "EnsembleCofunction")


class EnsembleFunctionBase(EnsembleFunctionMixin):
    """
    A mixed (co)function defined on a :class:`firedrake.Ensemble`.
    The subcomponents are distributed over the ensemble members, and
    are specified locally in a :class:`firedrake.EnsembleFunctionSpace`.

    Parameters
    ----------

    function_space : :class:`firedrake.EnsembleFunctionSpace`.
        The function space of the (co)function.

    Notes
    -----
    Passing a :class:`firedrake.EnsembleDualSpace` to :class:`firedrake.EnsembleFunction`
    will return an instance of :class:`firedrake.EnsembleCofunction`.

    This class does not carry UFL symbolic information, unlike a
    :class:`firedrake.Function`. UFL expressions can only be defined
    locally on each ensemble member using a `firedrake.Function`
    from `EnsembleFunction.subfunctions`.

    See also:
    - Primal ensemble objects: :class:`firedrake.EnsembleFunctionSpace` and :class:`firedrake.EnsembleFunction`.
    - Dual ensemble objects: :class:`firedrake.EnsembleDualSpace` and :class:`firedrake.EnsembleCofunction`.
    """

    @PETSc.Log.EventDecorator()
    @EnsembleFunctionMixin._ad_annotate_init
    def __init__(self, function_space):
        self._fs = function_space

        self._fbuf = Function(function_space._full_local_space)

        # create a Vec containing the data for all functions on all
        # ensemble members. Because we use the Vec of each local mixed
        # function as the storage, if the data in the Function Vec
        # is valid then the data in the EnsembleFunction Vec is valid.

        with self._fbuf.dat.vec as fvec:
            n = function_space.nlocal_rank_dofs
            N = function_space.nglobal_dofs
            sizes = (n, N)
            self._vec = PETSc.Vec().createWithArray(
                fvec.array, size=sizes,
                comm=function_space.global_comm)
            self._vec.setFromOptions()

    def function_space(self):
        return self._fs

    @cached_property
    def subfunctions(self):
        """
        The (co)functions on the local ensemble member.
        """
        def local_function(i):
            V = self._fs.local_spaces[i]
            usubs = self._subcomponents(i)
            if len(usubs) == 1:
                dat = usubs[0].dat
            else:
                dat = MixedDat((u.dat for u in usubs))
            return Function(V, val=dat)

        return tuple(local_function(i)
                     for i in range(self._fs.nlocal_spaces))

    def _subcomponents(self, i):
        """
        Return the subfunctions of the local mixed function storage
        corresponding to the i-th local function.
        """
        return tuple(self._fbuf.subfunctions[j]
                     for j in self._fs._component_indices(i))

    @PETSc.Log.EventDecorator()
    def riesz_representation(self, riesz_map="L2", **kwargs):
        """
        Return the Riesz representation of this :class:`EnsembleFunction`
        with respect to the given Riesz map.

        Parameters
        ----------

        riesz_map
            The Riesz map to use (`l2`, `L2`, or `H1`). This can also be a callable.

        kwargs
            other arguments to be passed to the firedrake.riesz_map.
        """
        riesz = EnsembleFunction(self._fs.dual())
        for uself, uriesz in zip(self.subfunctions, riesz.subfunctions):
            uriesz.assign(
                uself.riesz_representation(
                    riesz_map=riesz_map, **kwargs))
        return riesz

    @PETSc.Log.EventDecorator()
    def assign(self, other, subsets=None):
        r"""Set the :class:`EnsembleFunction` to the value of another
        :class:`EnsembleFunction` other.

        Parameters
        ----------

        other : :class:`EnsembleFunction`
            The value to assign from.

        subsets : Collection[Optional[:class:`pyop2.types.set.Subset`]]
            One subset for each local :class:`Function`. None elements
            will be ignored.  The values of each local function will
            only be assigned on the nodes on the corresponding subset.
        """
        if type(other) is not type(self):
            raise ValueError(
                f"Cannot assign {type(self)} from {type(other)}")
        for i in range(self._fs.nlocal_spaces):
            self.subfunctions[i].assign(
                other.subfunctions[i],
                subset=subsets[i] if subsets else None)
        return self

    @PETSc.Log.EventDecorator()
    def copy(self):
        """
        Return a deep copy of the :class:`EnsembleFunction`.
        """
        new = type(self)(self.function_space())
        new.assign(self)
        return new

    @PETSc.Log.EventDecorator()
    def zero(self, subsets=None):
        """
        Set values to zero.

        Parameters
        ----------

        subsets : Collection[Optional[:class:`pyop2.types.set.Subset`]]
            One subset for each local :class:`Function`. None elements
            will be ignored.  The values of each local function will
            only be zeroed on the nodes on the corresponding subset.
        """
        for i in range(self._fs.nlocal_spaces):
            self.subfunctions[i].zero(
                subset=subsets[i] if subsets else None)
        return self

    @PETSc.Log.EventDecorator()
    def __iadd__(self, other):
        for us, uo in zip(self.subfunctions, other.subfunctions):
            us.assign(us + uo)
        return self

    @PETSc.Log.EventDecorator()
    def __imul__(self, other):
        if type(other) is type(self):
            for us, uo in zip(self.subfunctions, other.subfunctions):
                us.assign(us*uo)
        else:
            for us in self.subfunctions:
                us *= other
        return self

    @PETSc.Log.EventDecorator()
    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    @PETSc.Log.EventDecorator()
    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    @PETSc.Log.EventDecorator()
    def __rmul__(self, other):
        return self.__mul__(other)

    @contextmanager
    def vec(self):
        """
        Context manager for the global PETSc Vec with read/write access.

        It is invalid to access the Vec outside of a context manager.
        """
        # _fbuf.vec shares the same storage as _vec, so we need this
        # nested context manager so that the data gets copied to/from
        # the Function.dat storage and _vec.
        # However, this copy is done without _vec knowing, so we have
        # to manually increment the state.
        with self._fbuf.dat.vec:
            self._vec.stateIncrease()
            yield self._vec

    @contextmanager
    def vec_ro(self):
        """
        Context manager for the global PETSc Vec with read only access.

        It is invalid to access the Vec outside of a context manager.
        """
        # _fbuf.vec shares the same storage as _vec, so we need this
        # nested context manager to make sure that the data gets copied
        # to the Function.dat storage and _vec.
        with self._fbuf.dat.vec_ro:
            self._vec.stateIncrease()
            yield self._vec

    @contextmanager
    def vec_wo(self):
        """
        Context manager for the global PETSc Vec with write only access.

        It is invalid to access the Vec outside of a context manager.
        """
        # _fbuf.vec shares the same storage as _vec, so we need this
        # nested context manager to make sure that the data gets copied
        # from the Function.dat storage and _vec.
        with self._fbuf.dat.vec_wo:
            yield self._vec


class EnsembleFunction(EnsembleFunctionBase):
    """
    A mixed finite element Function distributed over an ensemble.

    Parameters
    ----------

    function_space : :class:`EnsembleFunctionSpace`
        The function space of the function.
    """
    def __new__(cls, function_space):
        if isinstance(function_space, EnsembleDualSpace):
            return EnsembleCofunction(function_space)
        return super().__new__(cls)

    def __init__(self, function_space):
        if not isinstance(function_space, EnsembleFunctionSpace):
            raise TypeError(
                "EnsembleFunction must be created using an EnsembleFunctionSpace")
        super().__init__(function_space)

    def norm(self, *args, **kwargs):
        """
        Compute the norm of the function.
        Any arguments are forwarded to :function:`firedrake.norm`.

        Parameters
        ----------

        norm_type : str
            The type of norm to compute.
            See :function:`firedrake.norm` for options.
        """
        return self._fs.ensemble_comm.allreduce(
            sum(norm(u, *args, **kwargs) for u in self.subfunctions))


class EnsembleCofunction(EnsembleFunctionBase):
    """
    A mixed finite element Cofunction distributed over an ensemble.

    Parameters
    ----------

    function_space : :class:`EnsembleDualSpace`
        The function space of the cofunction.
    """
    def __init__(self, function_space):
        if not isinstance(function_space, EnsembleDualSpace):
            raise TypeError(
                "EnsembleCofunction must be created using an EnsembleDualSpace")
        super().__init__(function_space)
