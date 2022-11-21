import functools
import operator

import numpy as np
from pyadjoint.tape import annotate_tape
from pyop2.utils import cached_property
import pytools
from ufl.algorithms import extract_coefficients
from ufl.constantvalue import as_ufl
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction

from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.utils import ScalarType, split_by
from firedrake.vector import Vector


class CoefficientCollector(MultiFunction):
    """Multifunction used for converting an expression into a weighted sum of coefficients.

    Calling ``map_expr_dag(CoefficientCollector(), expr)`` will return a tuple whose entries
    are of the form ``(coefficient, weight)``. Expressions that cannot be expressed as a
    weighted sum will raise an exception.

    Note: As well as being simple weighted sums (e.g. ``u.assign(2*v1 + 3*v2)``), one can
    also assign constant expressions of the appropriate shape (e.g. ``u.assign(1.0)`` or
    ``u.assign(2*v + 3)``). Therefore the returned tuple must be split since ``coefficient``
    may be either a :class:`firedrake.constant.Constant` or :class:`firedrake.function.Function`.
    """

    def product(self, o, a, b):
        scalars, vectors = split_by(self._is_scalar_equiv, [a, b])
        # Case 1: scalar * scalar
        if len(scalars) == 2:
            # Compress the first argument (arbitrary)
            scalar, vector = scalars
        # Case 2: scalar * vector
        elif len(scalars) == 1:
            scalar, = scalars
            vector, = vectors
        # Case 3: vector * vector (invalid)
        else:
            raise ValueError("Expressions containing the product of two vector-valued "
                             "subexpressions cannot be used for assignment. Consider using "
                             "interpolate instead.")
        scaling = self._as_scalar(scalar)
        return tuple((coeff, weight*scaling) for coeff, weight in vector)

    def division(self, o, a, b):
        # Division is only valid if b (the divisor) is a scalar
        if self._is_scalar_equiv(b):
            divisor = self._as_scalar(b)
            return tuple((coeff, weight/divisor) for coeff, weight in a)
        else:
            raise ValueError("Expressions involving division by a vector-valued subexpression "
                             "cannot be used for assignment. Consider using interpolate instead.")

    def sum(self, o, a, b):
        # Note: a and b are tuples of (coefficient, weight) so addition is concatenation
        return a + b

    def power(self, o, a, b):
        # Only valid if a and b are scalars
        return ((Constant(self._as_scalar(a) ** self._as_scalar(b)), 1),)

    def abs(self, o, a):
        # Only valid if a is a scalar
        return ((Constant(abs(self._as_scalar(a))), 1),)

    def _scalar(self, o):
        return ((Constant(o), 1),)

    int_value = _scalar
    float_value = _scalar
    zero = _scalar

    def multi_index(self, o):
        pass

    def indexed(self, o, a, _):
        return a

    def component_tensor(self, o, a, _):
        return a

    def coefficient(self, o):
        return ((o, 1),)

    def expr(self, o, *operands):
        raise NotImplementedError(f"Handler not defined for {type(o)}")

    def _is_scalar_equiv(self, weighted_coefficients):
        """Return ``True`` if the sequence of ``(coefficient, weight)`` can be compressed to
        a single scalar value.

        This is only true when all coefficients are :class:`firedrake.Constant` and have
        shape ``(1,)``.
        """
        return all(isinstance(c, Constant) and c.dat.dim == (1,)
                   for (c, _) in weighted_coefficients)

    def _as_scalar(self, weighted_coefficients):
        """Compress a sequence of ``(coefficient, weight)`` tuples to a single scalar value.

        This is necessary because we do not know a priori whether a :class:`firedrake.Constant`
        is going to be used as a scale factor (e.g. ``u.assign(Constant(2)*v)``), or as a
        constant to be added (e.g. ``u.assign(2*v + Constant(3))``). Therefore we only
        compress to a scalar when we know it is required (e.g. inside a product with a
        :class:`firedrake.Function`).
        """
        return pytools.one(
            functools.reduce(operator.add, (c.dat.data_ro*w for c, w in weighted_coefficients))
        )


class Assigner:
    """Class performing pointwise assignment of an expression to a :class:`firedrake.Function`.

    :param assignee: The :class:`firedrake.Function` being assigned to.
    :param expression: The :class:`ufl.Expr` to evaluate.
    :param subset: Optional subset (:class:`op2.Subset`) to apply the assignment over.
    """
    symbol = "="

    _coefficient_collector = CoefficientCollector()

    def __init__(self, assignee, expression, subset=None):
        if isinstance(expression, Vector):
            expression = expression.function
        expression = as_ufl(expression)

        if not all(c.function_space() == assignee.function_space()
                   for c in extract_coefficients(expression)
                   if isinstance(c, Function)):
            raise ValueError("All functions in the expression must be in the same "
                             "function space as the assignee")

        self._assignee = assignee
        self._expression = expression
        self._subset = subset

    def __str__(self):
        return f"{self._assignee} {self.symbol} {self._expression}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._assignee!r}, {self._expression!r})"

    @PETSc.Log.EventDecorator()
    def assign(self):
        """Perform the assignment."""
        if annotate_tape():
            raise NotImplementedError(
                "Taping with explicit Assigner objects is not supported yet. "
                "Use Function.assign instead."
            )

        if self._is_real_space:
            self._assign_global(self._assignee.dat, [f.dat for f in self._functions])
        else:
            # If mixed, loop over individual components
            for assignee_dat, *func_dats in zip(self._assignee.dat.split,
                                                *(f.dat.split for f in self._functions)):
                self._assign_single_dat(assignee_dat, func_dats)
                # Halo values are also updated
                assignee_dat.halo_valid = True

    @cached_property
    def _constants(self):
        return tuple(c for (c, _) in self._weighted_coefficients
                     if isinstance(c, Constant))

    @cached_property
    def _constant_weights(self):
        return tuple(w for (c, w) in self._weighted_coefficients
                     if isinstance(c, Constant))

    @cached_property
    def _functions(self):
        return tuple(c for (c, _) in self._weighted_coefficients
                     if isinstance(c, Function))

    @cached_property
    def _function_weights(self):
        return tuple(w for (c, w) in self._weighted_coefficients
                     if isinstance(c, Function))

    @property
    def _indices(self):
        return self._subset.indices if self._subset else ...

    @property
    def _is_real_space(self):
        return self._assignee.function_space().ufl_element().family() == "Real"

    def _assign_global(self, assignee_global, function_globals):
        assignee_global.data[self._indices] = self._compute_rvalue(function_globals)

    # TODO: It would be more efficient in permissible cases to use VecMAXPY instead of numpy operations.
    def _assign_single_dat(self, assignee_dat, function_dats):
        assignee_dat.data_with_halos[self._indices] = self._compute_rvalue(function_dats)

    def _compute_rvalue(self, function_dats=()):
        # There are two components to the rvalue: weighted functions (in the same function space),
        # and constants (e.g. u.assign(2*v + 3)).
        if self._is_real_space:
            func_data = np.array([f.data_ro[self._indices] for f in function_dats])
        else:
            func_data = np.array([f.data_ro_with_halos[self._indices] for f in function_dats])
        func_rvalue = (func_data.T @ self._function_weights).T
        const_data = np.array([c.dat.data_ro for c in self._constants], dtype=ScalarType)
        const_rvalue = const_data.T @ self._constant_weights
        return func_rvalue + const_rvalue

    @cached_property
    def _weighted_coefficients(self):
        # TODO: It would be nice to stash this on the expression so we can avoid extra
        # traversals for non-persistent Assigner objects, but expressions do not currently
        # have caches attached to them.
        return map_expr_dag(self._coefficient_collector, self._expression)


class IAddAssigner(Assigner):
    """Assigner class for :func:`firedrake.Function.__iadd__`."""
    symbol = "+="

    def _assign_single_dat(self, assignee_dat, function_dats):
        assignee_dat.data_with_halos[self._indices] += self._compute_rvalue(function_dats)


class ISubAssigner(Assigner):
    """Assigner class for :func:`firedrake.Function.__isub__`."""
    symbol = "-="

    def _assign_single_dat(self, assignee_dat, function_dats):
        assignee_dat.data_with_halos[self._indices] -= self._compute_rvalue(function_dats)


class IMulAssigner(Assigner):
    """Assigner class for :func:`firedrake.Function.__imul__`."""
    symbol = "*="

    def _assign_single_dat(self, assignee_dat, function_dats):
        if function_dats:
            raise ValueError("Only multiplication by scalars is supported")
        assignee_dat.data_with_halos[self._indices] *= self._compute_rvalue()


class IDivAssigner(Assigner):
    """Assigner class for :func:`firedrake.Function.__itruediv__`."""
    symbol = "/="

    def _assign_single_dat(self, assignee_dat, function_dats):
        if function_dats:
            raise ValueError("Only division by scalars is supported")
        assignee_dat.data_with_halos[self._indices] /= self._compute_rvalue()
