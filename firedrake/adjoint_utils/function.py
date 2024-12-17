from functools import wraps, cached_property
import ufl
from ufl.domain import extract_unique_domain
from pyadjoint.overloaded_type import create_overloaded_object, FloatingType
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape, no_annotations
from firedrake.adjoint_utils.blocks import FunctionAssignBlock, ProjectBlock, SubfunctionBlock, FunctionMergeBlock, SupermeshProjectBlock
import firedrake
from .checkpointing import disk_checkpointing, CheckpointFunction, \
    CheckpointBase, checkpoint_init_data, DelegatedFunctionCheckpoint


class FunctionMixin(FloatingType):

    @staticmethod
    def _ad_annotate_init(init):
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            FloatingType.__init__(self, *args,
                                  block_class=kwargs.pop("block_class", None),
                                  _ad_floating_active=kwargs.pop("_ad_floating_active", False),
                                  _ad_args=kwargs.pop("_ad_args", None),
                                  output_block_class=kwargs.pop("output_block_class", None),
                                  _ad_output_args=kwargs.pop("_ad_output_args", None),
                                  _ad_outputs=kwargs.pop("_ad_outputs", None),
                                  ad_block_tag=kwargs.pop("ad_block_tag", None), **kwargs)
            init(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def _ad_annotate_project(project):
        @wraps(project)
        def wrapper(self, b, *args, **kwargs):
            ad_block_tag = kwargs.pop("ad_block_tag", None)
            annotate = annotate_tape(kwargs)

            if annotate:
                bcs = kwargs.get("bcs", [])
                if isinstance(b, firedrake.Function) and extract_unique_domain(b) != self.function_space().mesh():
                    block = SupermeshProjectBlock(b, self.function_space(), self, bcs, ad_block_tag=ad_block_tag)
                else:
                    block = ProjectBlock(b, self.function_space(), self, bcs, ad_block_tag=ad_block_tag)

                tape = get_working_tape()
                tape.add_block(block)

            with stop_annotating():
                output = project(self, b, *args, **kwargs)

            if annotate:
                block.add_output(output.create_block_variable())

            return output
        return wrapper

    @staticmethod
    def _ad_annotate_subfunctions(subfunctions):
        @wraps(subfunctions)
        def wrapper(self, *args, **kwargs):
            ad_block_tag = kwargs.pop("ad_block_tag", None)
            annotate = annotate_tape(kwargs)
            with stop_annotating():
                output = subfunctions(self, *args, **kwargs)

            if annotate:
                output = tuple(type(self)(output[i].function_space(),
                                          output[i],
                                          block_class=SubfunctionBlock,
                                          _ad_floating_active=True,
                                          _ad_args=[self, i],
                                          _ad_output_args=[i],
                                          output_block_class=FunctionMergeBlock,
                                          _ad_outputs=[self],
                                          ad_block_tag=ad_block_tag)
                               for i in range(len(output)))
            return output
        return wrapper

    @staticmethod
    def _ad_annotate_copy(copy):
        @wraps(copy)
        def wrapper(self, *args, **kwargs):
            ad_block_tag = kwargs.pop("ad_block_tag", None)
            annotate = annotate_tape(kwargs)
            func = copy(self, *args, **kwargs)

            if annotate:
                if kwargs.pop("deepcopy", False):
                    block = FunctionAssignBlock(func, self, ad_block_tag=ad_block_tag)
                    tape = get_working_tape()
                    tape.add_block(block)
                    block.add_output(func.create_block_variable())
                else:
                    # TODO: Implement. Here we would need to use floating types.
                    raise NotImplementedError("Currently kwargs['deepcopy'] must be set True")

            return func

        return wrapper

    @staticmethod
    def _ad_annotate_assign(assign):
        @wraps(assign)
        def wrapper(self, other, *args, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Firedrake assign call."""
            ad_block_tag = kwargs.pop("ad_block_tag", None)
            # do not annotate in case of self assignment
            annotate = annotate_tape(kwargs) and self != other

            if annotate:
                if not isinstance(other, ufl.core.operator.Operator):
                    other = create_overloaded_object(other)
                block = FunctionAssignBlock(self, other, ad_block_tag=ad_block_tag)
                tape = get_working_tape()
                tape.add_block(block)

            with stop_annotating():
                ret = assign(self, other, *args, **kwargs)

            if annotate:
                block_var = self.create_block_variable()
                block.add_output(block_var)

                if isinstance(other, type(self)):
                    if self.function_space().mesh() == other.function_space().mesh():
                        block_var._checkpoint = DelegatedFunctionCheckpoint(other.block_variable)

            return ret

        return wrapper

    @staticmethod
    def _ad_not_implemented(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if annotate_tape(kwargs):
                raise NotImplementedError("Automatic differentiation is not supported for this operation.")
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def _ad_annotate_iadd(__iadd__):
        @wraps(__iadd__)
        def wrapper(self, other, **kwargs):
            with stop_annotating():
                func = __iadd__(self, other, **kwargs)

            ad_block_tag = kwargs.pop("ad_block_tag", None)
            annotate = annotate_tape(kwargs)
            if annotate:
                block = FunctionAssignBlock(func, self + other, ad_block_tag=ad_block_tag)
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(func.create_block_variable())

            return func

        return wrapper

    @staticmethod
    def _ad_annotate_isub(__isub__):
        @wraps(__isub__)
        def wrapper(self, other, **kwargs):
            with stop_annotating():
                func = __isub__(self, other, **kwargs)

            ad_block_tag = kwargs.pop("ad_block_tag", None)
            annotate = annotate_tape(kwargs)
            if annotate:
                block = FunctionAssignBlock(func, self - other, ad_block_tag=ad_block_tag)
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(func.create_block_variable())

            return func

        return wrapper

    @staticmethod
    def _ad_annotate_imul(__imul__):
        @wraps(__imul__)
        def wrapper(self, other, **kwargs):
            with stop_annotating():
                func = __imul__(self, other, **kwargs)

            ad_block_tag = kwargs.pop("ad_block_tag", None)
            annotate = annotate_tape(kwargs)
            if annotate:
                block = FunctionAssignBlock(func, self*other, ad_block_tag=ad_block_tag)
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(func.create_block_variable())

            return func

        return wrapper

    @staticmethod
    def _ad_annotate_itruediv(__itruediv__):
        @wraps(__itruediv__)
        def wrapper(self, other, **kwargs):
            with stop_annotating():
                func = __itruediv__(self, other, **kwargs)

            ad_block_tag = kwargs.pop("ad_block_tag", None)
            annotate = annotate_tape(kwargs)
            if annotate:
                block = FunctionAssignBlock(func, self/other, ad_block_tag=ad_block_tag)
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(func.create_block_variable())

            return func

        return wrapper

    def _ad_create_checkpoint(self):
        if disk_checkpointing():
            return CheckpointFunction(self)
        else:
            return self.copy(deepcopy=True)

    def _ad_convert_riesz(self, value, options=None):
        from firedrake import Function, Cofunction

        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "L2")
        solver_options = options.get("solver_options", {})
        if value == 0.:
            # In adjoint-based differentiation, value == 0. arises only when
            # the functional is independent on the control variable.
            return Function(V)

        if not isinstance(value, (Cofunction, Function)):
            raise TypeError("Expected a Cofunction or a Function")

        if callable(riesz_representation):
            return riesz_representation(value)

        return RieszMap(self, riesz_representation,
                        solver_options=solver_options)(value)

    @no_annotations
    def _ad_convert_type(self, value, options=None):
        # `_ad_convert_type` is not annotated, unlike `_ad_convert_riesz`
        options = {} if options is None else options.copy()
        options.setdefault("riesz_representation", "L2")
        if options["riesz_representation"] is None:
            if value == 0.:
                # In adjoint-based differentiation, value == 0. arises only when
                # the functional is independent on the control variable.
                V = options.get("function_space", self.function_space())
                return firedrake.Cofunction(V.dual())
            else:
                return value
        else:
            return self._ad_convert_riesz(value, options=options)

    def _ad_restore_at_checkpoint(self, checkpoint):
        if isinstance(checkpoint, CheckpointBase):
            return checkpoint.restore()
        else:
            return checkpoint

    def _ad_will_add_as_dependency(self):
        """Method called when the object is added as a Block dependency.

        """
        with checkpoint_init_data():
            super()._ad_will_add_as_dependency()

    def _ad_mul(self, other):
        from firedrake import Function

        r = Function(self.function_space())
        # `self` can be a Cofunction in which case only left multiplication with a scalar is allowed.
        r.assign(other * self)
        return r

    def _ad_add(self, other):
        from firedrake import Function

        r = Function(self.function_space())
        Function.assign(r, self + other)
        return r

    def _ad_dot(self, other, options=None):
        from firedrake import assemble, action, Cofunction

        if isinstance(other, Cofunction):
            return assemble(action(other, self))

        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "L2")
        if riesz_representation == "l2":
            return self.dat.inner(other.dat)
        elif riesz_representation == "L2":
            return assemble(firedrake.inner(self, other)*firedrake.dx)
        elif riesz_representation == "H1":
            return assemble((firedrake.inner(self, other)
                            + firedrake.inner(firedrake.grad(self), other))*firedrake.dx)
        else:
            raise NotImplementedError(
                "Unknown Riesz representation %s" % riesz_representation)

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        range_begin, range_end = dst.vector().local_range()
        m_a_local = src[offset + range_begin:offset + range_end]
        dst.vector().set_local(m_a_local)
        dst.vector().apply('insert')
        offset += dst.vector().size()
        return dst, offset

    @staticmethod
    def _ad_to_list(m):
        if not hasattr(m, "gather"):
            m_v = m.vector()
        else:
            m_v = m
        m_a = m_v.gather()

        return m_a.tolist()

    def _ad_copy(self):
        from firedrake import Function

        r = Function(self.function_space())
        r.assign(self)
        return r

    def _ad_dim(self):
        return self.function_space().dim()

    def _ad_imul(self, other):
        self *= other
        return self

    def _ad_iadd(self, other):
        self += other
        return self

    def _ad_function_space(self, mesh):
        return self.ufl_function_space()

    def _reduce(self, r, r0):
        vec = self.vector().get_local()
        for i in range(len(vec)):
            r0 = r(vec[i], r0)
        return r0

    def _applyUnary(self, f):
        vec = self.vector()
        npdata = vec.get_local()
        for i in range(len(npdata)):
            npdata[i] = f(npdata[i])
        vec.set_local(npdata)

    def _applyBinary(self, f, y):
        vec = self.vector()
        npdata = vec.get_local()
        npdatay = y.vector().get_local()
        for i in range(len(npdata)):
            npdata[i] = f(npdata[i], npdatay[i])
        vec.set_local(npdata)

    def _ad_from_petsc(self, vec):
        with self.dat.vec_wo as self_v:
            vec.copy(result=self_v)

    def _ad_to_petsc(self, vec=None):
        with self.dat.vec_ro as self_v:
            if vec:
                self_v.copy(result=vec)
            else:
                vec = self_v.copy()
        return vec

    def __deepcopy__(self, memodict={}):
        return self.copy(deepcopy=True)


class CofunctionMixin(FunctionMixin):

    def _ad_dot(self, other):
        return firedrake.assemble(firedrake.action(self, other))


class RieszMap:
    """Return a map from a dual to a primal function space.

    A `RieszMap` can be called on a `Cofunction` in the appropriate space
    to yield the `Function` which is the Riesz representer under the.

    Parameters
    ----------
    function_space_or_inner_product: FunctionSpace or DualSpace or Function or Cofunction or ufl.Form
        The space from which to map, or a bilinear form defining an
        inner product.
    sobolev_space: String or ufl.sobolevspace.SobolevSpace.
        Used to determine the inner product.
    bcs: DirichletBC or list of DirichletBC
        Boundary conditions to apply to the Riesz map.
    solver_options: dict
        A dictionary of PETSc options to be passed to the solver.
    """

    def __init__(self, function_space_or_inner_product=None,
                 sobolev_space=ufl.L2, *, bcs=None,  solver_options=None):
        if isinstance(function_space_or_inner_product, ufl.Form):
            args = ufl.algorithms.extract_arguments(
                function_space_or_inner_product
            )
            if len(args) != 2:
                raise ValueError(f"inner_product has arity {len(args)}, "
                                 "should be 2.")
            function_space = args[0].function_space()
            inner_product = function_space_or_inner_product
        else:
            function_space = function_space_or_inner_product
            if hasattr(function_space, "function_space"):
                function_space = function_space.function_space()
            if ufl.duals.is_dual(function_space):
                function_space = function_space.dual()

            if str(sobolev_space) == "l2":
                inner_product = "l2"
            else:
                from firedrake import TrialFunction, TestFunction
                u = TrialFunction(function_space)
                v = TestFunction(function_space)
                inner_product = RieszMap._inner_product_form(
                    sobolev_space, u, v
                )

        self._function_space = function_space
        self._inner_product = inner_product
        self._bcs = bcs
        self._solver_options = solver_options

    @staticmethod
    def _inner_product_form(sobolev_space, u, v):
        from firedrake import inner, dx, grad
        inner_products = {
            "L2": lambda u, v: inner(u, v)*dx,
            "H1": lambda u, v: inner(u, v)*dx + inner(grad(u), grad(v))*dx
        }
        try:
            return inner_products[str(sobolev_space)](u, v)
        except KeyError:
            raise ValueError("No inner product defined for Sobolev space "
                             f"{sobolev_space}.")

    @cached_property
    def _solver(self):
        from firedrake import (LinearVariationalSolver,
                               LinearVariationalProblem, Function, Cofunction)
        rhs = Cofunction(self._function_space.dual())
        soln = Function(self._function_space)
        lvp = LinearVariationalProblem(self._inner_product, rhs, soln,
                                       bcs=self._bcs)
        solver = LinearVariationalSolver(
            lvp, solver_parameters=self._solver_options
        )
        return solver.solve, rhs, soln

    def __call__(self, cofunction):
        """Return the Riesz representer of a Cofunction."""
        from firedrake import Function
        if cofunction.function_space().dual() != self._function_space:
            raise ValueError("Function space mismatch in RieszMap.")
        output = Function(self._function_space)

        if self._inner_product == "l2":
            output.dat.data[:] = cofunction.dat.data[:]
        else:
            solve, rhs, soln = self._solver
            rhs.assign(cofunction)
            solve()
            output = Function(self._function_space)
            output.assign(soln)
        return output
