from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
import firedrake.dmhooks as dmhooks
import numpy


__all__ = ['FacetSplitPC']


class FacetSplitPC(PCBase):

    needs_python_pmat = False
    _prefix = "facet_"

    def initialize(self, pc):

        from ufl import InteriorElement, FacetElement, MixedElement, TensorElement, VectorElement
        from firedrake import FunctionSpace, TestFunctions, TrialFunctions, split
        from firedrake.assemble import allocate_matrix, TwoFormAssembler
        from firedrake.solving_utils import _SNESContext
        from functools import partial
        from mpi4py import MPI

        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        ctx = dmhooks.get_appctx(pc.getDM())
        if ctx is None:
            raise ValueError("No context found.")
        if not isinstance(ctx, _SNESContext):
            raise ValueError("Don't know how to get form from %r", ctx)

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix

        mat_type = ctx.mat_type
        problem = ctx._problem
        V = problem.J.arguments()[-1].function_space()

        # W = V_interior * V_facet
        scalar_element = V.ufl_element()
        if isinstance(scalar_element, (TensorElement, VectorElement)):
            scalar_element = scalar_element._sub_element
        tensorize = lambda e, shape: TensorElement(e, shape=shape) if shape else e
        elements = [appctx.get("interior_element", tensorize(InteriorElement(scalar_element), V.shape)),
                    appctx.get("facet_element", tensorize(FacetElement(scalar_element), V.shape))]
        W = FunctionSpace(V.mesh(), MixedElement(elements))
        if W.dim() != V.dim():
            raise ValueError("Dimensions of the original and decomposed spaces do not match")

        mixed_operator = problem.J(sum(TestFunctions(W)), sum(TrialFunctions(W)), coefficients={})
        mixed_bcs = [bc.reconstruct(V=W[-1], g=0) for bc in problem.bcs]

        ownership_ranges = V.dof_dset.layout_vec.getOwnershipRanges()
        start, end = ownership_ranges[V.comm.rank:V.comm.rank+2]

        v = V.get_work_function()
        w = W.get_work_function()
        with w.dat.vec_wo as wvec:
            wvec.setArray(numpy.linspace(0.0E0, 1.0E0, end-start, dtype=PETSc.RealType))

        w_expr = sum(split(w))
        try:
            v.interpolate(w_expr)
        except NotImplementedError:
            rtol = 1.0/max(numpy.diff(ownership_ranges))**2
            v.project(w_expr, solver_parameters={
                      "mat_type": "matfree",
                      "ksp_type": "cg",
                      "ksp_atol": 0,
                      "ksp_rtol": rtol,
                      "pc_type": "jacobi", })

        indices = numpy.rint((end-start-1)*v.dat.data_ro.reshape((-1,))+start).astype(PETSc.IntType)
        isorted = numpy.arange(start, end, dtype=PETSc.IntType)
        if V.comm.allreduce(numpy.array_equal(indices, isorted), MPI.PROD):
            self.perm = None
            self.iperm = None
        else:
            rindices = numpy.empty_like(indices)
            rindices[indices-start] = isorted
            self.perm = PETSc.IS().createGeneral(indices, comm=V.comm)
            self.iperm = PETSc.IS().createGeneral(rindices, comm=V.comm)

        def _permute_nullspace(nsp):
            if nsp is None or self.iperm is None:
                return nsp
            vecs = [vec.duplicate() for vec in nsp.getVecs()]
            for vec in vecs:
                vec.permute(self.iperm)
            return PETSc.NullSpace().create(constant=nsp.constant, vectors=vecs, comm=nsp.comm)

        if P.getType() == "python":
            self.mixed_op = allocate_matrix(mixed_operator,
                                            bcs=mixed_bcs,
                                            form_compiler_parameters=fcp,
                                            mat_type=mat_type,
                                            options_prefix=options_prefix)
            self._assemble_mixed_op = TwoFormAssembler(mixed_operator, tensor=self.mixed_op,
                                                       form_compiler_parameters=fcp,
                                                       bcs=mixed_bcs).assemble
            self._assemble_mixed_op()
            mixed_opmat = self.mixed_op.petscmat

            if False:
                # FIXME
                mixed_opmat.setNullSpace(_permute_nullspace(P.getNullSpace()))
                mixed_opmat.setNearNullSpace(_permute_nullspace(P.getNearNullSpace()))
                mixed_opmat.setTransposeNullSpace(_permute_nullspace(P.getTransposeNullSpace()))

        elif self.iperm:
            # self._permute_op = partial(P.permute, self.iperm, self.iperm)
            self._permute_op = partial(PETSc.Mat().createSubMatrixVirtual, P, self.iperm, self.iperm)
            mixed_opmat = self._permute_op()
        else:
            mixed_opmat = P

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -facet_pc_type ksp.
        scpc = PETSc.PC().create(comm=pc.comm)
        scpc.incrementTabLevel(1, parent=pc)

        # We set a DM and an appropriate SNESContext on the constructed PC so one
        # can do e.g. fieldsplit.
        mixed_dm = W.dm
        self._dm = mixed_dm

        # Create new appctx
        self._ctx_ref = self.new_snes_ctx(pc,
                                          mixed_operator,
                                          mixed_bcs,
                                          mat_type,
                                          fcp,
                                          options_prefix=options_prefix)

        scpc.setDM(mixed_dm)
        scpc.setOptionsPrefix(options_prefix)
        scpc.setOperators(A=mixed_opmat, P=mixed_opmat)
        with dmhooks.add_hooks(mixed_dm, self, appctx=self._ctx_ref, save=False):
            scpc.setFromOptions()
        self.pc = scpc

    def update(self, pc):
        if hasattr(self, "mixed_op"):
            self._assemble_mixed_op()
        elif hasattr(self, "_permute_op"):
            for mat in self.pc.getOperators():
                mat.destroy()
            P = self._permute_op()
            self.pc.setOperators(A=P, P=P)
        self.pc.setUp()

    def apply(self, pc, x, y):
        if self.iperm:
            x.permute(self.iperm)
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(x, y)
        if self.perm:
            x.permute(self.perm)
            y.permute(self.perm)

    def applyTranspose(self, pc, x, y):
        if self.iperm:
            x.permute(self.iperm)
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(x, y)
        if self.perm:
            x.permute(self.perm)
            y.permute(self.perm)

    def view(self, pc, viewer=None):
        super(FacetSplitPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC using interior-facet decomposition\n")
            self.pc.view(viewer)

    def destroy(self, pc):
        if hasattr(self, "pc"):
            if hasattr(self, "_permute_op"):
                for mat in self.pc.getOperators():
                    mat.destroy()
            self.pc.destroy()
        if hasattr(self, "iperm"):
            self.iperm.destroy()
        if hasattr(self, "perm"):
            self.perm.destroy()
