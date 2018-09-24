#include <AMReX_ParmParse.H>

#include <AMReX_BC_TYPES.H>
#include <AMReX_Box.H>
#include <AMReX_VisMF.H>
#include <incflo_diff_F.H>
#include <incflo_level.H>

// For multigrid
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>

using namespace std;

//
// Explicit diffusion
//
void incflo_level::incflo_compute_divtau(int lev,
										 MultiFab& divtau,
										 Vector<std::unique_ptr<MultiFab>>& vel)
{
	BL_PROFILE("incflo_level::incflo_compute_divtau");
	Box domain(geom[lev].Domain());

   // Get EB geometric info
   Array< const MultiCutFab*,AMREX_SPACEDIM> areafrac;
   Array< const MultiCutFab*,AMREX_SPACEDIM> facecent;
   const amrex::MultiFab*                    volfrac;
   const amrex::MultiCutFab*                 bndrycent;

   areafrac  =   ebfactory[lev] -> getAreaFrac();
   facecent  =   ebfactory[lev] -> getFaceCent();
   volfrac   = &(ebfactory[lev] -> getVolFrac());
   bndrycent = &(ebfactory[lev] -> getBndryCent());

#ifdef _OPENMP
#pragma omp parallel
#endif
   for (MFIter mfi(*vel[lev],true); mfi.isValid(); ++mfi) {

      // Tilebox
      Box bx = mfi.tilebox ();

      // this is to check efficiently if this tile contains any eb stuff
      const EBFArrayBox&  vel_fab = dynamic_cast<EBFArrayBox const&>((*vel[lev])[mfi]);
      const EBCellFlagFab&  flags = vel_fab.getEBCellFlagFab();

      if (flags.getType(bx) == FabType::covered)
      {
         divtau[mfi].setVal(1.2345e200, bx, 0, 3);
      }
      else
      {
         if (flags.getType(amrex::grow(bx,nghost)) == FabType::regular)
         {
            compute_divtau(
               BL_TO_FORTRAN_BOX(bx),
               BL_TO_FORTRAN_ANYD(divtau[mfi]),
               BL_TO_FORTRAN_ANYD((*vel[lev])[mfi]),
               (*mu_g[lev])[mfi].dataPtr(),
               (*lambda_g[lev])[mfi].dataPtr(),
               BL_TO_FORTRAN_ANYD((*ro_g[lev])[mfi]),
               domain.loVect (), domain.hiVect (),
               bc_ilo.dataPtr(), bc_ihi.dataPtr(),
               bc_jlo.dataPtr(), bc_jhi.dataPtr(),
               bc_klo.dataPtr(), bc_khi.dataPtr(),
               geom[lev].CellSize(), &nghost, &explicit_diffusion);
         }
         else
         {
            compute_divtau_eb(
               BL_TO_FORTRAN_BOX(bx),
               BL_TO_FORTRAN_ANYD(divtau[mfi]),
               BL_TO_FORTRAN_ANYD((*vel[lev])[mfi]),
               (*mu_g[lev])[mfi].dataPtr(),
               (*lambda_g[lev])[mfi].dataPtr(),
               BL_TO_FORTRAN_ANYD((*ro_g[lev])[mfi]),
               BL_TO_FORTRAN_ANYD(flags),
               BL_TO_FORTRAN_ANYD((*areafrac[0])[mfi]),
               BL_TO_FORTRAN_ANYD((*areafrac[1])[mfi]),
               BL_TO_FORTRAN_ANYD((*areafrac[2])[mfi]),
               BL_TO_FORTRAN_ANYD((*facecent[0])[mfi]),
               BL_TO_FORTRAN_ANYD((*facecent[1])[mfi]),
               BL_TO_FORTRAN_ANYD((*facecent[2])[mfi]),
               BL_TO_FORTRAN_ANYD((*volfrac)[mfi]),
               BL_TO_FORTRAN_ANYD((*bndrycent)[mfi]),
               domain.loVect (), domain.hiVect (),
               bc_ilo.dataPtr(), bc_ihi.dataPtr(),
               bc_jlo.dataPtr(), bc_jhi.dataPtr(),
               bc_klo.dataPtr(), bc_khi.dataPtr(),
               geom[lev].CellSize(), &nghost, &explicit_diffusion);

         }
      }
   }
}







//
// Implicit diffusion
//
void incflo_level::incflo_diffuse_velocity(int lev, amrex::Real dt)

{
	BL_PROFILE("incflo_level::incflo_diffuse_velocity");

	// Whole domain
	Box domain(geom[lev].Domain());

	// Swap ghost cells and apply BCs to velocity
	incflo_set_velocity_bcs(lev, 0);

	// Compute the coefficients
	incflo_compute_bcoeff_diff(lev);

	int bc_lo[3], bc_hi[3];

	// Set BCs for Poisson's solver
	set_diff_bc(bc_lo,
				bc_hi,
				domain.loVect(),
				domain.hiVect(),
				&nghost,
				bc_ilo.dataPtr(),
				bc_ihi.dataPtr(),
				bc_jlo.dataPtr(),
				bc_jhi.dataPtr(),
				bc_klo.dataPtr(),
				bc_khi.dataPtr());

	// Loop over the velocity components
	for(int i = 0; i < 3; i++)
	{
		rhs_diff[lev]->copy(*vel_g[lev], i, 0, 1, nghost, nghost);
		phi_diff[lev]->copy(*vel_g[lev], i, 0, 1, nghost, nghost);

		amrex::Print() << "Diffusing velocity component " << i << std::endl;

		// Solve (1 - div beta grad) u_new = RHS
		// Here RHS = "vel" which is the current approximation to the new-time velocity (without diffusion terms)
		solve_diffusion_equation(lev, bcoeff_diff, phi_diff, rhs_diff, bc_lo, bc_hi, dt);

		vel_g[lev]->copy(*phi_diff[lev], 0, i, 1, nghost, nghost);
	}

	// Swap ghost cells and apply BCs to velocity
	incflo_set_velocity_bcs(lev, 0);
}

//
// Computes the following decomposition:
//
//    u + grad(phi)/ro_g = u*,     where div(eps*u) = 0
//
// where u* is a non-div-free velocity field, stored
// by components in u_g, v_g, and w_g. The resulting div-free
// velocity field, u, overwrites the value of u* in u_g, v_g, and w_g.
//
// phi is an auxiliary function related to the pressure p_g by the relation:
//
//     new p_g  = old p_g + phi

//
// Solve :
//
//                  (alpha + div dot beta grad) u = RHS
//
void incflo_level::solve_diffusion_equation(int lev,
											Vector<Vector<std::unique_ptr<MultiFab>>>& b,
											Vector<std::unique_ptr<MultiFab>>& sol,
											Vector<std::unique_ptr<MultiFab>>& rhs,
											int bc_lo[],
											int bc_hi[],
											amrex::Real dt)
{
	BL_PROFILE("incflo_level::solve_diffusion_equation");

	//
	// First define the matrix (operator).
	// Class MLABecLaplacian describes the following operator:
	//
	//       (alpha * a - beta * (del dot b grad)) sol
	//
	LPInfo info;
	MLABecLaplacian matrix(geom, grids, dmap, info);
	Vector<const MultiFab*> tmp;
	array<MultiFab const*, AMREX_SPACEDIM> b_tmp;

	// Copy the PPE coefficient into the proper data strutcure
	tmp = amrex::GetVecOfConstPtrs(b[lev]);
	b_tmp[0] = tmp[0];
	b_tmp[1] = tmp[1];
	b_tmp[2] = tmp[2];

	// It is essential that we set MaxOrder of the solver to 2
	// if we want to use the standard sol(i)-sol(i-1) approximation
	// for the gradient at Dirichlet boundaries.
	// The solver's default order is 3 and this uses three points for the
	// gradient at a Dirichlet boundary.
	matrix.setMaxOrder(2);

	// LinOpBCType Definitions are in amrex/Src/Boundary/AMReX_LO_BCTYPES.H
	matrix.setDomainBC({(LinOpBCType)bc_lo[0], (LinOpBCType)bc_lo[1], (LinOpBCType)bc_lo[2]},
					   {(LinOpBCType)bc_hi[0], (LinOpBCType)bc_hi[1], (LinOpBCType)bc_hi[2]});

	// This sets alpha = 1 and beta = dt
	matrix.setScalars(1.0, dt);

	// Define RHS = (ro) * (vel_g)
	MultiFab::Multiply((*rhs_diff[lev]), (*ro_g[lev]), 0, 0, 1, rhs_diff[lev]->nGrow());

	// This sets the spatially varying A coefficients
	matrix.setACoeffs(lev, (*ro_g[lev]));

	// This sets the spatially varying b coefficients
	matrix.setBCoeffs(lev, b_tmp);

	// By this point we must have filled the Dirichlet values of sol stored in the ghost cells
	matrix.setLevelBC(lev, GetVecOfConstPtrs(sol)[lev]);

	//
	// Then setup the solver ----------------------
	//
	MLMG solver(matrix);

	solver.setMaxIter(mg_max_iter);
	solver.setMaxFmgIter(mg_max_fmg_iter);
	solver.setVerbose(mg_verbose);
	solver.setCGVerbose(mg_cg_verbose);
	solver.setCGMaxIter(mg_cg_maxiter);

	// This ensures that ghost cells of sol are correctly filled when returned from the solver
	solver.setFinalFillBC(true);

	//
	// Finally, solve the system
	//
	solver.solve(GetVecOfPtrs(sol), GetVecOfConstPtrs(rhs), mg_rtol, mg_atol);

	sol[lev]->FillBoundary(geom[lev].periodicity());
}

//
// Computes bcoeff = mu_g at the faces of the scalar cells
//
void incflo_level::incflo_compute_bcoeff_diff(int lev)
{
	BL_PROFILE("incflo_level::incflo_compute_bcoeff_diff");

	// Directions
	int xdir = 1;
	int ydir = 2;
	int zdir = 3;

#ifdef _OPENMP
#pragma omp parallel
#endif
	for(MFIter mfi(*mu_g[lev], true); mfi.isValid(); ++mfi)
	{
		// Tileboxes for staggered components
		Box ubx = mfi.tilebox(e_x);
		Box vbx = mfi.tilebox(e_y);
		Box wbx = mfi.tilebox(e_z);

		// X direction
		compute_bcoeff_diff(BL_TO_FORTRAN_BOX(ubx),
							BL_TO_FORTRAN_ANYD((*(bcoeff_diff[lev][0]))[mfi]),
							BL_TO_FORTRAN_ANYD((*mu_g[lev])[mfi]),
							&xdir);

		// Y direction
		compute_bcoeff_diff(BL_TO_FORTRAN_BOX(vbx),
							BL_TO_FORTRAN_ANYD((*(bcoeff_diff[lev][1]))[mfi]),
							BL_TO_FORTRAN_ANYD((*mu_g[lev])[mfi]),
							&ydir);

		// Z direction
		compute_bcoeff_diff(BL_TO_FORTRAN_BOX(wbx),
							BL_TO_FORTRAN_ANYD((*(bcoeff_diff[lev][2]))[mfi]),
							BL_TO_FORTRAN_ANYD((*mu_g[lev])[mfi]),
							&zdir);
	}

	bcoeff_diff[lev][0]->FillBoundary(geom[lev].periodicity());
	bcoeff_diff[lev][1]->FillBoundary(geom[lev].periodicity());
	bcoeff_diff[lev][2]->FillBoundary(geom[lev].periodicity());
}