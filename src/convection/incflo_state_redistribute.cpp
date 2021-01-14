#include <incflo.H>
#include <AMReX_EB_slopes_K.H>

using namespace amrex;

#ifdef AMREX_USE_EB
void incflo::state_redistribute_eb (Box const& bx, int ncomp,
                                    Array4<Real> const& dUdt,
                                    Array4<Real const> const& dUdt_in,
                                    Array4<Real> const& scratch,
                                    Array4<EBCellFlag const> const& flag,
                                    AMREX_D_DECL(Array4<Real const> const& apx,
                                                 Array4<Real const> const& apy,
                                                 Array4<Real const> const& apz),
                                    Array4<Real const> const& vfrac,
                                    AMREX_D_DECL(Array4<Real const> const& fcx,
                                                 Array4<Real const> const& fcy,
                                                 Array4<Real const> const& fcz),
                                    Array4<Real const> const& ccent,
                                    Geometry& lev_geom)
{
    const Box domain = lev_geom.Domain();
    const Box dbox = lev_geom.growPeriodicDomain(2);

    const auto& is_periodic_x = lev_geom.isPeriodic(0);
    const auto& is_periodic_y = lev_geom.isPeriodic(1);

    amrex::Print() << " DOING BOX " << bx << " with ncomp " << ncomp << std::endl;

    Box const& bxg1 = amrex::grow(bx,1);
    Box const& bxg2 = amrex::grow(bx,2);

    // Set to 1 if cell is in my nbhd, otherwise 0
    IArrayBox nbor_fab      (bxg1,9);

    // How many nbhds is this cell in
    FArrayBox nrs_fab       (bxg2,1);

    // Total volume of all cells in my nbhd
    FArrayBox nbhd_vol_fab  (bxg2,1);

    // Centroid of my nbhd
    FArrayBox cent_hat_fab  (bxg2,AMREX_SPACEDIM);

    // Slopes in my nbhd
    FArrayBox slopes_hat_fab(bxg2,AMREX_SPACEDIM);

    // Solution at the centroid of my nbhd
    FArrayBox soln_hat_fab  (bxg2,ncomp);

    nbor_fab.setVal(0);
    nrs_fab.setVal(1.0);
    nbhd_vol_fab.setVal(0.);
    soln_hat_fab.setVal(0.);
    cent_hat_fab.setVal(0.);
    slopes_hat_fab.setVal(0.);

    Array4<int>  nbor     = nbor_fab.array();
    Array4<Real> nbhd_vol  = nbhd_vol_fab.array();
    Array4<Real> nrs      = nrs_fab.array();
    Array4<Real> soln_hat = soln_hat_fab.array();
    Array4<Real> cent_hat = cent_hat_fab.array();
    Array4<Real> slopes_hat = slopes_hat_fab.array();

    amrex::ParallelFor(bxg1,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (!flag(i,j,k).isCovered())
        {
          // Always include the small cell itself
          nbor(i,j,k,4) = 1;

          if (vfrac(i,j,k) < 0.5)
          {
            // We only include cells into a neighborhood if they are in the interior
            //    or in periodic ghost cells
            bool allow_lo_x = (i > domain.smallEnd(0) || is_periodic_x);
            bool allow_lo_y = (j > domain.smallEnd(1) || is_periodic_y);
            bool allow_hi_x = (i < domain.bigEnd(0)   || is_periodic_x);
            bool allow_hi_y = (j < domain.bigEnd(1)   || is_periodic_y);

            if ( apx(i,j,k) > 0.)
            {
                if (fcx(i,j,k,0) <= 0.)
                {
                    if (allow_lo_x)
                    {
                        if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                        if (allow_lo_y)
                            if (vfrac(i-1,j-1,k) > 0.) nbor(i,j,k,0) = 1;
                    }
                    if (allow_lo_y)
                        if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                } else {
                    if (allow_lo_x)
                    {
                        if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                        if (allow_hi_y)
                            if (vfrac(i-1,j+1,k) > 0.) nbor(i,j,k,6) = 1;
                    }
                    if (allow_hi_y)
                        if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                }
            }

            if (apx(i+1,j,k) > 0.)
            {
                if (fcx(i+1,j,k,0) <= 0.)
                {
                    if (allow_hi_x)
                    {
                        if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                        if (vfrac(i+1,j-1,k) > 0.) nbor(i,j,k,2) = 1;
                    }
                    if (allow_lo_y)
                        if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                } else {
                    if (allow_hi_x)
                    {
                        if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                        if (vfrac(i+1,j+1,k) > 0.) nbor(i,j,k,8) = 1;
                    }
                    if (allow_hi_y)
                        if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                }
            }

            if (apy(i,j,k) > 0.)
            {
                if (fcy(i,j,k,0) <= 0.)
                {
                    if (allow_lo_x)
                    {
                        if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                        if (allow_lo_y)
                            if (vfrac(i-1,j-1,k) > 0.) nbor(i,j,k,0) = 1;
                    }
                    if (allow_lo_y)
                        if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                } else {
                    if (allow_hi_x)
                    {
                        if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                        if (allow_lo_y)
                            if (vfrac(i+1,j-1,k) > 0.) nbor(i,j,k,2) = 1;
                    }
                    if (allow_lo_y)
                        if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                }
            }

            if (apy(i,j+1,k) > 0.)
            {
                if (fcy(i,j+1,k,0) <= 0.)
                {
                    if (allow_lo_x)
                    {
                        if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                        if (allow_hi_y)
                            if (vfrac(i-1,j+1,k) > 0.) nbor(i,j,k,6) = 1;
                    }
                    if (allow_hi_y)
                        if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                } else {
                    if (allow_hi_x)
                    {
                        if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                        if (allow_hi_y)
                            if (vfrac(i+1,j+1,k) > 0.) nbor(i,j,k,8) = 1;
                    }
                    if (allow_hi_y)
                        if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                }
            }

            for (int jj = -1; jj <= 1; jj++)  
            for (int ii = -1; ii <= 1; ii++)  
            {
                int index = (jj+1)*3 + (ii+1);
                if (nbor(i,j,k,index) == 1)
                {
                    // amrex::Print() << IntVect(ii,jj) << " is connected with vol " << vfrac(i+ii,j+jj,k) << std::endl;
                    nbhd_vol(i,j,k) += vfrac(i+ii,j+jj,k);
                    nrs(i+ii,j+jj,k) += 1.;
                }
            }
            // amrex::Print() << "VOL IN NBOR OF CELL " << IntVect(i,j) << " " << nbhd_vol(i,j,k) << std::endl;
        }
      }
    });

#if 0
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (nrs(i,j,k) > 1 and i < 10 )
        {
            amrex::Print() << "NRS GE 1:  " << IntVect(i,j) << " " << nrs(i,j,k) << std::endl;
        }
    });
#endif


    // Define xhat,yhat (from Berger and Guliani) 
    amrex::ParallelFor(bxg2,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (vfrac(i,j,k) > 0.5)
        {
            cent_hat(i,j,k,0) = ccent(i,j,k,0);
            cent_hat(i,j,k,1) = ccent(i,j,k,1);

        } else if (vfrac(i,j,k) > 0.0) {

            for (int jj = -1; jj <= 1; jj++)  
            for (int ii = -1; ii <= 1; ii++)  
            {
                int index = (jj+1)*3 + (ii+1);
                if (nbor(i,j,k,index) == 1)
                {
                    int r = i+ii;
                    int s = j+jj;
                    cent_hat(i,j,k,0) += (ccent(r,s,k,0) + ii) * vfrac(r,s,k) / nrs(r,s,k);
                    cent_hat(i,j,k,1) += (ccent(r,s,k,1) + jj) * vfrac(r,s,k) / nrs(r,s,k);
                }
                cent_hat(i,j,k,0) /= nbhd_vol(i,j,k);
                cent_hat(i,j,k,1) /= nbhd_vol(i,j,k);
            }
        } else {
            cent_hat(i,j,k,0) = 0.;
            cent_hat(i,j,k,1) = 0.;
        }
    });

    // Define Qhat (from Berger and Guliani)
    amrex::ParallelFor(bx, ncomp,  
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (vfrac(i,j,k) > 0.5)
        {
            soln_hat(i,j,k,n) = dUdt_in(i,j,k,n);

        } else if (vfrac(i,j,k) > 0.0) {

            for (int jj = -1; jj <= 1; jj++)  
            for (int ii = -1; ii <= 1; ii++)  
            {
                int index = (jj+1)*3 + (ii+1);
                if (nbor(i,j,k,index) == 1)
                {
                    soln_hat(i,j,k,n) += dUdt_in(i+ii,j+jj,k,n) * vfrac(i+ii,j+jj,k) / nrs(i+ii,j+jj,k);
                }
                soln_hat(i,j,k,n) /= nbhd_vol(i,j,k);
            }
        } else {
            soln_hat(i,j,k,n) = 0.; // NOTE -- we shouldn't end up using this .... but lets check later
        }
    });

    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        dUdt(i,j,k,n) = 0;
    });

    for (int n = 0; n < ncomp; n++)
    {
        amrex::ParallelFor(bxg1,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (vfrac(i,j,k) > 0.0)
            {
                const auto& slopes_eb = amrex_lim_slopes_eb(i,j,k,n,soln_hat,cent_hat,
                                                            AMREX_D_DECL(fcx,fcy,fcz), flag);
                slopes_hat(i,j,k,0) = slopes_eb[0];
                slopes_hat(i,j,k,1) = slopes_eb[1];
            } else {
                slopes_hat(i,j,k,0) = 0.; // NOTE -- we shouldn't end up using this .... but lets check later
                slopes_hat(i,j,k,1) = 0.; // NOTE -- we shouldn't end up using this .... but lets check later
            }
        });

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            for (int jj = -1; jj <= 1; jj++)  
            for (int ii = -1; ii <= 1; ii++)  
            {
                int index = (jj+1)*3 + (ii+1);
                if (nbor(i,j,k,index) == 1)
                {
                    int r = i+ii;
                    int s = j+jj;
                    if (r < 0 or s < 0) 
                      amrex::Print() << "ACCESSING OUT OF BOUNDS: " << IntVect(i,j) << " " << IntVect(r,s) << std::endl;
                    dUdt(r,s,k,n) += (soln_hat(i,j,k,n) + slopes_hat(i,j,k,0) * (ccent(r,s,k,0)-cent_hat(i,j,k,0))
                                                        + slopes_hat(i,j,k,1) * (ccent(r,s,k,1)-cent_hat(i,j,k,1)) );
                if (r == 16 and s == 90) amrex::Print() << "ADDING TO (16,90) " << index << std::endl;
   
                }
            }

            dUdt(i,j,k,n) /= nrs(i,j,k);

//          if (i > 10 and i < 15 and vfrac(i,j,k) > 0.) 
//          if (std::abs(dUdt(i,j,k,n)) > 1.e-8) 
            if ( i == 16 and vfrac(i,j,k) > 0.)
               amrex::Print() << "CONV " << IntVect(i,j) << " " << n << " " << vfrac(i,j,k) << " " << dUdt_in(i,j,k,n) << " " << dUdt(i,j,k,n) << std::endl;
        });
    }
}
#endif
