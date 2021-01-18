#ifdef AMREX_USE_EB
#include <Redistribution.H>
#include <AMReX_EB_slopes_K.H>

using namespace amrex;

void 
redistribution::state_redistribute_eb (
                       Box const& bx, int ncomp,
                       Array4<Real> const& dUdt,
                       Array4<Real const> const& dUdt_in,
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

    const auto& is_periodic_x = lev_geom.isPeriodic(0);
    const auto& is_periodic_y = lev_geom.isPeriodic(1);

    amrex::Print() << " IN STATE_REDISTRIBUTE DOING BOX " << bx << " with ncomp " << ncomp << std::endl;

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
    nrs_fab.setVal(0.0);
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
          // Always include the cell itself
          nbor(i,j,k,4) = 1;

          if (vfrac(i,j,k) < 0.5)
          {
            // We only include cells into a neighborhood if they are in the interior
            //    or in periodic ghost cells
            bool allow_lo_x = (i > domain.smallEnd(0) || is_periodic_x);
            bool allow_lo_y = (j > domain.smallEnd(1) || is_periodic_y);
            bool allow_hi_x = (i < domain.bigEnd(0)   || is_periodic_x);
            bool allow_hi_y = (j < domain.bigEnd(1)   || is_periodic_y);

            if (apx(i,j,k) > 0. && allow_lo_x)
            {
                if (fcx(i,j,k,0) <= 0. && allow_lo_y)
                {
                    if (vfrac(i-1,j-1,k) > 0.) nbor(i,j,k,0) = 1;
                    if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                } else if (allow_hi_y) {
                    if (vfrac(i-1,j+1,k) > 0.) nbor(i,j,k,6) = 1;
                    if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                }
                if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
            }

            if (apx(i+1,j,k) > 0. && allow_hi_x)
            {
                if (fcx(i+1,j,k,0) <= 0. && allow_lo_y)
                {
                    if (vfrac(i+1,j-1,k) > 0.) nbor(i,j,k,2) = 1;
                    if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                } else if (allow_hi_y) {
                    if (vfrac(i+1,j+1,k) > 0.) nbor(i,j,k,8) = 1;
                    if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                }
                if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
            }

            if (apy(i,j,k) > 0. && allow_lo_y)
            {
                if (fcy(i,j,k,0) <= 0. && allow_lo_x) {
                    if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                    if (vfrac(i-1,j-1,k) > 0.) nbor(i,j,k,0) = 1;
                } else if (allow_hi_x) {
                    if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                    if (vfrac(i+1,j-1,k) > 0.) nbor(i,j,k,2) = 1;
                }
                if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
            }

            if (apy(i,j+1,k) > 0. && allow_hi_y)
            {
                if (fcy(i,j+1,k,0) <= 0. && allow_lo_x)
                {
                    if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                    if (vfrac(i-1,j+1,k) > 0.) nbor(i,j,k,6) = 1;
                } else if (allow_hi_x) {
                    if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                    if (vfrac(i+1,j+1,k) > 0.) nbor(i,j,k,8) = 1;
                }
                if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
            }
        }

        for (int jj = -1; jj <= 1; jj++)  
        for (int ii = -1; ii <= 1; ii++)  
        {
            int index = (jj+1)*3 + (ii+1);
            if (nbor(i,j,k,index) == 1)
            {
                int r = i+ii;
                int s = j+jj;
                nrs(r,s,k) += 1.;
            }
        }
      }
    });

    amrex::ParallelFor(bxg1,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (!flag(i,j,k).isCovered())
        {
            if ( ( (i >= domain.smallEnd(0) && i <= domain.bigEnd(0)) || is_periodic_x ) &&
                 ( (j >= domain.smallEnd(1) && j <= domain.bigEnd(1)) || is_periodic_y ) )
            {
                for (int jj = -1; jj <= 1; jj++)  
                for (int ii = -1; ii <= 1; ii++)  
                {
                    int index = (jj+1)*3 + (ii+1);
                    if (nbor(i,j,k,index) == 1)
                    {
                        int r = i+ii;
                        int s = j+jj;
                        if ( ( (r >= domain.smallEnd(0) && r <= domain.bigEnd(0)) || is_periodic_x ) &&
                             ( (s >= domain.smallEnd(1) && s <= domain.bigEnd(1)) || is_periodic_y ) )
                        nbhd_vol(i,j,k) += vfrac(r,s,k) / nrs(r,s,k);
                    }
                }
            }
        }
    });

    { // STRT:SUM OF VOLUMES
        Real sum1v(0);
        Real sum2v(0);

        for (int i = 0; i <= domain.bigEnd(0); i++)  
        for (int j = 0; j <= domain.bigEnd(1); j++)  
        {
            sum1v += vfrac(i,j,0);
            sum2v += nbhd_vol(i,j,0);
        }
        amrex::Print() << " SUMS OF VOLS " << sum1v << " " << sum2v << std::endl;
    } //  END:SUM OF VOLUMES

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
            }
            cent_hat(i,j,k,0) /= nbhd_vol(i,j,k);
            cent_hat(i,j,k,1) /= nbhd_vol(i,j,k);
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
                    int r = i+ii;
                    int s = j+jj;
                    soln_hat(i,j,k,n) += dUdt_in(r,s,k,n) * vfrac(r,s,k) / nrs(r,s,k);
                }
            }

            soln_hat(i,j,k,n) /= nbhd_vol(i,j,k);
        } else {
            soln_hat(i,j,k,n) = 1.e40; // NOTE -- we shouldn't end up using this 
        }
    });

    { // STRT:SUM OF QHAT
        Real sum1s(0);
        Real sum2s(0);

        for (int i = 0; i <= domain.bigEnd(0); i++)  
        for (int j = 0; j <= domain.bigEnd(1); j++)  
        {
            sum1s += vfrac(i,j,0)*dUdt_in(i,j,0,0);
            sum2s += nbhd_vol(i,j,0)*soln_hat(i,j,0,0);
        }
        amrex::Print() << " SUMS OF QHAT " << sum1s << " " << sum2s << std::endl;
    } //  END:SUM OF QHAT 

#if 1
    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        dUdt(i,j,k,n) = 0;
    });
#endif

    for (int n = 0; n < ncomp; n++)
    {
#if 1
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
                slopes_hat(i,j,k,0) = 1.e40; // NOTE -- we shouldn't end up using this .... but lets check later
                slopes_hat(i,j,k,1) = 1.e40; // NOTE -- we shouldn't end up using this .... but lets check later
            }
        });

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            for (int jj = -1; jj <= 1; jj++)  
            for (int ii = -1; ii <= 1; ii++)  
            {
                // Note every cell is at least in its own neighborhood so this will update every (i,j,k)
                int index = (jj+1)*3 + (ii+1);
                if (nbor(i,j,k,index) == 1)
                {
                    int r = i+ii;
                    int s = j+jj;
                    dUdt(r,s,k,n) += (soln_hat(i,j,k,n) + slopes_hat(i,j,k,0) * (ccent(r,s,k,0)-cent_hat(i,j,k,0))
                                                        + slopes_hat(i,j,k,1) * (ccent(r,s,k,1)-cent_hat(i,j,k,1)) );
                }
            }
        });

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (!flag(i,j,k).isCovered())
            {
                dUdt(i,j,k,n) /= nrs(i,j,k);
            }
        });
#endif

#if 0
        // PRINTING ONLY
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (!flag(i,j,k).isCovered())
            {
                if ( (i >= 15 && i <= 17) && vfrac(i,j,k) > 0. 
                     && (std::abs(dUdt_in(i,j,k,n)) > 1.e-8 || std::abs(dUdt(i,j,k,n)) > 1.e-8) )
                   amrex::Print() << "OLD / NEW CONV " << IntVect(i,j) << " " << vfrac(i,j,k) << 
                        " " << dUdt_in(i,j,k,n) << " " << dUdt(i,j,k,n) << std::endl;
            }
        });
#endif
    }

    //
    // This tests whether the redistribution procedure was conservative
    //
    { // STRT:SUM OF FINAL DUDT
        Real sum1(0);
        Real sum2(0);

        for (int j = 0; j <= domain.bigEnd(1); j++)  
        for (int i = 0; i <= domain.bigEnd(0); i++)  
        {
            sum1 += vfrac(i,j,0)*dUdt_in(i,j,0,0);
            sum2 += vfrac(i,j,0)*dUdt(i,j,0,0);
        }
        if (std::abs(sum1-sum2) > 1.e-8 * sum1 && std::abs(sum1-sum2) > 1.e-8)
           amrex::Print() << " SUMS DO NOT MATCH " << sum1 << " " << sum2 << std::endl;
        else
           amrex::Print() << " SUMS DO     MATCH " << sum1 << " " << sum2 << std::endl;
    } //  END:SUM OF FINAL DUDT
}
#endif
