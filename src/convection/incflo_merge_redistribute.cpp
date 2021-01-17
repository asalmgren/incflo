#include <Redistribution.H>
#include <AMReX_EB_slopes_K.H>

using namespace amrex;

#ifdef AMREX_USE_EB
void
redistribution::merge_redistribute_eb (
                       Box const& bx, int ncomp,
                       Array4<Real> const& dUdt,
                       Array4<Real const> const& dUdt_in,
                       AMREX_D_DECL(Array4<Real const> const& umac,
                                    Array4<Real const> const& vmac,
                                    Array4<Real const> const& wmac),
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

    amrex::Print() << " IN MERGE_REDISTRIBUTE DOING BOX " << bx << " with ncomp " << ncomp << std::endl;

    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (vfrac(i,j,k) > 0.0)
        {
            dUdt(i,j,k,n) = dUdt_in(i,j,k,n);
        }
    });

    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {

       // bool allow_lo_x = (i > domain.smallEnd(0) || is_periodic_x);
       // bool allow_lo_y = (j > domain.smallEnd(1) || is_periodic_y);
       // bool allow_hi_x = (i < domain.bigEnd(0)   || is_periodic_x);
       // bool allow_hi_y = (j < domain.bigEnd(1)   || is_periodic_y);

       bool allow_lo_x = true;
       bool allow_lo_y = true;
       bool allow_lo_z = true;
       bool allow_hi_x = true;
       bool allow_hi_y = true;
       bool allow_hi_z = true;

       // 
       // Only do merging for cells 
       //    1) with vfrac < 1/2 and in "outflow" directions
       //    2) in "outflow" directions
       //    3) if the merging will decrease the change in the small cell
       // 
       if (vfrac(i,j,k) > 0.0 and vfrac(i,j,k) < 0.5)
       {
           int num_merge = 0; Real sum_vol = vfrac(i,j,k); 
           Real sum_upd = vfrac(i,j,k)*dUdt_in(i,j,k,n);

           dUdt(i,j,k,n) = dUdt_in(i,j,k,n);

           if ( i < domain.bigEnd(0) or (i == domain.bigEnd(0) and apx(i+1,j,k) == 0.) )
           {
                if ( allow_lo_x && apx(i,j,k) > 0. && umac(i,j,k) < 0. &&
                     std::abs(dUdt_in(i-1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through left face
                {
                  sum_vol += vfrac(i-1,j,k);
                  sum_upd += vfrac(i-1,j,k)*dUdt_in(i-1,j,k,n);
                  num_merge++;
                }
                if ( allow_hi_x && apx(i+1,j,k) > 0. && umac(i+1,j,k) > 0. &&
                     std::abs(dUdt_in(i+1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
                {
                  sum_vol += vfrac(i+1,j,k);
                  sum_upd += vfrac(i+1,j,k)*dUdt_in(i+1,j,k,n);
                  num_merge++;
                }
                if ( allow_lo_y && apy(i,j,k) > 0. && vmac(i,j,k) < 0. &&
                     std::abs(dUdt_in(i,j-1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through bottom face
                {
                  sum_vol += vfrac(i,j-1,k);
                  sum_upd += vfrac(i,j-1,k)*dUdt_in(i,j-1,k,n);
                  num_merge++;
                }
                if (allow_hi_y && apy(i,j+1,k) > 0. && vmac(i,j+1,k) > 0. &&
                     std::abs(dUdt_in(i,j+1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
                {
                  sum_vol += vfrac(i,j+1,k);
                  sum_upd += vfrac(i,j+1,k)*dUdt_in(i,j+1,k,n);
                  num_merge++;
                }
#if (AMREX_SPACEDIM == 3)
                if ( allow_lo_z && apz(i,j,k) > 0. && wmac(i,j,k) < 0. &&
                     std::abs(dUdt_in(i,j,k-1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through bottom face
                {
                  sum_vol += vfrac(i,j,k-1);
                  sum_upd += vfrac(i,j,k-1)*dUdt_in(i,j,k-1,n);
                  num_merge++;
                }
                if (allow_hi_z && apz(i,j+1,k) > 0. && wmac(i,j,k+1) > 0. &&
                     std::abs(dUdt_in(i,j,k+1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
                {
                  sum_vol += vfrac(i,j,k+1);
                  sum_upd += vfrac(i,j,k+1)*dUdt_in(i,j,k+1,n);
                  num_merge++;
                }
#endif

           } else if (apx(i+1,j,k) > 0.) { // At outflow face

               if (vfrac(i,j-1,k) > 0.)
               {
                 sum_vol += vfrac(i,j-1,k);
                 sum_upd += vfrac(i,j-1,k)*dUdt_in(i,j-1,k,n);
                 num_merge++;

               } else if (vfrac(i,j+1,k) > 0.)
               {
                 sum_vol += vfrac(i,j+1,k);
                 sum_upd += vfrac(i,j+1,k)*dUdt_in(i,j+1,k,n);
                 num_merge++;
               }
           }
  
           Real avg_update = sum_upd / sum_vol;

           if (num_merge > 1) 
           {
              amrex::Print() << "Not sure what to do here" << IntVect(AMREX_D_DECL(i,j,k)) << " " 
                             << vfrac(i,j,k) << std::endl;
              amrex::Abort(0);
           }

           if ( i < domain.bigEnd(0) or (i == domain.bigEnd(0) and apx(i+1,j,k) == 0.) )
           {
               if ( allow_lo_x && apx(i,j,k) > 0. && umac(i,j,k) < 0. &&
                    std::abs(dUdt_in(i-1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through left face
               {
                 dUdt(i  ,j,k,n) = avg_update;
                 dUdt(i-1,j,k,n) = avg_update;
               }

               if (allow_hi_x && apx(i+1,j,k) > 0. && umac(i+1,j,k) > 0. &&
                   std::abs(dUdt_in(i+1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
               {
                 dUdt(i  ,j,k,n) = avg_update;
                 dUdt(i+1,j,k,n) = avg_update;
               }

               if (allow_lo_y && apy(i,j,k) > 0. && vmac(i,j,k) < 0. &&
                    std::abs(dUdt_in(i,j-1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through bottom face
               {
                 dUdt(i,j  ,k,n) = avg_update;
                 dUdt(i,j-1,k,n) = avg_update;
               }

               if (allow_hi_y && apy(i,j+1,k) > 0. && vmac(i,j+1,k) > 0. &&
                    std::abs(dUdt_in(i,j+1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through top face
               {
                  dUdt(i,j  ,k,n) = avg_update;
                  dUdt(i,j+1,k,n) = avg_update;
               }

#if (AMREX_SPACEDIM == 3)
               if (allow_lo_z && apz(i,j,k) > 0. && wmac(i,j,k) < 0. &&
                    std::abs(dUdt_in(i,j,k-1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through down face
               {
                 dUdt(i,j,k  ,n) = avg_update;
                 dUdt(i,j,k-1,n) = avg_update;
               }

               if (allow_hi_z && apz(i,j,k+1) > 0. && wmac(i,j,k+1) > 0. &&
                    std::abs(dUdt_in(i,j,k+1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through up face
               {
                  dUdt(i,j,k  ,n) = avg_update;
                  dUdt(i,j,k+1,n) = avg_update;
               }
#endif

           } else if (apx(i+1,j,k) > 0.) { // At outflow face

               if (vfrac(i,j-1,k) > 0.)
               {
                   dUdt(i,j  ,k,n) = avg_update;
                   dUdt(i,j-1,k,n) = avg_update;

               } 
               else if (vfrac(i,j+1,k) > 0.) 
               {
                   dUdt(i,j  ,k,n) = avg_update;
                   dUdt(i,j+1,k,n) = avg_update;
               }
           }
       }
    });

    // PRINTING ONLY
#if 0
    for (int n = 0; n < ncomp; n++)
    {
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
    }
#endif

    //
    // This tests whether the redistribution procedure was conservative
    //
    { // STRT:SUM OF FINAL DUDT

        for (int n = 0; n < ncomp; n++) 
        {
          Real sum1(0);
          Real sum2(0);
          for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); k++)  
          for (int j = domain.smallEnd(1); j <= domain.bigEnd(1); j++)  
          for (int i = domain.smallEnd(0); i <= domain.bigEnd(0); i++)  
          {
            sum1 += vfrac(i,j,k)*dUdt_in(i,j,k,n);
            sum2 += vfrac(i,j,k)*dUdt(i,j,k,n);
          }
          if (std::abs(sum1-sum2) > 1.e-8 * sum1 && std::abs(sum1-sum2) > 1.e-8)
          {
            amrex::Print() << " TESTING COMPONENT " << n << std::endl; 
            amrex::Print() << " SUMS DO NOT MATCH " << sum1 << " " << sum2 << std::endl;
            amrex::Abort(0);
          }
        }
    } //  END:SUM OF FINAL DUDT
}
#endif
