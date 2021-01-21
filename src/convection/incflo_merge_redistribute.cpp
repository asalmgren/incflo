#ifdef AMREX_USE_EB

#include <Redistribution.H>
#include <AMReX_EB_slopes_K.H>

using namespace amrex;

void
redistribution::merge_redistribute_update (
                       Box const& bx, int ncomp,
                       Array4<Real>       const& dUdt_out,
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

    const Real small_vel = 1.e-8;

    const auto& is_periodic_x = lev_geom.isPeriodic(0);
    const auto& is_periodic_y = lev_geom.isPeriodic(1);
#if (AMREX_SPACEDIM == 3)
    const auto& is_periodic_z = lev_geom.isPeriodic(2);
#endif

    amrex::Print() << " IN MERGE_REDISTRIBUTE DOING BOX " << bx << " with ncomp " << ncomp << std::endl;

    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (vfrac(i,j,k) > 0.0)
        {
            dUdt_out(i,j,k,n) = dUdt_in(i,j,k,n);
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
           Real sum_upd = vfrac(i,j,k)*(dUdt_in(i,j,k,n));

           dUdt_out(i,j,k,n) = dUdt_in(i,j,k,n);

           // At lo-x or hi-x outflow face
           if ( (i == domain.smallEnd(0) && !is_periodic_x && apx(i  ,j,k)*umac(i  ,j,k) < -small_vel) ||
                (i == domain.bigEnd(0)   && !is_periodic_x && apx(i+1,j,k)*umac(i+1,j,k) >  small_vel) )
           {
               if (vfrac(i,j-1,k) > 0.)
               {
                 sum_vol += vfrac(i,j-1,k);
                 sum_upd += vfrac(i,j-1,k)*(dUdt_in(i,j-1,k,n));
                 num_merge++;
               } else if (vfrac(i,j+1,k) > 0.)
               {
                 sum_vol += vfrac(i,j+1,k);
                 sum_upd += vfrac(i,j+1,k)*(dUdt_in(i,j+1,k,n));
                 num_merge++;
               }
           }
           // At lo-y or hi-y outflow face
           else if ( (j == domain.smallEnd(1) && !is_periodic_y && apy(i,j  ,k)*vmac(i,j  ,k) < -small_vel) ||
                     (j == domain.bigEnd(1)   && !is_periodic_y && apy(i,j+1,k)*vmac(i,j+1,k) >  small_vel) )
           {
               if (vfrac(i-1,j,k) > 0.)
               {
                 sum_vol += vfrac(i-1,j,k);
                 sum_upd += vfrac(i-1,j,k)*(dUdt_in(i-1,j,k,n));
                 num_merge++;

               } else if (vfrac(i+1,j,k) > 0.)
               {
                 sum_vol += vfrac(i+1,j,k);
                 sum_upd += vfrac(i+1,j,k)*(dUdt_in(i+1,j,k,n));
                 num_merge++;
               }
#if (AMREX_SPACEDIM == 3)
           // At lo-z or hi-z outflow face
           } else if ( (k == domain.smallEnd(2) && !is_periodic_z && apz(i,j,k  )*wmac(i,j,k  ) < -small_vel) ||
                       (k == domain.bigEnd(2)   && !is_periodic_z && apz(i,j,k+1)*wmac(i,j,k+1) >  small_vel) )
           {
               if (vfrac(i,j,k-1) > 0.)
               {
                 sum_vol += vfrac(i-1,j,k);
                 sum_upd += vfrac(i-1,j,k)*(dUdt_in(i-1,j,k,n));
                 num_merge++;

               } else if (vfrac(i+1,j,k) > 0.)
               {
                 sum_vol += vfrac(i+1,j,k);
                 sum_upd += vfrac(i+1,j,k)*(dUdt_in(i+1,j,k,n));
                 num_merge++;
               }
#endif
           } else {

                if ( allow_lo_x && apx(i,j,k) > 0. && umac(i,j,k) < -small_vel &&
                     std::abs(dUdt_in(i-1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through left face
                {
                  sum_vol += vfrac(i-1,j,k);
                  sum_upd += vfrac(i-1,j,k)*(dUdt_in(i-1,j,k,n));
                  num_merge++;
                }
                if ( allow_hi_x && apx(i+1,j,k) > 0. && umac(i+1,j,k) > small_vel &&
                     std::abs(dUdt_in(i+1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
                {
                  sum_vol += vfrac(i+1,j,k);
                  sum_upd += vfrac(i+1,j,k)*(dUdt_in(i+1,j,k,n));
                  num_merge++;
                }
                if ( allow_lo_y && apy(i,j,k) > 0. && vmac(i,j,k) < -small_vel &&
                     std::abs(dUdt_in(i,j-1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through bottom face
                {
                  sum_vol += vfrac(i,j-1,k);
                  sum_upd += vfrac(i,j-1,k)*(dUdt_in(i,j-1,k,n));
                  num_merge++;
                }
                if (allow_hi_y && apy(i,j+1,k) > 0. && vmac(i,j+1,k) > small_vel &&
                     std::abs(dUdt_in(i,j+1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
                {
                  sum_vol += vfrac(i,j+1,k);
                  sum_upd += vfrac(i,j+1,k)*(dUdt_in(i,j+1,k,n));
                  num_merge++;
                }
#if (AMREX_SPACEDIM == 3)
                if ( allow_lo_z && apz(i,j,k) > 0. && wmac(i,j,k) < -small_vel &&
                     std::abs(dUdt_in(i,j,k-1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through bottom face
                {
                  sum_vol += vfrac(i,j,k-1);
                  sum_upd += vfrac(i,j,k-1)*(dUdt_in(i,j,k-1,n));
                  num_merge++;
                }
                if (allow_hi_z && apz(i,j+1,k) > 0. && wmac(i,j,k+1) > small_vel &&
                     std::abs(dUdt_in(i,j,k+1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
                {
                  sum_vol += vfrac(i,j,k+1);
                  sum_upd += vfrac(i,j,k+1)*(dUdt_in(i,j,k+1,n));
                  num_merge++;
                }
#endif
           }
  
           Real avg_update = sum_upd / sum_vol;

           if (num_merge > 1) 
           {
              amrex::Print() << "Not sure what to do here" << IntVect(AMREX_D_DECL(i,j,k)) << std::endl;
              amrex::Print() << "Cell has volfrac        " << vfrac(i,j,k) << std::endl;
              amrex::Print() << "Cell has x-areas        " << apx(i,j,k) << " " << apx(i+1,j,k) << std::endl;
              amrex::Print() << "Cell has y-areas        " << apy(i,j,k) << " " << apy(i,j+1,k) << std::endl;
#if (AMREX_SPACEDIM == 3)
              amrex::Print() << "Cell has z-areas        " << apz(i,j,k) << " " << apz(i,j,k+1) << std::endl;
#endif
              amrex::Print() << "Cell has x-vels         " << umac(i,j,k) << " " << umac(i+1,j,k) << std::endl;
              amrex::Print() << "Cell has y-vels         " << vmac(i,j,k) << " " << vmac(i,j+1,k) << std::endl;
#if (AMREX_SPACEDIM == 3)
              amrex::Print() << "Cell has z-vels         " << wmac(i,j,k) << " " << wmac(i,j,k+1) << std::endl;
#endif
              amrex::Abort(0);
           }

           // At lo-x or hi-x outflow face
           if ( (i == domain.smallEnd(0) && !is_periodic_x && apx(i  ,j,k)*umac(i  ,j,k) < -small_vel) ||
                (i == domain.bigEnd(0)   && !is_periodic_x && apx(i+1,j,k)*umac(i+1,j,k) >  small_vel) )
           {
               if (vfrac(i,j-1,k) > 0.)
               {
                   dUdt_out(i,j  ,k,n) = avg_update;
                   dUdt_out(i,j-1,k,n) = avg_update;

               } else if (vfrac(i,j+1,k) > 0.)
               {
                   dUdt_out(i,j  ,k,n) = avg_update;
                   dUdt_out(i,j+1,k,n) = avg_update;
               }
           }
           else if ( (j == domain.smallEnd(1) && !is_periodic_y && apy(i,j  ,k)*vmac(i,j  ,k) < -small_vel) ||
                     (j == domain.bigEnd(1)   && !is_periodic_y && apy(i,j+1,k)*vmac(i,j+1,k) >  small_vel) )
           {
               if (vfrac(i-1,j,k) > 0.)
               {
                   dUdt_out(i  ,j,k,n) = avg_update;
                   dUdt_out(i-1,j,k,n) = avg_update;

               } else if (vfrac(i+1,j,k) > 0.)
               {
                   dUdt_out(i  ,j,k,n) = avg_update;
                   dUdt_out(i+1,j,k,n) = avg_update;
               }
#if (AMREX_SPACEDIM == 3)
           } else if ( (k == domain.smallEnd(2) && !is_periodic_z && apz(i,j,k  )*wmac(i,j,k  ) < -small_vel) ||
                       (k == domain.bigEnd(2)   && !is_periodic_z && apz(i,j,k+1)*wmac(i,j,k+1) >  small_vel) )
           {
               if (vfrac(i,j,k-1) > 0.)
               {
                   dUdt_out(i  ,j,k,n) = avg_update;
                   dUdt_out(i-1,j,k,n) = avg_update;

               } else if (vfrac(i+1,j,k) > 0.)
               {
                   dUdt_out(i  ,j,k,n) = avg_update;
                   dUdt_out(i+1,j,k,n) = avg_update;
               }
#endif
           } else {

               if ( allow_lo_x && apx(i,j,k) > 0. && umac(i,j,k) < -small_vel &&
                    std::abs(dUdt_in(i-1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through left face
               {
                 dUdt_out(i  ,j,k,n) = avg_update;
                 dUdt_out(i-1,j,k,n) = avg_update;
               }

               if (allow_hi_x && apx(i+1,j,k) > 0. && umac(i+1,j,k) > small_vel &&
                   std::abs(dUdt_in(i+1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
               {
                 dUdt_out(i  ,j,k,n) = avg_update;
                 dUdt_out(i+1,j,k,n) = avg_update;
               }

               if (allow_lo_y && apy(i,j,k) > 0. && vmac(i,j,k) < -small_vel &&
                    std::abs(dUdt_in(i,j-1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through bottom face
               {
                 dUdt_out(i,j  ,k,n) = avg_update;
                 dUdt_out(i,j-1,k,n) = avg_update;
               }

               if (allow_hi_y && apy(i,j+1,k) > 0. && vmac(i,j+1,k) > small_vel &&
                    std::abs(dUdt_in(i,j+1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through top face
               {
                  dUdt_out(i,j  ,k,n) = avg_update;
                  dUdt_out(i,j+1,k,n) = avg_update;
               }

#if (AMREX_SPACEDIM == 3)
               if (allow_lo_z && apz(i,j,k) > 0. && wmac(i,j,k) < -small_vel &&
                    std::abs(dUdt_in(i,j,k-1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through down face
               {
                 dUdt_out(i,j,k  ,n) = avg_update;
                 dUdt_out(i,j,k-1,n) = avg_update;
               }

               if (allow_hi_z && apz(i,j,k+1) > 0. && wmac(i,j,k+1) > small_vel &&
                    std::abs(dUdt_in(i,j,k+1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through up face
               {
                  dUdt_out(i,j,k  ,n) = avg_update;
                  dUdt_out(i,j,k+1,n) = avg_update;
               }
#endif
           }
       }
    });

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
            sum2 += vfrac(i,j,k)*dUdt_out(i,j,k,n);
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

void
redistribution::merge_redistribute_full (
                       Box const& bx, int ncomp,
                       Array4<Real>       const& dUdt_out,
                       Array4<Real const> const& dUdt_in,
                       Array4<Real const> const& U_in,
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
                       Geometry& lev_geom, Real dt)
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
            dUdt_out(i,j,k,n) = dUdt_in(i,j,k,n);
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

       Real dt_inv = 1./dt;

       // 
       // Only do merging for cells 
       //    1) with vfrac < 1/2 and in "outflow" directions
       //    2) in "outflow" directions
       //    3) if the merging will decrease the change in the small cell
       // 
       if (vfrac(i,j,k) > 0.0 and vfrac(i,j,k) < 0.5)
       {
           int num_merge = 0; Real sum_vol = vfrac(i,j,k); 
           Real sum_upd = vfrac(i,j,k)*(dUdt_in(i,j,k,n)+U_in(i,j,k,n)*dt_inv);

           dUdt_out(i,j,k,n) = dUdt_in(i,j,k,n);

           if ( i < domain.bigEnd(0) or (i == domain.bigEnd(0) and apx(i+1,j,k) == 0.) )
           {
                if ( allow_lo_x && apx(i,j,k) > 0. && umac(i,j,k) < 0. &&
                     std::abs(dUdt_in(i-1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through left face
                {
                  sum_vol += vfrac(i-1,j,k);
                  sum_upd += vfrac(i-1,j,k)*(dUdt_in(i-1,j,k,n)+U_in(i-1,j,k,n)*dt_inv);
                  num_merge++;
                }
                if ( allow_hi_x && apx(i+1,j,k) > 0. && umac(i+1,j,k) > 0. &&
                     std::abs(dUdt_in(i+1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
                {
                  if (i == 15 and j == 95) amrex::Print() << "LEFT RIGHT DUDT " << dUdt_in(i,j,k,n) << " " << dUdt_in(i+1,j,k,n) << std::endl;
                  sum_vol += vfrac(i+1,j,k);
                  sum_upd += vfrac(i+1,j,k)*(dUdt_in(i+1,j,k,n)+U_in(i+1,j,k,n)*dt_inv);
                  num_merge++;
                }
                if ( allow_lo_y && apy(i,j,k) > 0. && vmac(i,j,k) < 0. &&
                     std::abs(dUdt_in(i,j-1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through bottom face
                {
                  sum_vol += vfrac(i,j-1,k);
                  sum_upd += vfrac(i,j-1,k)*(dUdt_in(i,j-1,k,n)+U_in(i,j-1,k,n)*dt_inv);
                  num_merge++;
                }
                if (allow_hi_y && apy(i,j+1,k) > 0. && vmac(i,j+1,k) > 0. &&
                     std::abs(dUdt_in(i,j+1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
                {
                  sum_vol += vfrac(i,j+1,k);
                  sum_upd += vfrac(i,j+1,k)*(dUdt_in(i,j+1,k,n)+U_in(i,j+1,k,n)*dt_inv);
                  num_merge++;
                }
#if (AMREX_SPACEDIM == 3)
                if ( allow_lo_z && apz(i,j,k) > 0. && wmac(i,j,k) < 0. &&
                     std::abs(dUdt_in(i,j,k-1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through bottom face
                {
                  sum_vol += vfrac(i,j,k-1);
                  sum_upd += vfrac(i,j,k-1)*(dUdt_in(i,j,k-1,n)+U_in(i,j,k-1,n)*dt_inv);
                  num_merge++;
                }
                if (allow_hi_z && apz(i,j+1,k) > 0. && wmac(i,j,k+1) > 0. &&
                     std::abs(dUdt_in(i,j,k+1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
                {
                  sum_vol += vfrac(i,j,k+1);
                  sum_upd += vfrac(i,j,k+1)*(dUdt_in(i,j,k+1,n)+U_in(i,j,k,n+1)*dt_inv);
                  num_merge++;
                }
#endif

           } else if (apx(i+1,j,k) > 0.) { // At outflow face

               if (vfrac(i,j-1,k) > 0.)
               {
                 sum_vol += vfrac(i,j-1,k);
                 sum_upd += vfrac(i,j-1,k)*(dUdt_in(i,j-1,k,n)+U_in(i,j-1,k,n)*dt_inv);
                 num_merge++;

               } else if (vfrac(i,j+1,k) > 0.)
               {
                 sum_vol += vfrac(i,j+1,k);
                 sum_upd += vfrac(i,j+1,k)*(dUdt_in(i,j+1,k,n)+U_in(i,j-1,k,n)*dt_inv);
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
                 dUdt_out(i  ,j,k,n) = avg_update - U_in(i,j,k,n)*dt_inv;
                 dUdt_out(i-1,j,k,n) = avg_update - U_in(i-1,j,k,n)*dt_inv;
               }

               if (allow_hi_x && apx(i+1,j,k) > 0. && umac(i+1,j,k) > 0. &&
                   std::abs(dUdt_in(i+1,j,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through right face
               {
                 dUdt_out(i  ,j,k,n) = avg_update - U_in(i,j,k,n)*dt_inv;
                 dUdt_out(i+1,j,k,n) = avg_update - U_in(i+1,j,k,n)*dt_inv;
               }

               if (allow_lo_y && apy(i,j,k) > 0. && vmac(i,j,k) < 0. &&
                    std::abs(dUdt_in(i,j-1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through bottom face
               {
                 dUdt_out(i,j  ,k,n) = avg_update - U_in(i,j,k,n)*dt_inv;
                 dUdt_out(i,j-1,k,n) = avg_update - U_in(i,j-1,k,n)*dt_inv;
               }

               if (allow_hi_y && apy(i,j+1,k) > 0. && vmac(i,j+1,k) > 0. &&
                    std::abs(dUdt_in(i,j+1,k,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through top face
               {
                  dUdt_out(i,j  ,k,n) = avg_update - U_in(i,j,k,n)*dt_inv;
                  dUdt_out(i,j+1,k,n) = avg_update - U_in(i,j+1,k,n)*dt_inv;
               }

#if (AMREX_SPACEDIM == 3)
               if (allow_lo_z && apz(i,j,k) > 0. && wmac(i,j,k) < 0. &&
                    std::abs(dUdt_in(i,j,k-1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through down face
               {
                 dUdt_out(i,j,k  ,n) = avg_update - U_in(i,j,k,n)*dt_inv;
                 dUdt_out(i,j,k-1,n) = avg_update - U_in(i,j,k-1,n)*dt_inv;
               }

               if (allow_hi_z && apz(i,j,k+1) > 0. && wmac(i,j,k+1) > 0. &&
                    std::abs(dUdt_in(i,j,k+1,n)) < std::abs(dUdt_in(i,j,k,n)) ) // Outflow through up face
               {
                  dUdt_out(i,j,k  ,n) = avg_update - U_in(i,j,k,n)*dt_inv;
                  dUdt_out(i,j,k+1,n) = avg_update - U_in(i,j,k+1,n)*dt_inv;
               }
#endif

           } else if (apx(i+1,j,k) > 0.) { // At outflow face

               if (vfrac(i,j-1,k) > 0.)
               {
                   dUdt_out(i,j  ,k,n) = avg_update - U_in(i,j,k,n)*dt_inv;
                   dUdt_out(i,j-1,k,n) = avg_update - U_in(i,j-1,k,n)*dt_inv;

               } 
               else if (vfrac(i,j+1,k) > 0.) 
               {
                   dUdt_out(i,j  ,k,n) = avg_update - U_in(i,j,k,n)*dt_inv;
                   dUdt_out(i,j+1,k,n) = avg_update - U_in(i,j+1,k,n)*dt_inv;
               }
           }
       }
    });

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
            sum2 += vfrac(i,j,k)*dUdt_out(i,j,k,n);
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
