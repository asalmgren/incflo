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
                       Array4<int> const& itracker,
                       Geometry& lev_geom)
{
    const Box domain = lev_geom.Domain();

    const Real small_vel  = 1.e-8;

    // We will use small_norm as an off just to break the tie when at 45 degrees ...
    const Real small_norm = 1.e-8;

    const auto& is_periodic_x = lev_geom.isPeriodic(0);
    const auto& is_periodic_y = lev_geom.isPeriodic(1);
#if (AMREX_SPACEDIM == 3)
    const auto& is_periodic_z = lev_geom.isPeriodic(2);
#endif

    amrex::Print() << " IN MERGE_REDISTRIBUTE DOING BOX " << bx << " with ncomp " << ncomp << std::endl;

    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (vfrac(i,j,k) > 0.0)
        {
            for (int n = 0; n < ncomp; n++)
                dUdt_out(i,j,k,n) = dUdt_in(i,j,k,n);
        } else {
            for (int n = 0; n < ncomp; n++)
                dUdt_out(i,j,k,n) = 1.e100;
        } 
    });

    amrex::ParallelFor(bx, 
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
       if (vfrac(i,j,k) > 0.0)
       {
         if (vfrac(i,j,k) < 0.5)
         {
           Real apnorm, apnorm_inv;
           const Real dapx = apx(i+1,j  ,k  ) - apx(i,j,k);
           const Real dapy = apy(i  ,j+1,k  ) - apy(i,j,k);
#if (AMREX_SPACEDIM == 2)
           apnorm = std::sqrt(dapx*dapx+dapy*dapy);
#else
           const Real dapz = apz(i  ,j  ,k+1) - apz(i,j,k);
           apnorm = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);
#endif
           apnorm_inv = 1.0/apnorm;
           const Real nx = dapx * apnorm_inv;
           const Real ny = dapy * apnorm_inv;
#if (AMREX_SPACEDIM == 3)
           const Real nz = dapz * apnorm_inv;
#endif

           IntVect off;
#if (AMREX_SPACEDIM == 2)
           if (std::abs(nx) > std::abs(ny)+small_norm) 
           {
               if (nx > 0) 
                   off = IntVect(1,0);
               else
                   off = IntVect(-1,0);

           } else {
               if (ny > 0) 
                   off = IntVect(0,1);
               else
                   off = IntVect(0,-1);
           }
#else
           if ( (std::abs(nx) > std::abs(ny)+small_norm) && (std::abs(nx)> std::abs(nz)+small_norm) )
           {
               if (nx > 0) 
                   off = IntVect(1,0,0);
               else
                   off = IntVect(-1,0,0);
           } else if ( (std::abs(ny) > std::abs(nx)+small_norm) && std::abs(ny) > std::abs(nz)+small_norm) 
           {
               if (ny > 0) 
                   off = IntVect(0,1,0);
               else
                   off = IntVect(0,-1,0);
           } else 
           {
               if (nz > 0) 
                   off = IntVect(0,0,1);
               else
                   off = IntVect(0,0,-1);
           }
#endif
    
           // Override above logic if at a domain boundary (and non-periodic)
           if ( !is_periodic_x && (i == domain.smallEnd(0) || i == domain.bigEnd(0)) )
           {
               if (ny > 0) 
                   off = IntVect(AMREX_D_DECL(0,1,0));
               else 
                   off = IntVect(AMREX_D_DECL(0,-1,0));
           }
           if ( !is_periodic_y && (j == domain.smallEnd(1) || j == domain.bigEnd(1)) )
           {
               if (nx > 0) 
                   off = IntVect(AMREX_D_DECL(1,0,0));
               else 
                   off = IntVect(AMREX_D_DECL(-1,0,0));
           }
#if (AMREX_SPACEDIM == 3)
           if ( !is_periodic_z && (k == domain.smallEnd(2) || k == domain.bigEnd(2)) )
           {
               if (nx > 0) 
                   off = IntVect(0,0,1);
               else 
                   off = IntVect(0,0,-1);
           }
#endif

           Real sum_vol = vfrac(i,j,k) + vfrac(i+off[0],j+off[1],k+off[2]); 

           if (sum_vol < 0.5)
           {
               amrex::Print() << "Cell " << IntVect(AMREX_D_DECL(i,j,k)) << " with volfrac " << vfrac(i,j,k) << 
                                 " trying to merge with " << IntVect(AMREX_D_DECL(i+off[0],j+off[1],k+off[2])) <<
                                 " with volfrac " << vfrac(i+off[0],j+off[1],k+off[2]) << std::endl;
               amrex::Abort(" total vol not big enough ");
           }

#if (AMREX_SPACEDIM == 2)
           itracker(i+off[0],j+off[1],k) += 1;

           if (vfrac(i+off[0],j+off[1],k) == 0.)
               amrex::Abort(" Trying to merge with covered cell");

           for (int n = 0; n < ncomp; n++)
           { 
               Real sum_upd = vfrac(i       ,j       ,k) * dUdt_in(i       ,j       ,k,n) 
                            + vfrac(i+off[0],j+off[1],k) * dUdt_in(i+off[0],j+off[1],k,n); 

               Real avg_update = sum_upd / sum_vol;

               dUdt_out(i       ,j       ,k,n) = avg_update;
               dUdt_out(i+off[0],j+off[1],k,n) = avg_update;
           }
#else
           itracker(i+off[0],j+off[1],k+off[2]) += 1;

           if (vfrac(i+off[0],j+off[1],k+off[2]) == 0.)
               amrex::Abort(" Trying to merge with covered cell");

           for (int n = 0; n < ncomp; n++)
           { 
               Real sum_upd = vfrac(i       ,j       ,k       ) * dUdt_in(i       ,j       ,k       ,n) 
                            + vfrac(i+off[0],j+off[1],k+off[2]) * dUdt_in(i+off[0],j+off[1],k+off[2],n); 

               Real avg_update = sum_upd / sum_vol;

               dUdt_out(i       ,j       ,k       ,n) = avg_update;
               dUdt_out(i+off[0],j+off[1],k+off[2],n) = avg_update;
           }
#endif
         }
       }
    });

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (itracker(i,j,k) > 1)
        {
           amrex::Print() << "ITracker > 1 at " << IntVect(AMREX_D_DECL(i,j,k)) << " with value " << itracker(i,j,k) << std::endl;
           amrex::Abort();
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
#if (AMREX_SPACEDIM == 3)
          for (int k = bx.smallEnd(2); k <= domain.bigEnd(2); k++)  
#else
          int k = 0; 
#endif
          for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); j++)  
          {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); i++)  
            {
              if (vfrac(i,j,k) > 0.)
              {
                  sum1 += vfrac(i,j,k)*dUdt_in(i,j,k,n);
                  sum2 += vfrac(i,j,k)*dUdt_out(i,j,k,n);
              }
            }
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
           int num_merge = 0; 
           Real sum_vol = vfrac(i,j,k); 
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
#if (AMREX_SPACEDIM == 3)
          for (int k = bx.smallEnd(2); k <= domain.bigEnd(2); k++)  
#else
          int k = 0; 
#endif
          for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); j++)  
          for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); i++)  
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
