#ifdef AMREX_USE_EB

#include <Redistribution.H>
#include <AMReX_EB_slopes_K.H>

using namespace amrex;

#if (AMREX_SPACEDIM == 3)
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

    // We will use small_norm as an off just to break the tie when at 45 degrees ...
    const Real small_norm = 1.e-8;

    const auto& is_periodic_x = lev_geom.isPeriodic(0);
    const auto& is_periodic_y = lev_geom.isPeriodic(1);
    const auto& is_periodic_z = lev_geom.isPeriodic(2);

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
           const Real dapz = apz(i  ,j  ,k+1) - apz(i,j,k);
           apnorm = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);
           apnorm_inv = 1.0/apnorm;
           const Real nx = dapx * apnorm_inv;
           const Real ny = dapy * apnorm_inv;
           const Real nz = dapz * apnorm_inv;

           IntVect off;
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
           if ( !is_periodic_z && (k == domain.smallEnd(2) || k == domain.bigEnd(2)) )
           {
               if (nx > 0) 
                   off = IntVect(0,0,1);
               else 
                   off = IntVect(0,0,-1);
           }

           Real sum_vol = vfrac(i,j,k) + vfrac(i+off[0],j+off[1],k+off[2]); 

               amrex::Print() << "Cell " << IntVect(AMREX_D_DECL(i,j,k)) << " with volfrac " << vfrac(i,j,k) << 
                                 " trying to merge with " << IntVect(AMREX_D_DECL(i+off[0],j+off[1],k+off[2])) <<
                                 " with volfrac " << vfrac(i+off[0],j+off[1],k+off[2]) << std::endl;
           if (sum_vol < 0.5)
           {
               amrex::Abort(" total vol not big enough ");
           }

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
          for (int k = bx.smallEnd(2); k <= domain.bigEnd(2); k++)  
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
#endif
#endif
