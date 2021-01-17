#ifdef AMREX_USE_EB
#include <Redistribution.H>

using namespace amrex;

void redistribution::redistribute_eb (Box const& bx, int ncomp,
                                      Array4<Real> const& dUdt,
                                      Array4<Real const> const& dUdt_in,
                                      Array4<Real> const& scratch,
                                      AMREX_D_DECL(amrex::Array4<amrex::Real const> const& umac,
                                                   amrex::Array4<amrex::Real const> const& vmac,
                                                   amrex::Array4<amrex::Real const> const& wmac),
                                      Array4<EBCellFlag const> const& flag,
                                      AMREX_D_DECL(amrex::Array4<amrex::Real const> const& apx,
                                                   amrex::Array4<amrex::Real const> const& apy,
                                                   amrex::Array4<amrex::Real const> const& apz),
                                      amrex::Array4<amrex::Real const> const& vfrac,
                                      AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fcx,
                                                   amrex::Array4<amrex::Real const> const& fcy,
                                                   amrex::Array4<amrex::Real const> const& fcz),
                                      amrex::Array4<amrex::Real const> const& ccc,
                                      Geometry& lev_geom, bool is_velocity)
{
    // int redist_type = 0;   // no redistribution
    // int redist_type = 1;   // flux_redistribute
    // int redist_type = 2;   // state_redistribute
    int redist_type = 3;   // merge_redistribute
  
    {
        if (redist_type == 1)
        {
            flux_redistribute_eb (bx, ncomp, dUdt, dUdt_in, scratch, flag, vfrac, lev_geom);
    
        } else if (redist_type == 2) {
            state_redistribute_eb(bx, ncomp, dUdt, dUdt_in, flag,
                                  AMREX_D_DECL(apx, apy, apz), vfrac,
                                  AMREX_D_DECL(fcx, fcy, fcz), ccc, lev_geom);

        } else if (redist_type == 3) {
            merge_redistribute_eb(bx, ncomp, dUdt, dUdt_in,
                                  AMREX_D_DECL(umac, vmac, wmac), flag,
                                  AMREX_D_DECL(apx, apy, apz), vfrac,
                                  AMREX_D_DECL(fcx, fcy, fcz), ccc, lev_geom);

        } else if (redist_type == 0) {
            amrex::ParallelFor(bx, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    dUdt(i,j,k,n) = dUdt_in(i,j,k,n);
                }
            );

        } else {
           amrex::Error("Not a legit redist_type");
        }
    }
}
#endif
