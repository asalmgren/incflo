#include <incflo.H>

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
                                    Geometry& lev_geom)
{
    const Box dbox = lev_geom.growPeriodicDomain(2);

    amrex::Print() << " DOING BOX " << bx << " with ncomp " << ncomp << std::endl;

    Box const& bxg1 = amrex::grow(bx,1);
    Box const& bxg2 = amrex::grow(bx,2);

    IArrayBox nbor_fab(bx,9);
    FArrayBox nbor_wt_fab(bx,1);
    nbor_fab.setVal(0);
    nbor_wt_fab.setVal(0.);
    Array4<int>  nbor    = nbor_fab.array();
    Array4<Real> nbor_wt = nbor_wt_fab.array();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (flag(i,j,k).isSingleValued() and vfrac(i,j,k) < 0.5 && i > 3 && i < 5) 
        {
            amrex::Print() << "SMALL CELL " << IntVect(i,j) << " with vfrac " << vfrac(i,j,k) << std::endl;

            if (apx(i,j,k) > 0.)
            {
                if (fcx(i,j,k,0) <= 0.)
                {
                    if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                    if (vfrac(i-1,j-1,k) > 0.) nbor(i,j,k,0) = 1;
                    if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                } else {
                    if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                    if (vfrac(i-1,j+1,k) > 0.) nbor(i,j,k,6) = 1;
                    if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                }
            }

            if (apx(i+1,j,k) > 0.)
            {
                if (fcx(i+1,j,k,0) <= 0.)
                {
                    if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                    if (vfrac(i+1,j-1,k) > 0.) nbor(i,j,k,2) = 1;
                    if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                } else {
                    if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                    if (vfrac(i+1,j+1,k) > 0.) nbor(i,j,k,8) = 1;
                    if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                }
            }

            if (apy(i,j,k) > 0.)
            {
                if (fcy(i,j,k,0) <= 0.)
                {
                    if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                    if (vfrac(i-1,j-1,k) > 0.) nbor(i,j,k,0) = 1;
                    if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                } else {
                    if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                    if (vfrac(i+1,j-1,k) > 0.) nbor(i,j,k,2) = 1;
                    if (vfrac(i  ,j-1,k) > 0.) nbor(i,j,k,1) = 1;
                }
            }

            if (apy(i,j+1,k) > 0.)
            {
                if (fcy(i,j+1,k,0) <= 0.)
                {
                    if (vfrac(i-1,j  ,k) > 0.) nbor(i,j,k,3) = 1;
                    if (vfrac(i-1,j+1,k) > 0.) nbor(i,j,k,6) = 1;
                    if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                } else {
                    if (vfrac(i+1,j  ,k) > 0.) nbor(i,j,k,5) = 1;
                    if (vfrac(i+1,j+1,k) > 0.) nbor(i,j,k,8) = 1;
                    if (vfrac(i  ,j+1,k) > 0.) nbor(i,j,k,7) = 1;
                }
            }

            nbor_wt(i,j,k) = vfrac(i,j,k);

            for (int jj = -1; jj <= 1; jj++)  
            for (int ii = -1; ii <= 1; ii++)  
            {
                int index = (jj+1)*3 + (ii+1);
                if (nbor(i,j,k,index) == 1)
                {
                    amrex::Print() << IntVect(ii,jj) << " is connected with vol " << vfrac(i+ii,j+jj,k) << std::endl;
                    nbor_wt(i,j,k) += vfrac(i+ii,j+jj,k);
                }
            }
            amrex::Print() << "VOL IN NBOR OF CELL " << IntVect(i,j) << " " << nbor_wt(i,j,k) << std::endl;
        }
    });

    exit(0);

    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        dUdt(i,j,k,n) = dUdt_in(i,j,k,n) + tmp(i,j,k,n);
    });
}
#endif
