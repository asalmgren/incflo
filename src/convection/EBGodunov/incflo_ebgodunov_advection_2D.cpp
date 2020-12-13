#include <incflo_godunov_corner_couple.H>
#include <incflo_godunov_trans_bc.H>

#include <Godunov.H>
#include <EBGodunov.H>

#include <AMReX_MultiCutFab.H>
#include <AMReX_EBMultiFabUtil_2D_C.H>

using namespace amrex;

void
ebgodunov::compute_godunov_advection (Box const& bx, int ncomp,
                                      Array4<Real> const& dqdt,
                                      Array4<Real const> const& q,
                                      Array4<Real const> const& u_mac,
                                      Array4<Real const> const& v_mac,
                                      Array4<Real const> const& fq,
                                      Array4<Real const> const& divu,
                                      Real l_dt,
                                      Vector<BCRec> const& h_bcrec,
                                             BCRec const*  pbc,
                                      int const* iconserv,
                                      Real* p, 
                                      Array4<EBCellFlag const> const& flag_arr,
                                      AMREX_D_DECL(Array4<Real const> const& apx,
                                                   Array4<Real const> const& apy,
                                                   Array4<Real const> const& apz),
                                      Array4<Real const> const& vfrac_arr,
                                      AMREX_D_DECL(Array4<Real const> const& fcx,
                                                   Array4<Real const> const& fcy,
                                                   Array4<Real const> const& fcz),
                                      Array4<Real const> const& ccent_arr,
                                      Geometry& geom,
                                      bool is_velocity )
{
    Box const& xbx = amrex::surroundingNodes(bx,0);
    Box const& ybx = amrex::surroundingNodes(bx,1);
    Box const& bxg1 = amrex::grow(bx,1);
    Box xebox = Box(xbx).grow(1,1);
    Box yebox = Box(ybx).grow(0,1);

    const Real dx = geom.CellSize(0);
    const Real dy = geom.CellSize(1);
    Real dtdx = l_dt/dx;
    Real dtdy = l_dt/dy;

    Box const& domain = geom.Domain();
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);
    const auto dxinv = geom.InvCellSizeArray();

    Array4<Real> Imx = makeArray4(p, bxg1, ncomp);
    p +=         Imx.size();
    Array4<Real> Ipx = makeArray4(p, bxg1, ncomp);
    p +=         Ipx.size();
    Array4<Real> Imy = makeArray4(p, bxg1, ncomp);
    p +=         Imy.size();
    Array4<Real> Ipy = makeArray4(p, bxg1, ncomp);
    p +=         Ipy.size();
    Array4<Real> xlo = makeArray4(p, xebox, ncomp);
    p +=         xlo.size();
    Array4<Real> xhi = makeArray4(p, xebox, ncomp);
    p +=         xhi.size();
    Array4<Real> ylo = makeArray4(p, yebox, ncomp);
    p +=         ylo.size();
    Array4<Real> yhi = makeArray4(p, yebox, ncomp);
    p +=         yhi.size();
    Array4<Real> xyzlo = makeArray4(p, bxg1, ncomp);
    p +=         xyzlo.size();
    Array4<Real> xyzhi = makeArray4(p, bxg1, ncomp);
    p +=         xyzhi.size();

    amrex::Print() << "DOING EBGODNOV ADVECTION " << xebox << std::endl; 

    for (int n = 0; n < ncomp; n++) 
       if (!iconserv[n]) amrex::Abort("Trying to update in non-conservative in ebgodunov");

    // 
    // Use PLM to generate Im and Ip 
    // 
    // amrex::ParallelFor(xebox, ncomp,
    // [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    // {
    //     EBGodunov_plm_fpu_x(i, j, k, n, l_dt, dx, Imx(i,j,k,n), Ipx(i-1,j,k,n),
    //                          q, u_mac(i,j,k), pbc[n], dlo.x, dhi.x, is_velocity);
    // });

    // amrex::ParallelFor(yebox, ncomp,
    // [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    // {
    //     EBGodunov_plm_fpu_y(i, j, k, n, l_dt, dy, Imy(i,j,k,n), Ipy(i,j-1,k,n),
    //                           q, v_mac(i,j,k), pbc[n], dlo.y, dhi.y, is_velocity);
    // });

    ebgodunov::plm_fpu_x (bx, ncomp, Imx, Ipx, q, u_mac,
                          flag_arr,AMREX_D_DECL(apx,apy,apz),vfrac_arr,
                          AMREX_D_DECL(fcx,fcy,fcz),ccent_arr,
                          geom, l_dt, h_bcrec, pbc, is_velocity);
    ebgodunov::plm_fpu_y (bx, ncomp, Imy, Ipy, q, v_mac, 
                          flag_arr,AMREX_D_DECL(apx,apy,apz),vfrac_arr,
                          AMREX_D_DECL(fcx,fcy,fcz),ccent_arr,
                          geom, l_dt, h_bcrec, pbc, is_velocity);

    amrex::ParallelFor(
        xebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (apx(i,j,k) > 0.)
            {
                Real lo = Ipx(i-1,j,k,n);
                Real hi = Imx(i  ,j,k,n);

                Real uad = u_mac(i,j,k);

                auto bc = pbc[n];  

                Godunov_trans_xbc(i, j, k, n, q, lo, hi, uad, bc.lo(0), bc.hi(0), dlo.x, dhi.x, is_velocity);
    
                xlo(i,j,k,n) = lo; 
                xhi(i,j,k,n) = hi;

                Real st = (uad >= 0.) ? lo : hi;
                Real fux = (amrex::Math::abs(uad) < small_vel)? 0. : 1.;
                Imx(i,j,k,n) = fux*st + (1. - fux)*0.5*(hi + lo);
            } else {
                Imx(i,j,k,n) = 0.;
            }

        },
        yebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (apy(i,j,k) > 0.)
            {
                Real lo = Ipy(i,j-1,k,n);
                Real hi = Imy(i,j  ,k,n);

                Real vad = v_mac(i,j,k);
    
                auto bc = pbc[n];

                Godunov_trans_ybc(i, j, k, n, q, lo, hi, vad, bc.lo(1), bc.hi(1), dlo.y, dhi.y, is_velocity);

                ylo(i,j,k,n) = lo;
                yhi(i,j,k,n) = hi;

                Real st = (vad >= 0.) ? lo : hi;
                Real fuy = (amrex::Math::abs(vad) < small_vel)? 0. : 1.;
                Imy(i,j,k,n) = fuy*st + (1. - fuy)*0.5*(hi + lo);
            } else {
                Imy(i,j,k,n) = 0.;
            }
        });

    Array4<Real> xedge = Imx;
    Array4<Real> yedge = Imy;

    // We can reuse the space in Ipx, Ipy and Ipz.

    //
    // x-direction
    //
    Box const& xbxtmp = amrex::grow(bx,0,1);
    Array4<Real> yzlo = makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(xbxtmp,1), ncomp);
    amrex::ParallelFor(
    Box(yzlo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (apx(i,j,k) > 0.)
        {
            const auto bc = pbc[n];
            Real l_yzlo, l_yzhi;

            l_yzlo = ylo(i,j,k,n);
            l_yzhi = yhi(i,j,k,n);
            Real vad = v_mac(i,j,k);
            Godunov_trans_ybc(i, j, k, n, q, l_yzlo, l_yzhi, vad, bc.lo(1), bc.hi(1), dlo.y, dhi.y, is_velocity);

            Real st = (vad >= 0.) ? l_yzlo : l_yzhi;
            Real fu = (amrex::Math::abs(vad) < small_vel) ? 0.0 : 1.0;
            yzlo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_yzhi + l_yzlo);
        } else {
            yzlo(i,j,k,n) = 0.;
        }
    });
    //
    Array4<Real> qx = makeArray4(Ipx.dataPtr(), xbx, ncomp);
    amrex::ParallelFor(xbx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Real stl, sth;

        if (apx(i,j,k) > 0.)
        {
            Real uxl = (apx(i,j,k)*u_mac(i,j,k) - apx(i-1,j,k)*u_mac(i-1,j,k)) / vfrac_arr(i-1,j,k);
            stl = xlo(i,j,k,n) - (0.5*dtdx) * q(i-1,j,k,n) * uxl
                               - (0.5*dtdy)*(yzlo(i-1,j+1,k  ,n)*v_mac(i-1,j+1,k  )
                                           - yzlo(i-1,j  ,k  ,n)*v_mac(i-1,j  ,k  ));

            Real uxh = (apx(i+1,j,k)*u_mac(i+1,j,k) - apx(i,j,k)*u_mac(i,j,k)) / vfrac_arr(i,j,k);
            sth = xhi(i,j,k,n) - (0.5*dtdx) * q(i  ,j,k,n) * uxh
                               - (0.5*dtdy)*(yzlo(i,j+1,k  ,n)*v_mac(i,j+1,k  )
                                           - yzlo(i,j  ,k  ,n)*v_mac(i,j  ,k  ));

            if (fq) {
                stl += 0.5*l_dt*fq(i-1,j,k,n);
                sth += 0.5*l_dt*fq(i  ,j,k,n);
            }

            auto bc = pbc[n]; 
            Godunov_cc_xbc_lo(i, j, k, n, q, stl, sth, u_mac, bc.lo(0), dlo.x, is_velocity);
            Godunov_cc_xbc_hi(i, j, k, n, q, stl, sth, u_mac, bc.hi(0), dhi.x, is_velocity);

            Real temp = (u_mac(i,j,k) >= 0.) ? stl : sth; 
            temp = (amrex::Math::abs(u_mac(i,j,k)) < small_vel) ? 0.5*(stl + sth) : temp;
            qx(i,j,k,n) = temp;
        } else {
            qx(i,j,k,n) = 0.;
        }
    }); 

    //
    // y-direction
    //
    Box const& ybxtmp = amrex::grow(bx,1,1);
    Array4<Real> xzlo = makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(ybxtmp,0), ncomp);
    amrex::ParallelFor(
    Box(xzlo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (apy(i,j,k) > 0.)
        {
            const auto bc = pbc[n];
            Real l_xzlo, l_xzhi;

            l_xzlo = xlo(i,j,k,n);
            l_xzhi = xhi(i,j,k,n);

            Real uad = u_mac(i,j,k);
            Godunov_trans_xbc(i, j, k, n, q, l_xzlo, l_xzhi, uad, bc.lo(0), bc.hi(0), dlo.x, dhi.x, is_velocity);

            Real st = (uad >= 0.) ? l_xzlo : l_xzhi;
            Real fu = (amrex::Math::abs(uad) < small_vel) ? 0.0 : 1.0;
            xzlo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_xzhi + l_xzlo);
        } else {
            xzlo(i,j,k,n) = 0.;
        }
    });
    //

    Array4<Real> qy = makeArray4(Ipy.dataPtr(), ybx, ncomp);
    amrex::ParallelFor(ybx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Real stl, sth;

        if (apy(i,j,k) > 0.)
        {
            Real vyl = (apy(i,j,k)*v_mac(i,j,k) - apy(i,j-1,k)*v_mac(i,j-1,k)) / vfrac_arr(i,j-1,k);
            stl = ylo(i,j,k,n) - (0.5*dtdy)*q(i,j-1,k,n)*vyl
                               - (0.5*dtdx)*(xzlo(i+1,j-1,k  ,n)*u_mac(i+1,j-1,k  )
                                           - xzlo(i  ,j-1,k  ,n)*u_mac(i  ,j-1,k  ));

            Real vyh = (apy(i,j+1,k)*v_mac(i,j+1,k) - apy(i,j,k)*v_mac(i,j,k)) / vfrac_arr(i,j-1,k);
            sth = yhi(i,j,k,n) - (0.5*dtdy)*q(i,j  ,k,n)* vyh
                               - (0.5*dtdx)*(xzlo(i+1,j,k  ,n)*u_mac(i+1,j,k  )
                                           - xzlo(i  ,j,k  ,n)*u_mac(i  ,j,k  ));

            if (fq) {
                stl += 0.5*l_dt*fq(i,j-1,k,n);
                sth += 0.5*l_dt*fq(i,j  ,k,n);
            }

            auto bc = pbc[n];
            Godunov_cc_ybc_lo(i, j, k, n, q, stl, sth, v_mac, bc.lo(1), dlo.y, is_velocity);
            Godunov_cc_ybc_hi(i, j, k, n, q, stl, sth, v_mac, bc.hi(1), dhi.y, is_velocity);

            Real temp = (v_mac(i,j,k) >= 0.) ? stl : sth; 
            temp = (amrex::Math::abs(v_mac(i,j,k)) < small_vel) ? 0.5*(stl + sth) : temp; 
            qy(i,j,k,n) = temp;
        } else {
            qy(i,j,k,n) = 0.;
        }
    });

    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (vfrac_arr(i,j,k) > 0.)
        {
            dqdt(i,j,k,n) = (dxinv[0]*( apx(i  ,j,k)*u_mac(i  ,j,k)*qx(i  ,j,k,n) -
                                        apx(i+1,j,k)*u_mac(i+1,j,k)*qx(i+1,j,k,n) )
                +            dxinv[1]*( apy(i  ,j,k)*v_mac(i,j  ,k)*qy(i,j  ,k,n) -
                                        apy(i,j+1,k)*v_mac(i,j+1,k)*qy(i,j+1,k,n))) / vfrac_arr(i,j,k);
            if (i == 12 and (j == 19 or j == 12))
            {
               amrex::Print() << "DVDT " << IntVect(i,j) << " " << n << " " << dqdt(i,j,k,n) << std::endl;
               amrex::Print() << "APX  " << apx(i,j,k)  << " " << apx(i+1,j,k)  << std::endl;
               amrex::Print() << " QX  " <<   qx(i,j,k,n)  << " " << qx(i+1,j,k,n)  << std::endl;
               amrex::Print() << "UMAC " << u_mac(i,j,k) << " " << u_mac(i+1,j,k)  << std::endl;
               amrex::Print() << "APY  " << apy(i,j,k)  << " " << apy(i,j+1,k)  << std::endl;
               amrex::Print() << " QY  " <<   qy(i,j,k,n)  << " " << qy(i,j+1,k,n)  << std::endl;
               amrex::Print() << "VMAC " << v_mac(i,j,k) << " " << v_mac(i,j+1,k)  << std::endl;
               amrex::Print() << "VFRAC " << vfrac_arr(i,j,k) << std::endl;
            }
        } else {
            dqdt(i,j,k,n) = 0.0;
        }
    });
}
