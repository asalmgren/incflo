#include <incflo_godunov_trans_bc.H>

#include <Godunov.H>
#include <EBGodunov.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real
linear_interp (Real xout,
               Real x1in, Real val1,
               Real x2in, Real val2)
{
   Real val_out =  val1 + (xout-x1in)/(x2in-x1in) * (val2-val1);
   return val_out;
}

void ebgodunov::predict_godunov (Real /*time*/, 
                                 MultiFab& u_mac, MultiFab& v_mac,
                                 MultiFab const& vel, 
                                 MultiFab const& vel_forces,
                                 Vector<BCRec> const& h_bcrec,
                                        BCRec  const* d_bcrec,
                                 EBFArrayBoxFactory const* ebfact,
                                 Geometry& geom, 
                                 Real l_dt, 
                                 MultiFab const& gmacphi_x, MultiFab const& gmacphi_y,
                                 bool use_mac_phi_in_godunov)
{
    Box const& domain = geom.Domain();
    const Real* dx    = geom.CellSize();

    auto const& flags = ebfact->getMultiEBCellFlagFab();
    auto const& fcent = ebfact->getFaceCent();
    auto const& ccent = ebfact->getCentroid();
    auto const& vfrac = ebfact->getVolFrac();

    auto const& areafrac = ebfact->getAreaFrac();

    const int ncomp = AMREX_SPACEDIM;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox scratch;
        for (MFIter mfi(vel,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            Box const& bxg1 = amrex::grow(bx,1);
            Box const& xbx = mfi.nodaltilebox(0);
            Box const& ybx = mfi.nodaltilebox(1);

            EBCellFlagFab const& flagfab = flags[mfi];
            Array4<EBCellFlag const> const& flagarr = flagfab.const_array();

            Array4<Real> const& a_umac = u_mac.array(mfi);
            Array4<Real> const& a_vmac = v_mac.array(mfi);

            Array4<Real const> const& gmacphi_x_arr = gmacphi_x.const_array(mfi);
            Array4<Real const> const& gmacphi_y_arr = gmacphi_y.const_array(mfi);

            Array4<Real const> const& a_vel = vel.const_array(mfi);
            Array4<Real const> const& a_f = vel_forces.const_array(mfi);

            scratch.resize(bxg1, ncomp*(4*AMREX_SPACEDIM)+AMREX_SPACEDIM);
            Real* p = scratch.dataPtr();

            Array4<Real> Imx = makeArray4(p,bxg1,ncomp);
            p +=         Imx.size();
            Array4<Real> Ipx = makeArray4(p,bxg1,ncomp);
            p +=         Ipx.size();
            Array4<Real> Imy = makeArray4(p,bxg1,ncomp);
            p +=         Imy.size();
            Array4<Real> Ipy = makeArray4(p,bxg1,ncomp);
            p +=         Ipy.size();
            Array4<Real> u_ad = makeArray4(p,Box(bx).grow(1,1).surroundingNodes(0),1);
            p +=         u_ad.size();
            Array4<Real> v_ad = makeArray4(p,Box(bx).grow(0,1).surroundingNodes(1),1);
            p +=         v_ad.size();

            // This tests on covered cells just in the box itself
            if (flagfab.getType(bx) == FabType::covered)
            {
                // We shouldn't need to zero these ... I think?

            // This tests on only regular cells including two rows of ghost cells
            } else if (flagfab.getType(amrex::grow(bx,2)) == FabType::regular) {

                godunov::predict_plm_x (bx, Imx, Ipx, a_vel, a_vel,
                                        geom, l_dt, h_bcrec, d_bcrec);
                godunov::predict_plm_y (bx, Imy, Ipy, a_vel, a_vel,
                                        geom, l_dt, h_bcrec, d_bcrec);

                bool local_use_forces_in_trans = false;
                godunov::make_trans_velocities(Box(u_ad), Box(v_ad),
                                               u_ad, v_ad,
                                               Imx, Imy, Ipx, Ipy, a_vel, a_f,
                                               domain, l_dt, d_bcrec, local_use_forces_in_trans);

                godunov::predict_godunov_on_box(bx, ncomp, xbx, ybx, 
                                                a_umac, a_vmac,
                                                a_vel, u_ad, v_ad, 
                                                Imx, Imy, Ipx, Ipy, a_f, 
                                                domain, dx, l_dt, d_bcrec, 
                                                local_use_forces_in_trans,
                                                gmacphi_x_arr, gmacphi_y_arr, 
                                                use_mac_phi_in_godunov, p);

            } else {
 
                AMREX_D_TERM(Array4<Real const> const& fcx = fcent[0]->const_array(mfi);,
                             Array4<Real const> const& fcy = fcent[1]->const_array(mfi);,
                             Array4<Real const> const& fcz = fcent[2]->const_array(mfi););

                Array4<Real const> const& ccent_arr = ccent.const_array(mfi);
                Array4<Real const> const& vfrac_arr = vfrac.const_array(mfi);

                ebgodunov::predict_plm_x (bx, Imx, Ipx, a_vel, a_vel,
                                         flagarr, vfrac_arr,
                                         AMREX_D_DECL(fcx,fcy,fcz),ccent_arr,
                                         geom, l_dt, h_bcrec, d_bcrec);
                ebgodunov::predict_plm_y (bx, Imy, Ipy, a_vel, a_vel,
                                         flagarr, vfrac_arr,
                                         AMREX_D_DECL(fcx,fcy,fcz),ccent_arr,
                                         geom, l_dt, h_bcrec, d_bcrec);

                ebgodunov::make_trans_velocities(Box(u_ad), Box(v_ad),
                                                 u_ad, v_ad,
                                                 Imx, Imy, Ipx, Ipy, a_vel,
                                                 flagarr, domain, d_bcrec);
   
                AMREX_D_TERM(Array4<Real const> const& apx = areafrac[0]->const_array(mfi);,
                             Array4<Real const> const& apy = areafrac[1]->const_array(mfi);,
                             Array4<Real const> const& apz = areafrac[2]->const_array(mfi););

                ebgodunov::predict_godunov_on_box(bx, ncomp, xbx, ybx, 
                                                  a_umac, a_vmac,
                                                  a_vel, u_ad, v_ad, 
                                                  Imx, Imy, Ipx, Ipy, a_f, 
                                                  domain, dx, l_dt, d_bcrec, 
                                                  flagarr, apx,apy,vfrac_arr,
                                                  fcx,fcy,
                                                  gmacphi_x_arr, gmacphi_y_arr, 
                                                  use_mac_phi_in_godunov, p);
            }

            Gpu::streamSynchronize();  // otherwise we might be using too much memory
        }
    }
}

void ebgodunov::make_trans_velocities (Box const& xbx, Box const& ybx, 
                                       Array4<Real> const& u_ad,
                                       Array4<Real> const& v_ad,
                                       Array4<Real const> const& Imx,
                                       Array4<Real const> const& Imy,
                                       Array4<Real const> const& Ipx,
                                       Array4<Real const> const& Ipy,
                                       Array4<Real const> const& vel,
                                       Array4<EBCellFlag const> const& flag,
                                       const Box& domain,
                                       BCRec  const* pbc)
  {
    const Dim3 dlo = amrex::lbound(domain);
    const Dim3 dhi = amrex::ubound(domain);

    amrex::ParallelFor(xbx, ybx, //zbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // We only care about x-velocity on x-faces here
        if (flag(i,j,k).isConnected(-1,0,0))  
        {
            constexpr int n = 0;

            Real lo = Ipx(i-1,j,k,n);
            Real hi = Imx(i  ,j,k,n);

            auto bc = pbc[n];
            Godunov_trans_xbc(i, j, k, n, vel, lo, hi, bc.lo(0), bc.hi(0), dlo.x, dhi.x, true);

            Real st = ( (lo+hi) >= 0.) ? lo : hi;
            bool ltm = ( (lo <= 0. && hi >= 0.) || (amrex::Math::abs(lo+hi) < small_vel) );
            u_ad(i,j,k) = ltm ? 0. : st;
        } else {
            u_ad(i,j,k) = 0.;
        } 
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // We only care about y-velocity on y-faces here
        if (flag(i,j,k).isConnected(0,-1,0))  
        {
            constexpr int n = 1;

            Real lo = Ipy(i,j-1,k,n);
            Real hi = Imy(i,j  ,k,n);

            auto bc = pbc[n];
            Godunov_trans_ybc(i, j, k, n, vel, lo, hi, bc.lo(1), bc.hi(1), dlo.y, dhi.y, true);

            Real st = ( (lo+hi) >= 0.) ? lo : hi;
            bool ltm = ( (lo <= 0. && hi >= 0.) || (amrex::Math::abs(lo+hi) < small_vel) );
            v_ad(i,j,k) = ltm ? 0. : st;
        } else {
            v_ad(i,j,k) = 0.;
        }
    });
}

void ebgodunov::predict_godunov_on_box (Box const& bx, int ncomp,
                                        Box const& xbx, Box const& ybx, 
                                        Array4<Real> const& qx,
                                        Array4<Real> const& qy,
                                        Array4<Real const> const& q,
                                        Array4<Real const> const& u_ad,
                                        Array4<Real const> const& v_ad,
                                        Array4<Real> const& Imx,
                                        Array4<Real> const& Imy,
                                        Array4<Real> const& Ipx,
                                        Array4<Real> const& Ipy,
                                        Array4<Real const> const& f,
                                        const Box& domain, const Real* dx_arr,
                                        Real l_dt, BCRec  const* pbc,
                                        Array4<EBCellFlag const> const& flag,
                                        Array4<Real const> const& apx,
                                        Array4<Real const> const& apy,
                                        Array4<Real const> const& vfrac_arr,
                                        Array4<Real const> const& fcx,
                                        Array4<Real const> const& fcy,
                                        Array4<Real const> const& gmacphi_x,
                                        Array4<Real const> const& gmacphi_y,
                                        bool l_use_mac_phi_in_godunov,
                                        Real* p)
{
    const Dim3 dlo = amrex::lbound(domain);
    const Dim3 dhi = amrex::ubound(domain);
    Real dx = dx_arr[0];
    Real dy = dx_arr[1];

    Box xebox = Box(bx).grow(1,1).surroundingNodes(0);
    Box yebox = Box(bx).grow(0,1).surroundingNodes(1);
    Array4<Real> xlo = makeArray4(p, xebox, ncomp);
    p += xlo.size();
    Array4<Real> xhi = makeArray4(p, xebox, ncomp);
    p += xhi.size();
    Array4<Real> ylo = makeArray4(p, yebox, ncomp);
    p += ylo.size();
    Array4<Real> yhi = makeArray4(p, yebox, ncomp);
    p += yhi.size();

    //
    // Upwind the states that result from plm_x and plm_y
    //
    amrex::ParallelFor(
        xebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real lo = Ipx(i-1,j,k,n);
            Real hi = Imx(i  ,j,k,n);

            Real uad = u_ad(i,j,k);
            auto bc = pbc[n];

            Godunov_trans_xbc(i, j, k, n, q, lo, hi, bc.lo(0), bc.hi(0), dlo.x, dhi.x, true);

            xlo(i,j,k,n) = lo;
            xhi(i,j,k,n) = hi;

            Real st = (uad >= 0.) ? lo : hi;
            Real fu = (amrex::Math::abs(uad) < small_vel) ? 0.0 : 1.0;
            Imx(i, j, k, n) = fu*st + (1.0 - fu) *0.5 * (hi + lo); // store xedge
        },
        yebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real lo = Ipy(i,j-1,k,n);
            Real hi = Imy(i,j  ,k,n);

            Real vad = v_ad(i,j,k);
            auto bc = pbc[n];

            Godunov_trans_ybc(i, j, k, n, q, lo, hi, bc.lo(1), bc.hi(1), dlo.y, dhi.y, true);

            ylo(i,j,k,n) = lo;
            yhi(i,j,k,n) = hi;

            Real st = (vad >= 0.) ? lo : hi;
            Real fu = (amrex::Math::abs(vad) < small_vel) ? 0.0 : 1.0;
            Imy(i, j, k, n) = fu*st + (1.0 - fu)*0.5*(hi + lo); // store yedge
        });

    // We can reuse the space in Ipy

    //
    // X-Flux
    //
    Box const xbxtmp = Box(xbx).enclosedCells().grow(0,1);
    Array4<Real> yhat = makeArray4(Ipy.dataPtr(), amrex::surroundingNodes(xbxtmp,1), 1);
    // Add du/dy term to u on x-faces
    // Start with {ylo,yhi} --> upwind using vad to {yhat}

    amrex::ParallelFor(Box(yhat),
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (flag(i,j,k).isConnected(0,-1,0))  
        {
            constexpr int n = 0;
            const auto bc = pbc[n];
            Real l_yzlo, l_yzhi;

            l_yzlo = ylo(i,j,k,n);
            l_yzhi = yhi(i,j,k,n);
            Real vad = v_ad(i,j,k);
            Godunov_trans_ybc(i, j, k, n, q, l_yzlo, l_yzhi, bc.lo(1), bc.hi(1), dlo.y, dhi.y, true);

            Real st = (vad >= 0.) ? l_yzlo : l_yzhi;
            Real fu = (amrex::Math::abs(vad) < small_vel) ? 0.0 : 1.0;
            yhat(i,j,k) = fu*st + (1.0 - fu) * 0.5 * (l_yzhi + l_yzlo);
        } else {
            yhat(i,j,k) = 0.0;
        } 
    });

    //
    // Define du/dy and add (v du/dy) to u on x-faces
    //
    amrex::ParallelFor(xbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (flag(i,j,k).isConnected(-1,0,0))  
        {
        constexpr int n = 0;
        auto bc = pbc[n];

        // stl is on the left  side of the lo-x side of cell (i,j)
        // sth is on the right side of the lo-x side of cell (i,j)
        Real stl, sth;
        Real v_tmp_j, v_tmp_jp1;
        Real y_hat_j, y_hat_jp1;

        // Left side of interface
        if (flag(i-1,j,k).isRegular())
        {
            // For full cells this is the transverse term
            stl = xlo(i,j,k,n) - (0.25*l_dt/dy)*(v_ad(i-1,j+1,k  )+v_ad(i-1,j,k))*
                                                (yhat(i-1,j+1,k  )-yhat(i-1,j,k));

        } else { // should be single-valued since it is connected across this face

            Real dudy_l;  //  (yhat(i-1,j+1,k)-yhat(i-1,j,k)) if regular
            Real v_tmp_l; //  0.5 * (v_ad(i-1,j+1,k)+v_ad(i-1,j,k)) if regular

            stl = xlo(i,j,k,n);

            // If either y-face is covered, set the tranxverse term to zero
            if (apy(i-1,j,k) > 0.0 && apy(i-1,j+1,k) > 0.0) 
            {
                // Tangential extrapolation in the x-direction at j-1/2
                if (apy(i-1,j,k) == 1.0)
                {
                    v_tmp_j = v_ad(i-1,j,k);
                    y_hat_j = yhat(i-1,j,k);
                }
                else if (fcy(i-1,j,k) <= 0.0) 
                {
                    // Interp to face center from face centroids
                    v_tmp_j = linear_interp(0.0, fcy(i-1,j,k), v_ad(i-1,j,k), (-1.+fcy(i-2,j,k)), v_ad(i-2,j,k));
                    y_hat_j = linear_interp(0.0, fcy(i-1,j,k), yhat(i-1,j,k), (-1.+fcy(i-2,j,k)), yhat(i-2,j,k));
                }
                else if (fcy(i-1,j,k) > 0.0) 
                {
                    // Interp to face center from face centroids
                    v_tmp_j = linear_interp(0.0, fcy(i-1,j,k), v_ad(i-1,j,k), ( 1.+fcy(i  ,j,k)), v_ad(i  ,j,k));
                    y_hat_j = linear_interp(0.0, fcy(i-1,j,k), yhat(i-1,j,k), ( 1.+fcy(i  ,j,k)), yhat(i  ,j,k));
                }

                // Tangential extrapolation in the x-direction at j+1/2
                if (apy(i-1,j+1,k) == 1.0)
                {
                    v_tmp_jp1 = v_ad(i-1,j+1,k);
                    y_hat_jp1 = yhat(i-1,j+1,k);
                }
                else if (fcy(i-1,j+1,k) <= 0.0) 
                {
                    // Interp to face center from face centroids
                    v_tmp_jp1 = linear_interp(0.0, fcy(i-1,j+1,k), v_ad(i-1,j+1,k), (-1.+fcy(i-2,j+1,k)), v_ad(i-2,j+1,k));
                    y_hat_jp1 = linear_interp(0.0, fcy(i-1,j+1,k), yhat(i-1,j+1,k), (-1.+fcy(i-2,j+1,k)), yhat(i-2,j+1,k));
                }
                else if (fcy(i-1,j+1,k) > 0.0) 
                {
                    // Interp to face center from face centroids
                    v_tmp_jp1 = linear_interp(0.0, fcy(i-1,j+1,k), v_ad(i-1,j+1,k), ( 1.+fcy(i  ,j+1,k)), v_ad(i  ,j+1,k));
                    y_hat_jp1 = linear_interp(0.0, fcy(i-1,j+1,k), yhat(i-1,j+1,k), ( 1.+fcy(i  ,j+1,k)), yhat(i  ,j+1,k));
                }

                v_tmp_l = 0.5 * (v_tmp_jp1 + v_tmp_j);
                 dudy_l =       (y_hat_jp1 - y_hat_j);

                Real trans_stl = v_tmp_l * dudy_l / dy;

                stl += -0.5 * l_dt * trans_stl;

                if (vfrac_arr(i-1,j,k) > 0.)
                    stl += 0.5 * l_dt * f(i-1,j,k,n);
            }
        }

        // Right side of interface
        if (flag(i,j,k).isRegular())
        {
            // For full cells this is the transverse term
            sth = xhi(i,j,k,n) - (0.25*l_dt/dy)*(v_ad(i  ,j+1,k  )+v_ad(i  ,j,k))*
                                                (yhat(i  ,j+1,k  )-yhat(i  ,j,k));

        } else { // should be single-valued since it is connected across this face

            Real dudy_h;  //       (yhat(i,j+1,k)-yhat(i,j,k)) if regular
            Real v_tmp_h; // 0.5 * (v_ad(i,j+1,k)+v_ad(i,j,k)) if regular

            sth = xhi(i,j,k,n);

            // If either face is covered, use the cell-centered y-velocity
            if (apy(i,j,k) > 0.0 && apy(i,j+1,k) > 0.0) 
            {
                // Tangential extrapolation in the x-direction at j-1/2
                if (apy(i,j,k) == 1.0)
                {
                    v_tmp_j = v_ad(i,j,k);
                    y_hat_j = yhat(i,j,k);
                }
                else if (fcy(i,j,k) <= 0.0) 
                {
                    // Interp to face center from face centroids
                    v_tmp_j = linear_interp(0.0, fcy(i,j,k), v_ad(i,j,k), (-1.+fcy(i-1,j,k)), v_ad(i-1,j,k));
                    y_hat_j = linear_interp(0.0, fcy(i,j,k), yhat(i,j,k), (-1.+fcy(i-1,j,k)), yhat(i-1,j,k));
                }
                else if (fcy(i,j,k) > 0.0) 
                {
                    // Interp to face center from face centroids
                    v_tmp_j = linear_interp(0.0, fcy(i,j,k), v_ad(i,j,k), ( 1.+fcy(i+1,j,k)), v_ad(i+1,j,k));
                    y_hat_j = linear_interp(0.0, fcy(i,j,k), yhat(i,j,k), ( 1.+fcy(i+1,j,k)), yhat(i+1,j,k));
                }

                // Tangential extrapolation in the x-direction at j+1/2
                if (apy(i,j+1,k) == 1.0)
                {
                    v_tmp_jp1 = v_ad(i,j+1,k);
                    y_hat_jp1 = yhat(i,j+1,k);
                }
                else if (fcy(i,j+1,k) <= 0.0) 
                {
                    v_tmp_jp1 = linear_interp(0.0, fcy(i,j+1,k), v_ad(i,j+1,k), (-1.+fcy(i-1,j+1,k)), v_ad(i-1,j+1,k));
                    y_hat_jp1 = linear_interp(0.0, fcy(i,j+1,k), yhat(i,j+1,k), (-1.+fcy(i-1,j+1,k)), yhat(i-1,j+1,k));
                }
                else if (fcy(i,j+1,k) > 0.0) 
                {
                    // Interp to face center from face centroids
                    v_tmp_jp1 = linear_interp(0.0, fcy(i,j+1,k), v_ad(i,j+1,k), ( 1.+fcy(i+1,j+1,k)), v_ad(i+1,j+1,k));
                    y_hat_jp1 = linear_interp(0.0, fcy(i,j+1,k), yhat(i,j+1,k), ( 1.+fcy(i+1,j+1,k)), yhat(i+1,j+1,k));
                }

                v_tmp_h = 0.5 * (v_tmp_jp1 + v_tmp_j);
                 dudy_h =       (y_hat_jp1 - y_hat_j);

                Real trans_sth = v_tmp_h * dudy_h / dy;

                sth += -0.5 * l_dt * trans_sth;

                if (vfrac_arr(i,j,k) > 0.)
                    sth += 0.5 * l_dt * f(i  ,j,k,n);
            }
        }

        if (l_use_mac_phi_in_godunov)
        {
            // Note that the getFluxes call returns (-1/rho G^Mac phi) so we use the negative here
            Real gphi_x = -gmacphi_x(i,j,k);
            stl -= 0.5 * l_dt * gphi_x;
            sth -= 0.5 * l_dt * gphi_x;
        }

        Godunov_cc_xbc_lo(i, j, k, n, q, stl, sth, bc.lo(0), dlo.x, true);
        Godunov_cc_xbc_hi(i, j, k, n, q, stl, sth, bc.hi(0), dhi.x, true);

        // Prevent backflow
        if ( (i==dlo.x) and (bc.lo(0) == BCType::foextrap || bc.lo(0) == BCType::hoextrap) )
        {
            sth = amrex::min(sth,0.);
            stl = sth;
        }
        if ( (i==dhi.x+1) and (bc.hi(0) == BCType::foextrap || bc.hi(0) == BCType::hoextrap) )
        {
             stl = amrex::max(stl,0.);
             sth = stl;
        }
        Real st = ( (stl+sth) >= 0.) ? stl : sth;
        bool ltm = ( (stl <= 0. && sth >= 0.) || (amrex::Math::abs(stl+sth) < small_vel) );
        qx(i,j,k) = ltm ? 0. : st;

        if (l_use_mac_phi_in_godunov)
        {
            // Note that the getFluxes call returns (-1/rho G^Mac phi) so we use the negative here
            Real gphi_x = -gmacphi_x(i,j,k);
            qx(i,j,k) += 0.5 * l_dt * gphi_x;
        }

        } else {
            qx(i,j,k) = 0.;
        } 
    });
    //
    // Y-Flux
    //
    Box const ybxtmp = Box(ybx).enclosedCells().grow(1,1);
    Array4<Real> xhat = makeArray4(Ipy.dataPtr(), amrex::surroundingNodes(ybxtmp,0), 1);

    // Start with {xlo,xhi} --> upwind using uad to {xhat}
    amrex::ParallelFor(Box(xhat),
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (flag(i,j,k).isConnected(-1,0,0))  
        {
            constexpr int n = 1;
            const auto bc = pbc[n];
            Real l_xzlo, l_xzhi;

            l_xzlo = xlo(i,j,k,n);
            l_xzhi = xhi(i,j,k,n);

            Real uad = u_ad(i,j,k);
            Godunov_trans_xbc(i, j, k, n, q, l_xzlo, l_xzhi, bc.lo(0), bc.hi(0), dlo.x, dhi.x, true);

            Real st = (uad >= 0.) ? l_xzlo : l_xzhi;
            Real fu = (amrex::Math::abs(uad) < small_vel) ? 0.0 : 1.0;
            xhat(i,j,k) = fu*st + (1.0 - fu) * 0.5 * (l_xzhi + l_xzlo);
        } else {
            xhat(i,j,k) = 0.0;
        } 
    });

    //
    // Define dv/dx and add (u dv/dx) to v on y-faces
    //
    amrex::ParallelFor(ybx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (flag(i,j,k).isConnected(0,-1,0))  
        {
        constexpr int n = 1;
        auto bc = pbc[n];

        // stl is on the low  side of the lo-y side of cell (i,j)
        // sth is on the high side of the lo-y side of cell (i,j)
        Real stl, sth;
        Real u_tmp_i, u_tmp_ip1;
        Real x_hat_i, x_hat_ip1;

        // Bottom side of interface
        if (flag(i,j-1,k).isRegular())
        {
            stl = ylo(i,j,k,n) - (0.25*l_dt/dx)*(u_ad(i+1,j-1,k  )+u_ad(i,j-1,k))*
                                                (xhat(i+1,j-1,k  )-xhat(i,j-1,k));

        } else { // should be single-valued since it is connected across this face

            Real dvdx_l;  //  (xhat(i+1,j-1,k  )-xhat(i,j-1,k)) if regular
            Real u_tmp_l; //  0.5 * (u_ad(i+1,j-1,k  )+u_ad(i,j-1,k)) if regular

            stl = ylo(i,j,k,n);

            // If either x-face is covered, set the tranxverse term to zero
            if (apx(i,j-1,k) > 0.0 && apx(i+1,j-1,k) > 0.0) 
            {
                // Tangential extrapolation in the y-direction at i-1/2
                if (apx(i,j-1,k) == 1.0)
                {
                    u_tmp_i = u_ad(i,j-1,k);
                    x_hat_i = xhat(i,j-1,k);
                }
                else if (fcx(i,j-1,k) <= 0.0) 
                {
                    // Interp to face center from face centroids
                    u_tmp_i = linear_interp(0.0, fcx(i,j-1,k), u_ad(i,j-1,k), (-1.+fcx(i,j-2,k)), u_ad(i,j-2,k));
                    x_hat_i = linear_interp(0.0, fcx(i,j-1,k), xhat(i,j-1,k), (-1.+fcx(i,j-2,k)), xhat(i,j-2,k));
                }
                else if (fcx(i,j-1,k) > 0.0) 
                {
                    // Interp to face center from face centroids
                    u_tmp_i = linear_interp(0.0, fcx(i,j-1,k), u_ad(i,j-1,k), ( 1.+fcx(i,j,k)), u_ad(i,j,k));
                    x_hat_i = linear_interp(0.0, fcx(i,j-1,k), xhat(i,j-1,k), ( 1.+fcx(i,j,k)), xhat(i,j,k));
                }

                // Tangential extrapolation in the y-direction at i+1/2
                if (apx(i+1,j-1,k) == 1.0)
                {
                    u_tmp_ip1 = u_ad(i+1,j-1,k);
                    x_hat_ip1 = xhat(i+1,j-1,k);
                }
                else if (fcx(i+1,j-1,k) <= 0.0) 
                {
                    // Interp to face center from face centroids
                    u_tmp_ip1 = linear_interp(0.0, fcx(i+1,j-1,k), u_ad(i+1,j-1,k), (-1.+fcx(i+1,j-2,k)), u_ad(i+1,j-2,k));
                    x_hat_ip1 = linear_interp(0.0, fcx(i+1,j-1,k), xhat(i+1,j-1,k), (-1.+fcx(i+1,j-2,k)), xhat(i+1,j-2,k));
                }
                else if (fcx(i+1,j-1,k) > 0.0) 
                {
                    // Interp to face center from face centroids
                    u_tmp_ip1 = linear_interp(0.0, fcx(i+1,j-1,k), u_ad(i+1,j-1,k), ( 1.+fcx(i+1,j,k)), u_ad(i+1,j,k));
                    x_hat_ip1 = linear_interp(0.0, fcx(i+1,j-1,k), xhat(i+1,j-1,k), ( 1.+fcx(i+1,j,k)), xhat(i+1,j,k));
                }

                u_tmp_l = 0.5 * (u_tmp_ip1 + u_tmp_i);
                 dvdx_l =       (x_hat_ip1 - x_hat_i);

                Real trans_stl = u_tmp_l * dvdx_l / dx;

                stl += - 0.5 * l_dt * trans_stl;

                if (vfrac_arr(i,j-1,k) > 0.)
                    stl += 0.5 * l_dt * f(i,j-1,k,n);
            }
        }

        // Top side of interface
        if (flag(i,j,k).isRegular())
        {
            sth = yhi(i,j,k,n) - (0.25*l_dt/dx)*(u_ad(i+1,j  ,k  )+u_ad(i,j  ,k))*
                                                (xhat(i+1,j  ,k  )-xhat(i,j  ,k));

        } else { // should be single-valued since it is connected across this face

            Real dvdx_h;  //  (xhat(i+1,j,k  )-xhat(i,j,k)) if regular
            Real u_tmp_h; //  0.5 * (u_ad(i+1,j,k  )+u_ad(i,j,k)) if regular

            Real trans_sth; 

            sth = yhi(i,j,k,n);

            // If either x-face is covered, set the tranxverse term to zero
            if (apx(i,j,k) == 0.0 or apx(i+1,j,k) == 0.0) {
                trans_sth = 0.0;
            } else {
                // Tangential extrapolation in the y-direction at i-1/2
                if (apx(i,j,k) == 1.0)
                {
                    u_tmp_i = u_ad(i,j,k);
                    x_hat_i = xhat(i,j,k);
                }
                else if (fcx(i,j,k) <= 0.0) 
                {
                    // Interp to face center from face centroids
                    u_tmp_i = linear_interp(0.0, fcx(i,j,k), u_ad(i,j,k), (-1.+fcx(i,j-1,k)), u_ad(i,j-1,k));
                    x_hat_i = linear_interp(0.0, fcx(i,j,k), xhat(i,j,k), (-1.+fcx(i,j-1,k)), xhat(i,j-1,k));
                }
                else if (fcx(i,j-1,k) > 0.0) 
                {
                    // Interp to face center from face centroids
                    u_tmp_i = linear_interp(0.0, fcx(i,j,k), u_ad(i,j,k), ( 1.+fcx(i,j+1,k)), u_ad(i,j+1,k));
                    x_hat_i = linear_interp(0.0, fcx(i,j,k), xhat(i,j,k), ( 1.+fcx(i,j+1,k)), xhat(i,j+1,k));
                }

                // Tangential extrapolation in the y-direction at i+1/2
                if (apx(i+1,j,k) == 1.0)
                {
                    u_tmp_ip1 = u_ad(i+1,j,k);
                    x_hat_ip1 = xhat(i+1,j,k);
                }
                else if (fcx(i+1,j,k) <= 0.0) 
                {
                    // Interp to face center from face centroids
                    u_tmp_ip1 = linear_interp(0.0, fcx(i+1,j,k), u_ad(i+1,j,k), (-1.+fcx(i+1,j-1,k)), u_ad(i+1,j-1,k));
                    x_hat_ip1 = linear_interp(0.0, fcx(i+1,j,k), xhat(i+1,j,k), (-1.+fcx(i+1,j-1,k)), xhat(i+1,j-1,k));
                }
                else if (fcx(i+1,j-1,k) > 0.0) 
                {
                    // Interp to face center from face centroids
                    u_tmp_ip1 = linear_interp(0.0, fcx(i+1,j,k), u_ad(i+1,j,k), ( 1.+fcx(i+1,j+1,k)), u_ad(i+1,j+1,k));
                    x_hat_ip1 = linear_interp(0.0, fcx(i+1,j,k), xhat(i+1,j,k), ( 1.+fcx(i+1,j+1,k)), xhat(i+1,j+1,k));
                }

                u_tmp_h = 0.5 * (u_tmp_ip1 + u_tmp_i);
                 dvdx_h =       (x_hat_ip1 - x_hat_i);

                trans_sth = u_tmp_h * dvdx_h / dx;

                sth +=  - 0.5 * l_dt * trans_sth;

                if (vfrac_arr(i,j  ,k) > 0.)
                    sth += 0.5 * l_dt * f(i,j  ,k,n);
            }
        }

        if (l_use_mac_phi_in_godunov)
        {
            // Note that the getFluxes call returns (-1/rho G^Mac phi) so we use the negative here
            Real gphi_y = -gmacphi_y(i,j,k);
            stl -= 0.5 * l_dt * gphi_y;
            sth -= 0.5 * l_dt * gphi_y;
        }

        Godunov_cc_ybc_lo(i, j, k, n, q, stl, sth, bc.lo(1), dlo.y, true);
        Godunov_cc_ybc_hi(i, j, k, n, q, stl, sth, bc.hi(1), dhi.y, true);


        // Prevent backflow
        if ( (j==dlo.y) and (bc.lo(1) == BCType::foextrap || bc.lo(1) == BCType::hoextrap) )
        {
            sth = amrex::min(sth,0.);
            stl = sth;
        }
        if ( (j==dhi.y+1) and (bc.hi(1) == BCType::foextrap || bc.hi(1) == BCType::hoextrap) )
        {
            stl = amrex::max(stl,0.);
            sth = stl;
        }

        Real st = ( (stl+sth) >= 0.) ? stl : sth;
        bool ltm = ( (stl <= 0. && sth >= 0.) || (amrex::Math::abs(stl+sth) < small_vel) );
        qy(i,j,k) = ltm ? 0. : st;

        if (l_use_mac_phi_in_godunov)
        {
            // Note that the getFluxes call returns (-1/rho G^Mac phi) so we use the negative here
            Real gphi_y = -gmacphi_y(i,j,k);
            qy(i,j,k) += 0.5 * l_dt * gphi_y;
        }

        } else {
            qy(i,j,k) = 0.;
        }
    });
}
