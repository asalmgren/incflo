#include <Hybrid.H>
#include <AMReX_Slopes_K.H>

using namespace amrex;

namespace {
    std::pair<bool,bool> has_extdir_or_ho (BCRec const* bcrec, int ncomp, int dir)
    {
        std::pair<bool,bool> r{false,false};
        for (int n = 0; n < ncomp; ++n) {
            r.first = r.first 
                 or (bcrec[n].lo(dir) == BCType::ext_dir)
                 or (bcrec[n].lo(dir) == BCType::hoextrap);
            r.second = r.second 
                 or (bcrec[n].hi(dir) == BCType::ext_dir)
                 or (bcrec[n].hi(dir) == BCType::hoextrap);
        }
        return r;
    }
}

void 
hybrid::predict_vels_on_faces (int lev, 
                               AMREX_D_DECL(MultiFab& u_mac, 
                                            MultiFab& v_mac,
                                            MultiFab& w_mac), 
                               MultiFab const& vel,
                               MultiFab& dudt,
                               Vector<BCRec> const& h_bcrec,
                                   BCRec  const* d_bcrec,
#ifdef AMREX_USE_EB
                               EBFArrayBoxFactory const* ebfact,
#endif
                               Geometry& geom)
{
#ifdef AMREX_USE_EB
    auto const& flags = ebfact->getMultiEBCellFlagFab();
    auto const& fcent = ebfact->getFaceCent();
    auto const& ccent = ebfact->getCentroid();
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            AMREX_D_TERM(Box const& ubx = mfi.nodaltilebox(0);,
                         Box const& vbx = mfi.nodaltilebox(1);,
                         Box const& wbx = mfi.nodaltilebox(2););
            AMREX_D_TERM(Array4<Real> const& u = u_mac.array(mfi);,
                         Array4<Real> const& v = v_mac.array(mfi);,
                         Array4<Real> const& w = w_mac.array(mfi););
            Array4<Real const> const& vcc = vel.const_array(mfi);
            Array4<Real      > const& dudt_arr = dudt.array(mfi);

            Box const& bx = mfi.tilebox();

            Box tmpbox = amrex::surroundingNodes(bx);

            FArrayBox tmpfab(tmpbox, AMREX_SPACEDIM*AMREX_SPACEDIM);
            Elixir eli = tmpfab.elixir();

            AMREX_D_TERM(Array4<Real> fx = tmpfab.array(0);,
                         Array4<Real> fy = tmpfab.array(3);,
                         Array4<Real> fz = tmpfab.array(6););

#ifdef AMREX_USE_EB
            EBCellFlagFab const& flagfab = flags[mfi];
            Array4<EBCellFlag const> const& flagarr = flagfab.const_array();
            auto const typ = flagfab.getType(amrex::grow(bx,2));
            if (typ == FabType::covered)
            {
#if (AMREX_SPACEDIM == 3)
                amrex::ParallelFor(ubx, vbx, wbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { u(i,j,k) = 0.0; },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { v(i,j,k) = 0.0; },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { w(i,j,k) = 0.0; });
#else
                amrex::ParallelFor(ubx, vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { u(i,j,k) = 0.0; },
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { v(i,j,k) = 0.0; });
#endif
            }
            else if (typ == FabType::singlevalued)
            {
                AMREX_D_TERM(Array4<Real const> const& fcx = fcent[0]->const_array(mfi);,
                             Array4<Real const> const& fcy = fcent[1]->const_array(mfi);,
                             Array4<Real const> const& fcz = fcent[2]->const_array(mfi););
                Array4<Real const> const& ccc = ccent.const_array(mfi);
                hybrid::predict_vels_on_faces_eb(bx,AMREX_D_DECL(ubx,vbx,wbx),
                                                 AMREX_D_DECL(u,v,w),vcc,flagarr,AMREX_D_DECL(fcx,fcy,fcz),ccc,
                                                 h_bcrec,d_bcrec,geom);
//              hybrid::compute_convective_rate_eb(bx, AMREX_SPACEDIM, dUdt_tmp, AMREX_D_DECL(fx, fy, fz),
//                                                 flag, vfrac, AMREX_D_DECL(apx, apy, apz), geom);
            }
            else
#endif
            {
                hybrid::predict_vels_on_faces(bx,AMREX_D_DECL(ubx,vbx,wbx),
                                              AMREX_D_DECL(u,v,w), AMREX_D_DECL(fx,fy,fz),
                                              vcc,h_bcrec,d_bcrec,geom);
                hybrid::compute_convective_rate(bx, AMREX_SPACEDIM, dudt_arr, AMREX_D_DECL(fx, fy, fz), geom);
            }
        }
    }
}

void 
hybrid::predict_vels_on_faces ( Box const& bx,
                                AMREX_D_DECL(Box const& ubx, 
                                             Box const& vbx, 
                                             Box const& wbx),
                                AMREX_D_DECL(Array4<Real> const& u, 
                                             Array4<Real> const& v,
                                             Array4<Real> const& w), 
                                AMREX_D_DECL(Array4<Real> const& fx, 
                                             Array4<Real> const& fy,
                                             Array4<Real> const& fz), 
                                Array4<Real const> const& vcc,
                                Vector<BCRec> const& h_bcrec,
                                       BCRec  const* d_bcrec,
                                Geometry& geom )
{
    constexpr Real small_vel = 1.e-10;

    int ncomp = AMREX_SPACEDIM; // This is only used because h_bcrec and d_bcrec hold the 
                                // bc's for all three velocity components

    const Box& domain_box = geom.Domain();
    const int domain_ilo = domain_box.smallEnd(0);
    const int domain_ihi = domain_box.bigEnd(0);
    const int domain_jlo = domain_box.smallEnd(1);
    const int domain_jhi = domain_box.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
    const int domain_klo = domain_box.smallEnd(2);
    const int domain_khi = domain_box.bigEnd(2);
#endif

    FArrayBox tmpfab(amrex::grow(bx,1), AMREX_SPACEDIM);

    // At an ext_dir or hoextrap boundary, 
    //    the boundary value is on the face, not cell center.
    auto extdir_lohi = has_extdir_or_ho(h_bcrec.data(), ncomp, static_cast<int>(Direction::x));
    bool has_extdir_or_ho_lo = extdir_lohi.first;
    bool has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_ilo >= ubx.smallEnd(0)-1) or
        (has_extdir_or_ho_hi and domain_ihi <= ubx.bigEnd(0)))
    {
        amrex::ParallelFor(ubx, [vcc,fx,domain_ilo,domain_ihi,d_bcrec]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            bool extdir_or_ho_ilo = (d_bcrec[0].lo(0) == BCType::ext_dir) or
                                    (d_bcrec[0].lo(0) == BCType::hoextrap);
            bool extdir_or_ho_ihi = (d_bcrec[0].hi(0) == BCType::ext_dir) or
                                    (d_bcrec[0].hi(0) == BCType::hoextrap);

            int order = 2;

            Real upls = vcc(i  ,j,k,0) - 0.5 * amrex_calc_xslope_extdir(
                 i  ,j,k,0,order,vcc,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);
            Real umns = (i-1,j,k,0) + 0.5 * amrex_calc_xslope_extdir(
                 i-1,j,k,0,order,vcc,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);

            Real vpls = vcc(i  ,j,k,1) - 0.5 * amrex_calc_xslope_extdir(
                 i  ,j,k,1,order,vcc,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);
            Real vmns = vcc(i-1,j,k,1) + 0.5 * amrex_calc_xslope_extdir(
                 i-1,j,k,1,order,vcc,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);

#if (AMREX_SPACEDIM == 3)
            Real wpls = vcc(i  ,j,k,2) - 0.5 * amrex_calc_xslope_extdir(
                 i  ,j,k,2,order,vcc,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);
            Real wmns = vcc(i-1,j,k,2) + 0.5 * amrex_calc_xslope_extdir(
                 i-1,j,k,2,order,vcc,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);
#endif

            Real u_val = 0.;
            Real v_val = 0.5 * (vpls + vmns);
#if (AMREX_SPACEDIM == 3)
            Real w_val = 0.5 * (wpls + wmns);
#endif

            if (umns >= 0.0 or upls <= 0.0) {
                
                Real avg = 0.5 * (upls + umns);

                if (avg >= small_vel) {
                    u_val = umns;
                    v_val = vmns;
#if (AMREX_SPACEDIM == 3)
                    w_val = wmns;
#endif
                }
                else if (avg <= -small_vel){
                    u_val = upls;
                    v_val = vpls;
#if (AMREX_SPACEDIM == 3)
                    w_val = wpls;
#endif
                }
            }

            if (i == domain_ilo && (d_bcrec[0].lo(0) == BCType::ext_dir)) {
                u_val = vcc(i-1,j,k,0);
                v_val = vcc(i-1,j,k,1);
#if (AMREX_SPACEDIM == 3)
                w_val = vcc(i-1,j,k,2);
#endif
            } else if (i == domain_ihi+1 && (d_bcrec[0].hi(0) == BCType::ext_dir)) {
                u_val = vcc(i  ,j,k,0);
                v_val = vcc(i  ,j,k,1);
#if (AMREX_SPACEDIM == 3)
                w_val = vcc(i  ,j,k,2);
#endif
            }

            fx(i,j,k,0) = u_val;
            fx(i,j,k,1) = v_val;
#if (AMREX_SPACEDIM == 3)
            fx(i,j,k,2) = w_val;
#endif
        });
    }
    else
    {
        amrex::ParallelFor(ubx, [vcc,fx]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            Real upls = vcc(i  ,j,k,0) - 0.5 * amrex_calc_xslope(i  ,j,k,0,order,vcc);
            Real umns = vcc(i-1,j,k,0) + 0.5 * amrex_calc_xslope(i-1,j,k,0,order,vcc);

            Real vpls = vcc(i  ,j,k,1) - 0.5 * amrex_calc_xslope(i  ,j,k,1,order,vcc);
            Real vmns = vcc(i-1,j,k,1) + 0.5 * amrex_calc_xslope(i-1,j,k,1,order,vcc);

#if (AMREX_SPACEDIM == 3)
            Real wpls = vcc(i  ,j,k,2) - 0.5 * amrex_calc_xslope(i  ,j,k,2,order,vcc);
            Real wmns = vcc(i-1,j,k,2) + 0.5 * amrex_calc_xslope(i-1,j,k,2,order,vcc);
#endif

            Real u_val(0);
            Real v_val(0.5 * (vpls + vmns));
#if (AMREX_SPACEDIM == 3)
            Real w_val(0.5 * (wpls + wmns));
#endif

            if (umns >= 0.0 or upls <= 0.0) {
                
                Real avg = 0.5 * (upls + umns);
                
                if (avg >= small_vel) {
                    u_val = umns;
                    v_val = vmns;
#if (AMREX_SPACEDIM == 3)
                    w_val = wmns;
#endif
                }
                else if (avg <= -small_vel){
                    u_val = upls;
                    v_val = vpls;
#if (AMREX_SPACEDIM == 3)
                    w_val = wpls;
#endif
                }
            }

            fx(i,j,k,0) = u_val;
            fx(i,j,k,1) = v_val;
#if (AMREX_SPACEDIM == 3)
            fx(i,j,k,2) = w_val;
#endif
        });
    }

    // At an ext_dir or hoextrap boundary, 
    //    the boundary value is on the face, not cell center.
    extdir_lohi = has_extdir_or_ho(h_bcrec.data(), ncomp, static_cast<int>(Direction::y));
    has_extdir_or_ho_lo = extdir_lohi.first;
    has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_jlo >= vbx.smallEnd(1)-1) or
        (has_extdir_or_ho_hi and domain_jhi <= vbx.bigEnd(1)))
    {
        amrex::ParallelFor(vbx, [vcc,fy,domain_jlo,domain_jhi,d_bcrec]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            bool extdir_or_ho_jlo = (d_bcrec[1].lo(1) == BCType::ext_dir) or
                                    (d_bcrec[1].lo(1) == BCType::hoextrap);
            bool extdir_or_ho_jhi = (d_bcrec[1].hi(1) == BCType::ext_dir) or
                                    (d_bcrec[1].hi(1) == BCType::hoextrap);

            int order = 2;
    
            Real upls = vcc(i,j  ,k,0) - 0.5 * amrex_calc_yslope_extdir(
                 i,j,k,0,order,vcc,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            Real umns = vcc(i,j-1,k,0) + 0.5 * amrex_calc_yslope_extdir(
                 i,j-1,0,1,order,vcc,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
    
            Real vpls = vcc(i,j  ,k,1) - 0.5 * amrex_calc_yslope_extdir(
                 i,j,k,1,order,vcc,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            Real vmns = vcc(i,j-1,k,1) + 0.5 * amrex_calc_yslope_extdir(
                 i,j-1,k,1,order,vcc,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);

#if (AMREX_SPACEDIM == 3)
            Real vpls = vcc(i,j  ,k,1) - 0.5 * amrex_calc_yslope_extdir(
                 i,j,k,1,order,vcc,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            Real vmns = vcc(i,j-1,k,1) + 0.5 * amrex_calc_yslope_extdir(
                 i,j-1,k,1,order,vcc,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
#endif

            Real u_val(0.5 * (upls + umns));
            Real v_val(0);
#if (AMREX_SPACEDIM == 3)
            Real w_val(0.5 * (wpls + wmns));
#endif

            if (vmns >= 0.0 or vpls <= 0.0) {

                Real avg = 0.5 * (vpls + vmns);

                if (avg >= small_vel) {
                    u_val = umns;
                    v_val = vmns;
#if (AMREX_SPACEDIM == 3)
                    w_val = wmns;
#endif
                }
                else if (avg <= -small_vel){
                    u_val = upls;
                    v_val = vpls;
#if (AMREX_SPACEDIM == 3)
                    w_val = wpls;
#endif
                }
            }

            if (j == domain_jlo && (d_bcrec[1].lo(1) == BCType::ext_dir)) {
                u_val = vcc(i,j-1,k,0);
                v_val = vcc(i,j-1,k,1);
#if (AMREX_SPACEDIM == 3)
                w_val = vcc(i,j-1,k,2);
#endif
            } else if (j == domain_jhi+1 && (d_bcrec[1].hi(1) == BCType::ext_dir)) {
                u_val = vcc(i,j  ,k,0);
                v_val = vcc(i,j  ,k,1);
#if (AMREX_SPACEDIM == 3)
                w_val = vcc(i,j  ,k,2);
#endif
            }

            fy(i,j,k,0) = u_val;
            fy(i,j,k,1) = v_val;
#if (AMREX_SPACEDIM == 3)
            fy(i,j,k,2) = w_val;
#endif
        });
    }
    else
    {
        amrex::ParallelFor(vbx, [vcc,fy]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            Real upls = vcc(i,j  ,k,0) - 0.5 * amrex_calc_yslope(i,j  ,k,0,order,vcc);
            Real umns = vcc(i,j-1,k,0) + 0.5 * amrex_calc_yslope(i,j-1,k,0,order,vcc);

            Real vpls = vcc(i,j  ,k,1) - 0.5 * amrex_calc_yslope(i,j  ,k,1,order,vcc);
            Real vmns = vcc(i,j-1,k,1) + 0.5 * amrex_calc_yslope(i,j-1,k,1,order,vcc);
#if (AMREX_SPACEDIM == 3)
            Real wpls = vcc(i,j  ,k,2) - 0.5 * amrex_calc_yslope(i,j  ,k,2,order,vcc);
            Real wmns = vcc(i,j-1,k,2) + 0.5 * amrex_calc_yslope(i,j-1,k,2,order,vcc);
#endif

            Real u_val(0.5 * (upls + umns));
            Real v_val(0);
#if (AMREX_SPACEDIM == 3)
            Real w_val(0.5 * (wpls + wmns));
#endif

            if (vmns >= 0.0 or vpls <= 0.0) {

                Real avg = 0.5 * (vpls + vmns);

                if (avg >= small_vel) {
                    u_val = vcc(i,j-1,k,0);
                    v_val = vcc(i,j-1,k,1);
#if (AMREX_SPACEDIM == 3)
                    v_val = vcc(i,j-1,k,2);
#endif
                }
                else if (avg <= -small_vel) {
                    u_val = vcc(i,j  ,k,0);
                    v_val = vcc(i,j  ,k,1);
#if (AMREX_SPACEDIM == 3)
                    v_val = vcc(i,j  ,k,2);
#endif
                }
            }

            fy(i,j,k,0) = u_val;
            fy(i,j,k,1) = v_val;
#if (AMREX_SPACEDIM == 3)
            fy(i,j,k,2) = w_val;
#endif
        });
    }

#if (AMREX_SPACEDIM == 3)
    // At an ext_dir or hoextrap boundary, 
    //    the boundary value is on the face, not cell center.
    extdir_lohi = has_extdir_or_ho(h_bcrec.data(), ncomp, static_cast<int>(Direction::z));
    has_extdir_or_ho_lo = extdir_lohi.first;
    has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_klo >= wbx.smallEnd(2)-1) or
        (has_extdir_or_ho_hi and domain_khi <= wbx.bigEnd(2)))
    {
        amrex::ParallelFor(wbx, [vcc,domain_klo,domain_khi,w,d_bcrec]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            bool extdir_or_ho_klo = (d_bcrec[2].lo(2) == BCType::ext_dir) or
                                    (d_bcrec[2].lo(2) == BCType::hoextrap);
            bool extdir_or_ho_khi = (d_bcrec[2].hi(2) == BCType::ext_dir) or
                                    (d_bcrec[2].hi(2) == BCType::hoextrap);

            const Real vcc_pls = vcc(i,j,k,2);
            const Real vcc_mns = vcc(i,j,k-1,2);

            int order = 2;

            Real wpls = vcc_pls - 0.5 * amrex_calc_zslope_extdir(
                 i,j,k  ,2,order,vcc,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
            Real wmns = vcc_mns + 0.5 * amrex_calc_zslope_extdir(
                 i,j,k-1,2,order,vcc,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);

            Real w_val(0);

            if (wmns >= 0.0 or wpls <= 0.0) {
                Real avg = 0.5 * (wpls + wmns);

                if (avg >= small_vel) {
                    w_val = wmns;
                }
                else if (avg <= -small_vel) {
                    w_val = wpls;
                }
            }

            if (k == domain_klo && (d_bcrec[2].lo(2) == BCType::ext_dir)) {
                w_val = vcc_mns;
            } else if (k == domain_khi+1 && (d_bcrec[2].hi(2) == BCType::ext_dir)) {
                w_val = vcc_pls;
            }

            w(i,j,k) = w_val;
        });
    }
    else
    {
        amrex::ParallelFor(wbx, [vcc,w]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            Real wpls = vcc(i,j,k  ,2) - 0.5 * amrex_calc_zslope(i,j,k  ,2,order,vcc);
            Real wmns = vcc(i,j,k-1,2) + 0.5 * amrex_calc_zslope(i,j,k-1,2,order,vcc);

            Real w_val(0);

            if (wmns >= 0.0 or wpls <= 0.0) {
                Real avg = 0.5 * (wpls + wmns);

                if (avg >= small_vel) {
                    w_val = wmns;
                }
                else if (avg <= -small_vel) {
                    w_val = wpls;
                }
            }

            w(i,j,k) = w_val;
        });
    }
#endif
}

void 
hybrid::compute_convective_rate (Box const& bx, int ncomp,
                                Array4<Real> const& dUdt,
                                AMREX_D_DECL(Array4<Real const> const& fx,
                                             Array4<Real const> const& fy,
                                             Array4<Real const> const& fz),
                                Geometry& geom)
{
    const auto dxinv = geom.InvCellSizeArray();
    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
#if (AMREX_SPACEDIM == 3)
        dUdt(i,j,k,n) = dxinv[0] * (fx(i,j,k,n) - fx(i+1,j,k,n))
            +           dxinv[1] * (fy(i,j,k,n) - fy(i,j+1,k,n))
            +           dxinv[2] * (fz(i,j,k,n) - fz(i,j,k+1,n));
#else
        dUdt(i,j,k,n) = dxinv[0] * (fx(i,j,k,n) - fx(i+1,j,k,n))
            +           dxinv[1] * (fy(i,j,k,n) - fy(i,j+1,k,n));
#endif
    });
}

#ifdef AMREX_USE_EB
void 
hybrid::compute_convective_rate_eb (Box const& bx, int ncomp,
                                 Array4<Real> const& dUdt,
                                 AMREX_D_DECL(Array4<Real const> const& fx,
                                              Array4<Real const> const& fy,
                                              Array4<Real const> const& fz),
                                 Array4<EBCellFlag const> const& flag,
                                 Array4<Real const> const& vfrac,
                                 AMREX_D_DECL(Array4<Real const> const& apx,
                                              Array4<Real const> const& apy,
                                              Array4<Real const> const& apz),
                                 Geometry& geom)
{
    const auto dxinv = geom.InvCellSizeArray();
    const Box dbox   = geom.growPeriodicDomain(2);
    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
#if (AMREX_SPACEDIM == 3)
        if (!dbox.contains(IntVect(AMREX_D_DECL(i,j,k))) or flag(i,j,k).isCovered()) {
            dUdt(i,j,k,n) = 0.0;
        } else if (flag(i,j,k).isRegular()) {
            dUdt(i,j,k,n) = dxinv[0] * (fx(i,j,k,n) - fx(i+1,j,k,n))
                +           dxinv[1] * (fy(i,j,k,n) - fy(i,j+1,k,n))
                +           dxinv[2] * (fz(i,j,k,n) - fz(i,j,k+1,n));
        } else {
            dUdt(i,j,k,n) = (1.0/vfrac(i,j,k)) *
                ( dxinv[0] * (apx(i,j,k)*fx(i,j,k,n) - apx(i+1,j,k)*fx(i+1,j,k,n))
                + dxinv[1] * (apy(i,j,k)*fy(i,j,k,n) - apy(i,j+1,k)*fy(i,j+1,k,n))
                + dxinv[2] * (apz(i,j,k)*fz(i,j,k,n) - apz(i,j,k+1)*fz(i,j,k+1,n)) );
        }
#else
        if (!dbox.contains(IntVect(AMREX_D_DECL(i,j,k))) or flag(i,j,k).isCovered()) {
            dUdt(i,j,k,n) = 0.0;
        } else if (flag(i,j,k).isRegular()) {
            dUdt(i,j,k,n) = dxinv[0] * (fx(i,j,k,n) - fx(i+1,j,k,n))
                +           dxinv[1] * (fy(i,j,k,n) - fy(i,j+1,k,n));
        } else {
            dUdt(i,j,k,n) = (1.0/vfrac(i,j,k)) *
                ( dxinv[0] * (apx(i,j,k)*fx(i,j,k,n) - apx(i+1,j,k)*fx(i+1,j,k,n))
                + dxinv[1] * (apy(i,j,k)*fy(i,j,k,n) - apy(i,j+1,k)*fy(i,j+1,k,n)) );
        }
#endif
    });
}
#endif
