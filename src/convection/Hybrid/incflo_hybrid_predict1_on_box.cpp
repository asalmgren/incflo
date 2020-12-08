#include <Hybrid.H>
#include <AMReX_Slopes_K.H>
#include <AMReX_MultiFabUtil.H>

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
hybrid::predict_vels_on_faces ( AMREX_D_DECL(Box const& ubx, 
                                             Box const& vbx, 
                                             Box const& wbx),
                                AMREX_D_DECL(Array4<Real> const& fx, 
                                             Array4<Real> const& fy,
                                             Array4<Real> const& fz), 
                                Array4<Real const> const& vel,
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

    // 
    // Make fluxes on x-faces
    // 

    // At an ext_dir or hoextrap boundary, 
    //    the boundary value is on the face, not cell center.
    auto extdir_lohi = has_extdir_or_ho(h_bcrec.data(), ncomp, static_cast<int>(Direction::x));
    bool has_extdir_or_ho_lo = extdir_lohi.first;
    bool has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_ilo >= ubx.smallEnd(0)-1) or
        (has_extdir_or_ho_hi and domain_ihi <= ubx.bigEnd(0)))
    {
        amrex::ParallelFor(ubx, [vel,fx,domain_ilo,domain_ihi,d_bcrec]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            bool extdir_or_ho_ilo = (d_bcrec[0].lo(0) == BCType::ext_dir) or
                                    (d_bcrec[0].lo(0) == BCType::hoextrap);
            bool extdir_or_ho_ihi = (d_bcrec[0].hi(0) == BCType::ext_dir) or
                                    (d_bcrec[0].hi(0) == BCType::hoextrap);
            Real upls = vel(i  ,j,k,0) - 0.5 * amrex_calc_xslope_extdir(
                 i  ,j,k,0,order,vel,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);
            Real umns = vel(i-1,j,k,0) + 0.5 * amrex_calc_xslope_extdir(
                 i-1,j,k,0,order,vel,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);

                 extdir_or_ho_ilo = (d_bcrec[1].lo(0) == BCType::ext_dir) or
                                    (d_bcrec[1].lo(0) == BCType::hoextrap);
                 extdir_or_ho_ihi = (d_bcrec[1].hi(0) == BCType::ext_dir) or
                                    (d_bcrec[1].hi(0) == BCType::hoextrap);
            Real vpls = vel(i  ,j,k,1) - 0.5 * amrex_calc_xslope_extdir(
                 i  ,j,k,1,order,vel,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);
            Real vmns = vel(i-1,j,k,1) + 0.5 * amrex_calc_xslope_extdir(
                 i-1,j,k,1,order,vel,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);

#if (AMREX_SPACEDIM == 3)
                 extdir_or_ho_ilo = (d_bcrec[2].lo(0) == BCType::ext_dir) or
                                    (d_bcrec[2].lo(0) == BCType::hoextrap);
                 extdir_or_ho_ihi = (d_bcrec[2].hi(0) == BCType::ext_dir) or
                                    (d_bcrec[2].hi(0) == BCType::hoextrap);
            Real wpls = vel(i  ,j,k,2) - 0.5 * amrex_calc_xslope_extdir(
                 i  ,j,k,2,order,vel,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);
            Real wmns = vel(i-1,j,k,2) + 0.5 * amrex_calc_xslope_extdir(
                 i-1,j,k,2,order,vel,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi);
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
                u_val = vel(i-1,j,k,0);
            } else if (i == domain_ihi+1 && (d_bcrec[0].hi(0) == BCType::ext_dir)) {
                u_val = vel(i  ,j,k,0);
            }

            if (i == domain_ilo && (d_bcrec[1].lo(0) == BCType::ext_dir)) {
                v_val = vel(i-1,j,k,1);
            } else if (i == domain_ihi+1 && (d_bcrec[1].hi(0) == BCType::ext_dir)) {
                v_val = vel(i  ,j,k,1);
            }

#if (AMREX_SPACEDIM == 3)
            if (i == domain_ilo && (d_bcrec[2].lo(0) == BCType::ext_dir)) {
                w_val = vel(i-1,j,k,2);
            } else if (i == domain_ihi+1 && (d_bcrec[2].hi(0) == BCType::ext_dir)) {
                w_val = vel(i  ,j,k,2);
            }
#endif

            fx(i,j,k,0) = u_val*u_val;
            fx(i,j,k,1) = v_val*u_val;
#if (AMREX_SPACEDIM == 3)
            fx(i,j,k,2) = w_val*u_val;
#endif
        });
    }
    else
    {
        amrex::ParallelFor(ubx, [vel,fx]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            Real upls = vel(i  ,j,k,0) - 0.5 * amrex_calc_xslope(i  ,j,k,0,order,vel);
            Real umns = vel(i-1,j,k,0) + 0.5 * amrex_calc_xslope(i-1,j,k,0,order,vel);

            Real vpls = vel(i  ,j,k,1) - 0.5 * amrex_calc_xslope(i  ,j,k,1,order,vel);
            Real vmns = vel(i-1,j,k,1) + 0.5 * amrex_calc_xslope(i-1,j,k,1,order,vel);

#if (AMREX_SPACEDIM == 3)
            Real wpls = vel(i  ,j,k,2) - 0.5 * amrex_calc_xslope(i  ,j,k,2,order,vel);
            Real wmns = vel(i-1,j,k,2) + 0.5 * amrex_calc_xslope(i-1,j,k,2,order,vel);
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

            fx(i,j,k,0) = u_val*u_val;
            fx(i,j,k,1) = v_val*u_val;
#if (AMREX_SPACEDIM == 3)
            fx(i,j,k,2) = w_val*u_val;
#endif
        });
    }

    // 
    // Make fluxes on y-faces
    // 

    // At an ext_dir or hoextrap boundary, 
    //    the boundary value is on the face, not cell center.
    extdir_lohi = has_extdir_or_ho(h_bcrec.data(), ncomp, static_cast<int>(Direction::y));
    has_extdir_or_ho_lo = extdir_lohi.first;
    has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_jlo >= vbx.smallEnd(1)-1) or
        (has_extdir_or_ho_hi and domain_jhi <= vbx.bigEnd(1)))
    {
        amrex::ParallelFor(vbx, [vel,fy,domain_jlo,domain_jhi,d_bcrec]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            int order = 2;
    
            bool extdir_or_ho_jlo = (d_bcrec[0].lo(1) == BCType::ext_dir) or
                                    (d_bcrec[0].lo(1) == BCType::hoextrap);
            bool extdir_or_ho_jhi = (d_bcrec[0].hi(1) == BCType::ext_dir) or
                                    (d_bcrec[0].hi(1) == BCType::hoextrap);
            Real upls = vel(i,j  ,k,0) - 0.5 * amrex_calc_yslope_extdir(
                 i,j,k,0,order,vel,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            Real umns = vel(i,j-1,k,0) + 0.5 * amrex_calc_yslope_extdir(
                 i,j-1,0,1,order,vel,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
    
                 extdir_or_ho_jlo = (d_bcrec[1].lo(1) == BCType::ext_dir) or
                                    (d_bcrec[1].lo(1) == BCType::hoextrap);
                 extdir_or_ho_jhi = (d_bcrec[1].hi(1) == BCType::ext_dir) or
                                    (d_bcrec[1].hi(1) == BCType::hoextrap);
            Real vpls = vel(i,j  ,k,1) - 0.5 * amrex_calc_yslope_extdir(
                 i,j,k,1,order,vel,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            Real vmns = vel(i,j-1,k,1) + 0.5 * amrex_calc_yslope_extdir(
                 i,j-1,k,1,order,vel,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);

#if (AMREX_SPACEDIM == 3)
                 extdir_or_ho_jlo = (d_bcrec[2].lo(1) == BCType::ext_dir) or
                                    (d_bcrec[2].lo(1) == BCType::hoextrap);
                 extdir_or_ho_jhi = (d_bcrec[2].hi(1) == BCType::ext_dir) or
                                    (d_bcrec[2].hi(1) == BCType::hoextrap);
            Real wpls = vel(i,j  ,k,2) - 0.5 * amrex_calc_yslope_extdir(
                 i,j,k,2,order,vel,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            Real wmns = vel(i,j-1,k,2) + 0.5 * amrex_calc_yslope_extdir(
                 i,j-1,k,2,order,vel,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
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

            if (j == domain_jlo && (d_bcrec[0].lo(1) == BCType::ext_dir)) {
                u_val = vel(i,j-1,k,0);
            } else if (j == domain_jhi+1 && (d_bcrec[0].hi(1) == BCType::ext_dir)) {
                u_val = vel(i,j  ,k,0);
            }
            fy(i,j,k,0) = u_val*v_val;

            if (j == domain_jlo && (d_bcrec[1].lo(1) == BCType::ext_dir)) {
                v_val = vel(i,j-1,k,1);
            } else if (j == domain_jhi+1 && (d_bcrec[1].hi(1) == BCType::ext_dir)) {
                v_val = vel(i,j  ,k,1);
            }
            fy(i,j,k,1) = v_val*v_val;

#if (AMREX_SPACEDIM == 3)
            if (j == domain_jlo && (d_bcrec[2].lo(1) == BCType::ext_dir)) {
                w_val = vel(i,j-1,k,2);
            } else if (j == domain_jhi+1 && (d_bcrec[2].hi(1) == BCType::ext_dir)) {
                w_val = vel(i,j  ,k,2);
            }
            fy(i,j,k,2) = w_val*v_val;
#endif
        });
    }
    else
    {
        amrex::ParallelFor(vbx, [vel,fy]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            Real upls = vel(i,j  ,k,0) - 0.5 * amrex_calc_yslope(i,j  ,k,0,order,vel);
            Real umns = vel(i,j-1,k,0) + 0.5 * amrex_calc_yslope(i,j-1,k,0,order,vel);

            Real vpls = vel(i,j  ,k,1) - 0.5 * amrex_calc_yslope(i,j  ,k,1,order,vel);
            Real vmns = vel(i,j-1,k,1) + 0.5 * amrex_calc_yslope(i,j-1,k,1,order,vel);

            Real u_val(0.5 * (upls + umns));
            Real v_val(0);

#if (AMREX_SPACEDIM == 3)
            Real wpls = vel(i,j  ,k,2) - 0.5 * amrex_calc_yslope(i,j  ,k,2,order,vel);
            Real wmns = vel(i,j-1,k,2) + 0.5 * amrex_calc_yslope(i,j-1,k,2,order,vel);

            Real w_val(0.5 * (wpls + wmns));
#endif

            if (vmns >= 0.0 or vpls <= 0.0) {

                Real avg = 0.5 * (vpls + vmns);

                if (avg >= small_vel) {
                    u_val = vel(i,j-1,k,0);
                    v_val = vel(i,j-1,k,1);
#if (AMREX_SPACEDIM == 3)
                    w_val = vel(i,j-1,k,2);
#endif
                }
                else if (avg <= -small_vel) {
                    u_val = vel(i,j  ,k,0);
                    v_val = vel(i,j  ,k,1);
#if (AMREX_SPACEDIM == 3)
                    w_val = vel(i,j  ,k,2);
#endif
                }
            }

            fy(i,j,k,0) = u_val*v_val;
            fy(i,j,k,1) = v_val*v_val;
#if (AMREX_SPACEDIM == 3)
            fy(i,j,k,2) = w_val*v_val;
#endif
        });
    }

#if (AMREX_SPACEDIM == 3)
    // 
    // Make fluxes on z-faces
    // 
    // At an ext_dir or hoextrap boundary, 
    //    the boundary value is on the face, not cell center.
    extdir_lohi = has_extdir_or_ho(h_bcrec.data(), ncomp, static_cast<int>(Direction::z));
    has_extdir_or_ho_lo = extdir_lohi.first;
    has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_klo >= wbx.smallEnd(2)-1) or
        (has_extdir_or_ho_hi and domain_khi <= wbx.bigEnd(2)))
    {
        amrex::ParallelFor(wbx, [vel,fz,domain_klo,domain_khi,d_bcrec]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real ucc_pls = vel(i,j,k  ,0);
            const Real ucc_mns = vel(i,j,k-1,0);

            const Real vcc_pls = vel(i,j,k  ,1);
            const Real vcc_mns = vel(i,j,k-1,1);

            const Real wcc_pls = vel(i,j,k  ,2);
            const Real wcc_mns = vel(i,j,k-1,2);

            int order = 2;

            bool extdir_or_ho_klo = (d_bcrec[0].lo(2) == BCType::ext_dir) or
                                    (d_bcrec[0].lo(2) == BCType::hoextrap);
            bool extdir_or_ho_khi = (d_bcrec[0].hi(2) == BCType::ext_dir) or
                                    (d_bcrec[0].hi(2) == BCType::hoextrap);
            Real upls = vel(i,j,k  ,0) - 0.5 * amrex_calc_zslope_extdir(
                 i,j,k  ,0,order,vel,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
            Real umns = vel(i,j,k-1,0) + 0.5 * amrex_calc_zslope_extdir(
                 i,j,k-1,0,order,vel,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);

                 extdir_or_ho_klo = (d_bcrec[1].lo(2) == BCType::ext_dir) or
                                    (d_bcrec[1].lo(2) == BCType::hoextrap);
                 extdir_or_ho_khi = (d_bcrec[1].hi(2) == BCType::ext_dir) or
                                    (d_bcrec[1].hi(2) == BCType::hoextrap);
            Real vpls = vel(i,j,k  ,1) - 0.5 * amrex_calc_zslope_extdir(
                 i,j,k  ,1,order,vel,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
            Real vmns = vel(i,j,k-1,1) + 0.5 * amrex_calc_zslope_extdir(
                 i,j,k-1,1,order,vel,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);

                 extdir_or_ho_klo = (d_bcrec[2].lo(2) == BCType::ext_dir) or
                                    (d_bcrec[2].lo(2) == BCType::hoextrap);
                 extdir_or_ho_khi = (d_bcrec[2].hi(2) == BCType::ext_dir) or
                                    (d_bcrec[2].hi(2) == BCType::hoextrap);
            Real wpls = vel(i,j,k  ,2) - 0.5 * amrex_calc_zslope_extdir(
                 i,j,k  ,2,order,vel,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
            Real wmns = vel(i,j,k-1,2) + 0.5 * amrex_calc_zslope_extdir(
                 i,j,k-1,2,order,vel,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);

            Real w_val(0);
            Real u_val(0.5*(upls+umns));
            Real v_val(0.5*(vpls+vmns));

            if (wmns >= 0.0 or wpls <= 0.0) 
            {
                Real avg = 0.5 * (wpls + wmns);

                if (avg >= small_vel) {
                    u_val = umns;
                    v_val = vmns;
                    w_val = wmns;
                }
                else if (avg <= -small_vel) {
                    u_val = upls;
                    v_val = vpls;
                    w_val = wpls;
                }
            }

            if (k == domain_klo && (d_bcrec[0].lo(2) == BCType::ext_dir)) {
                u_val = ucc_mns;
            } else if (k == domain_khi+1 && (d_bcrec[0].hi(2) == BCType::ext_dir)) {
                u_val = ucc_pls;
            }
            fz(i,j,k,0) = u_val*w_val;

            if (k == domain_klo && (d_bcrec[1].lo(2) == BCType::ext_dir)) {
                v_val = vcc_mns;
            } else if (k == domain_khi+1 && (d_bcrec[1].hi(2) == BCType::ext_dir)) {
                v_val = vcc_pls;
            }
            fz(i,j,k,1) = v_val*w_val;

            if (k == domain_klo && (d_bcrec[2].lo(2) == BCType::ext_dir)) {
                w_val = wcc_mns;
            } else if (k == domain_khi+1 && (d_bcrec[2].hi(2) == BCType::ext_dir)) {
                w_val = wcc_pls;
            }
            fz(i,j,k,2) = w_val*w_val;
        });
    }
    else
    {
        amrex::ParallelFor(wbx, [vel,fz]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            Real upls = vel(i,j,k  ,0) - 0.5 * amrex_calc_zslope(i,j,k  ,0,order,vel);
            Real umns = vel(i,j,k-1,0) + 0.5 * amrex_calc_zslope(i,j,k-1,0,order,vel);

            Real vpls = vel(i,j,k  ,1) - 0.5 * amrex_calc_zslope(i,j,k  ,1,order,vel);
            Real vmns = vel(i,j,k-1,1) + 0.5 * amrex_calc_zslope(i,j,k-1,1,order,vel);

            Real wpls = vel(i,j,k  ,2) - 0.5 * amrex_calc_zslope(i,j,k  ,2,order,vel);
            Real wmns = vel(i,j,k-1,2) + 0.5 * amrex_calc_zslope(i,j,k-1,2,order,vel);

            Real w_val(0);
            Real u_val(0.5*(upls+umns));
            Real v_val(0.5*(vpls+vmns));

            if (wmns >= 0.0 or wpls <= 0.0) 
            {
                Real avg = 0.5 * (wpls + wmns);

                if (avg >= small_vel) {
                    u_val = umns;
                    v_val = vmns;
                    w_val = wmns;
                }
                else if (avg <= -small_vel) {
                    u_val = upls;
                    v_val = vpls;
                    w_val = wpls;
                }
            }

            fz(i,j,k,0) = u_val*w_val;
            fz(i,j,k,1) = v_val*w_val;
            fz(i,j,k,2) = w_val*w_val;
        });
    }
#endif
}

