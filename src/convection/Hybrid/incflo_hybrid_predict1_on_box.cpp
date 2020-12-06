#include <Hybrid.H>
#include <AMReX_Slopes_K.H>
#ifdef AMREX_USE_EB
#include <AMReX_EB_slopes_K.H>
#endif
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
hybrid::predict_vels_on_faces ( Box const& bx,
                                AMREX_D_DECL(Box const& ubx, 
                                             Box const& vbx, 
                                             Box const& wbx),
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

            fx(i,j,k,0) = u_val*u_val;
            fx(i,j,k,1) = v_val*u_val;
#if (AMREX_SPACEDIM == 3)
            fx(i,j,k,2) = w_val*u_val;
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

            fx(i,j,k,0) = u_val*u_val;
            fx(i,j,k,1) = v_val*u_val;
#if (AMREX_SPACEDIM == 3)
            fx(i,j,k,2) = w_val*u_val;
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
            Real wpls = vcc(i,j  ,k,2) - 0.5 * amrex_calc_yslope_extdir(
                 i,j,k,2,order,vcc,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            Real wmns = vcc(i,j-1,k,2) + 0.5 * amrex_calc_yslope_extdir(
                 i,j-1,k,2,order,vcc,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
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

            fy(i,j,k,0) = u_val*v_val;
            fy(i,j,k,1) = v_val*v_val;
#if (AMREX_SPACEDIM == 3)
            fy(i,j,k,2) = w_val*v_val;
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

            fy(i,j,k,0) = u_val*v_val;
            fy(i,j,k,1) = v_val*v_val;
#if (AMREX_SPACEDIM == 3)
            fy(i,j,k,2) = w_val*v_val;
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

#ifdef AMREX_USE_EB
void 
hybrid::predict_vels_on_faces_eb (Box const& bx,
                                  AMREX_D_DECL(Box const& ubx, 
                                               Box const& vbx, 
                                               Box const& wbx),
                                  AMREX_D_DECL(Array4<Real> const& fx, 
                                               Array4<Real> const& fy,
                                               Array4<Real> const& fz), 
                                  Array4<Real const> const& vel,
                                  Array4<EBCellFlag const> const& flag,
                                  AMREX_D_DECL(Array4<Real const> const& fcx,
                                               Array4<Real const> const& fcy,
                                               Array4<Real const> const& fcz),
                                  Array4<Real const> const& ccc,
                                  Vector<BCRec> const& h_bcrec,
                                         BCRec  const* d_bcrec,
                               Geometry& geom)
{
    constexpr Real small_vel = 1.e-10;

    const Box& domain_box = geom.Domain();
    const int domain_ilo = domain_box.smallEnd(0);
    const int domain_ihi = domain_box.bigEnd(0);
    const int domain_jlo = domain_box.smallEnd(1);
    const int domain_jhi = domain_box.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
    const int domain_klo = domain_box.smallEnd(2);
    const int domain_khi = domain_box.bigEnd(2);
#endif

    int ncomp = AMREX_SPACEDIM; // This is only used because h_bcrec and d_bcrec hold the
                                // bc's for all three velocity components

    // At an ext_dir boundary, the boundary value is on the face, not cell center.
    auto extdir_lohi = has_extdir_or_ho(h_bcrec.data(), ncomp, static_cast<int>(Direction::x));
    bool has_extdir_or_ho_lo = extdir_lohi.first;
    bool has_extdir_or_ho_hi = extdir_lohi.second;

    // ****************************************************************************
    // Predict to x-faces
    // ****************************************************************************
    if ((has_extdir_or_ho_lo and domain_ilo >= ubx.smallEnd(0)-1) or
        (has_extdir_or_ho_hi and domain_ihi <= ubx.bigEnd(0)))
    {
        amrex::ParallelFor(Box(ubx),
        [fx,vel,flag,ccc,d_bcrec,
         AMREX_D_DECL(fcx,fcy,fcz),
         AMREX_D_DECL(domain_ilo,domain_jlo,domain_klo),
         AMREX_D_DECL(domain_ihi,domain_jhi,domain_khi)]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            AMREX_D_TERM(bool extdir_or_ho_ilo = (d_bcrec[0].lo(0) == BCType::ext_dir) or
                                                 (d_bcrec[0].lo(0) == BCType::hoextrap);,
                         bool extdir_or_ho_jlo = (d_bcrec[0].lo(1) == BCType::ext_dir) or
                                                 (d_bcrec[0].lo(1) == BCType::hoextrap);,
                         bool extdir_or_ho_klo = (d_bcrec[0].lo(2) == BCType::ext_dir) or
                                                 (d_bcrec[0].lo(2) == BCType::hoextrap););

            AMREX_D_TERM(bool extdir_or_ho_ihi = (d_bcrec[0].hi(0) == BCType::ext_dir) or
                                                 (d_bcrec[0].hi(0) == BCType::hoextrap);,
                         bool extdir_or_ho_jhi = (d_bcrec[0].hi(1) == BCType::ext_dir) or
                                                 (d_bcrec[0].hi(1) == BCType::hoextrap);,
                         bool extdir_or_ho_khi = (d_bcrec[0].hi(2) == BCType::ext_dir) or
                                                 (d_bcrec[0].hi(2) == BCType::hoextrap););

            // Initialize to zero just in case
            fx(i,j,k,0) = 0.0;
            fx(i,j,k,1) = 0.0;

            if (flag(i,j,k).isConnected(-1,0,0))
            {
               Real yf = fcx(i,j,k,0); // local (y,z) of centroid of x-face we are extrapolating to
#if (AMREX_SPACEDIM == 3)
               Real zf = fcx(i,j,k,1);
#endif
               AMREX_D_TERM(Real delta_x = 0.5 + ccc(i,j,k,0);,
                            Real delta_y = yf  - ccc(i,j,k,1);,
                            Real delta_z = zf  - ccc(i,j,k,2););

               Real u_mns = vel(i-1,j,k,0);
               Real u_pls = vel(i  ,j,k,0);
               Real v_mns = vel(i-1,j,k,0);
               Real v_pls = vel(i  ,j,k,0);

               Real cc_umax = amrex::max(u_pls, u_mns);
               Real cc_umin = amrex::min(u_pls, u_mns);
               Real cc_vmax = amrex::max(v_pls, v_mns);
               Real cc_vmin = amrex::min(v_pls, v_mns);

               // Compute slopes of component "0" of vel
               const auto& slopes_eb_hi_u = amrex_calc_slopes_extdir_eb(i,j,k,0,vel,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

               // Compute slopes of component "1" of vel
               const auto& slopes_eb_hi_v = amrex_calc_slopes_extdir_eb(i,j,k,1,vel,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

#if (AMREX_SPACEDIM == 3)
               Real upls = vel(i,j,k,0) - delta_x * slopes_eb_hi_u[0]
                                        + delta_y * slopes_eb_hi_u[1]
                                        + delta_z * slopes_eb_hi_u[2];
               Real vpls = vel(i,j,k,0) - delta_x * slopes_eb_hi_v[0]
                                        + delta_y * slopes_eb_hi_v[1]
                                        + delta_z * slopes_eb_hi_v[2];
#else
               Real upls = vel(i,j,k,0) - delta_x * slopes_eb_hi_u[0]
                                        + delta_y * slopes_eb_hi_u[1];
               Real vpls = vel(i,j,k,0) - delta_x * slopes_eb_hi_v[0]
                                        + delta_y * slopes_eb_hi_v[1];
#endif
               upls = amrex::max(amrex::min(upls, cc_umax), cc_umin);
               vpls = amrex::max(amrex::min(vpls, cc_vmax), cc_vmin);

               AMREX_D_TERM(delta_x = 0.5 - ccc(i-1,j,k,0);,
                            delta_y = yf  - ccc(i-1,j,k,1);,
                            delta_z = zf  - ccc(i-1,j,k,2););

               // Compute slopes of component "0" of vel
               const auto& slopes_eb_lo_u = amrex_calc_slopes_extdir_eb(i-1,j,k,0,vel,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

               // Compute slopes of component "1" of vel
               const auto& slopes_eb_lo_v = amrex_calc_slopes_extdir_eb(i-1,j,k,1,vel,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

#if (AMREX_SPACEDIM == 3)
               Real umns = vel(i-1,j,k,0) + delta_x * slopes_eb_lo_u[0]
                                          + delta_y * slopes_eb_lo_u[1]
                                          + delta_z * slopes_eb_lo_u[2];

               Real vmns = vel(i-1,j,k,1) + delta_x * slopes_eb_lo_v[0]
                                          + delta_y * slopes_eb_lo_v[1]
                                          + delta_z * slopes_eb_lo_v[2];
#else
               Real umns = vel(i-1,j,k,0) + delta_x * slopes_eb_lo_u[0]
                                          + delta_y * slopes_eb_lo_u[1];
               Real vmns = vel(i-1,j,k,1) + delta_x * slopes_eb_lo_v[0]
                                          + delta_y * slopes_eb_lo_v[1];
#endif
               umns = amrex::max(amrex::min(umns, cc_umax), cc_umin);
               vmns = amrex::max(amrex::min(vmns, cc_vmax), cc_vmin);

               Real u_val(0);
               Real v_val(0.5*(vpls+vmns));

               if ( umns >= 0.0 or upls <= 0.0 ) 
               {
                  Real avg = 0.5 * ( upls + umns );

                  if (avg >= small_vel) {
                    u_val = umns;
                    v_val = vmns;
                  }
                  else if (avg <= -small_vel) {
                    u_val = upls;
                    v_val = vpls;
                  }
               }

               if (i == domain_ilo && (d_bcrec[0].lo(0) == BCType::ext_dir)) {
                   u_val = u_mns;
                   v_val = v_mns;
               } else if (i == domain_ihi+1 && (d_bcrec[0].hi(0) == BCType::ext_dir)) {
                   u_val = u_pls;
                   v_val = v_pls;
               }
            
               fx(i,j,k,0) = u_val*u_val;
               fx(i,j,k,1) = v_val*u_val;
            }
        });
    }
    else
    {
        amrex::ParallelFor(Box(ubx),
        [fx,vel,flag,fcx,ccc]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Initialize to zero just in case
            fx(i,j,k,0) = 0.0;
            fx(i,j,k,1) = 0.0;

            if (flag(i,j,k).isConnected(-1,0,0))
            {
               Real yf = fcx(i,j,k,0); // local (y,z) of centroid of x-face we are extrapolating to
#if (AMREX_SPACEDIM == 3)
               Real zf = fcx(i,j,k,1);
#endif

               AMREX_D_TERM(Real delta_x = 0.5 + ccc(i,j,k,0);,
                            Real delta_y = yf  - ccc(i,j,k,1);,
                            Real delta_z = zf  - ccc(i,j,k,2););

               const Real u_mns = vel(i-1,j,k,0);
               const Real u_pls = vel(i  ,j,k,0);

               const Real v_mns = vel(i-1,j,k,1);
               const Real v_pls = vel(i  ,j,k,1);

               Real cc_umax = amrex::max(u_pls, u_mns);
               Real cc_umin = amrex::min(u_pls, u_mns);
               Real cc_vmax = amrex::max(v_pls, v_mns);
               Real cc_vmin = amrex::min(v_pls, v_mns);

               // Compute slopes of both components of vel
               const auto slopes_eb_hi_u = amrex_calc_slopes_eb(i,j,k,0,vel,ccc,flag);
               const auto slopes_eb_hi_v = amrex_calc_slopes_eb(i,j,k,1,vel,ccc,flag);

#if (AMREX_SPACEDIM == 3)
               Real upls = u_pls - delta_x * slopes_eb_hi_u[0]
                                 + delta_y * slopes_eb_hi_u[1]
                                 + delta_z * slopes_eb_hi_u[2];
               Real vpls = v_pls - delta_x * slopes_eb_hi_v[0]
                                 + delta_y * slopes_eb_hi_v[1]
                                 + delta_z * slopes_eb_hi_v[2];
#else
               Real upls = u_pls - delta_x * slopes_eb_hi_u[0]
                                 + delta_y * slopes_eb_hi_u[1];
               Real vpls = v_pls - delta_x * slopes_eb_hi_v[0]
                                 + delta_y * slopes_eb_hi_v[1];
#endif
               upls = amrex::max(amrex::min(upls, cc_umax), cc_umin);

               AMREX_D_TERM(delta_x = 0.5 - ccc(i-1,j,k,0);,
                            delta_y = yf  - ccc(i-1,j,k,1);,
                            delta_z = zf  - ccc(i-1,j,k,2););

               // Compute slopes of both components of vel
               const auto& slopes_eb_lo_u = amrex_calc_slopes_eb(i-1,j,k,0,vel,ccc,flag);
               const auto& slopes_eb_lo_v = amrex_calc_slopes_eb(i-1,j,k,1,vel,ccc,flag);

#if (AMREX_SPACEDIM == 3)
               Real umns = u_mns + delta_x * slopes_eb_lo_u[0]
                                 + delta_y * slopes_eb_lo_u[1]
                                 + delta_z * slopes_eb_lo_u[2];
               Real vmns = v_mns + delta_x * slopes_eb_lo_v[0]
                                 + delta_y * slopes_eb_lo_v[1]
                                 + delta_z * slopes_eb_lo_v[2];
#else
               Real umns = u_mns + delta_x * slopes_eb_lo_u[0]
                                 + delta_y * slopes_eb_lo_u[1];
               Real vmns = v_mns + delta_x * slopes_eb_lo_v[0]
                                 + delta_y * slopes_eb_lo_v[1];
#endif
               umns = amrex::max(amrex::min(umns, cc_umax), cc_umin);
               vmns = amrex::max(amrex::min(vmns, cc_vmax), cc_vmin);

               Real u_val(0);
               Real v_val(0.5*(vpls+vmns));

               if ( umns >= 0.0 or upls <= 0.0 ) 
               {
                  Real avg = 0.5 * ( upls + umns );

                  if (avg >= small_vel) {
                    u_val = umns;
                    v_val = vmns;
                  }
                  else if (avg <= -small_vel) {
                    u_val = upls;
                    v_val = vpls;
                  }
               }

               fx(i,j,k,0) = u_val*u_val;
               fx(i,j,k,1) = v_val*u_val;
            }
        });
    }

    // ****************************************************************************
    // Predict to y-faces
    // ****************************************************************************
    extdir_lohi = has_extdir_or_ho(h_bcrec.data(), ncomp, static_cast<int>(Direction::y));
    has_extdir_or_ho_lo = extdir_lohi.first;
    has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_jlo >= vbx.smallEnd(1)-1) or
        (has_extdir_or_ho_hi and domain_jhi <= vbx.bigEnd(1)))
    {
        amrex::ParallelFor(Box(vbx),
        [fy,vel,flag,ccc,d_bcrec,
         AMREX_D_DECL(fcx,fcy,fcz),
         AMREX_D_DECL(domain_ilo,domain_jlo,domain_klo),
         AMREX_D_DECL(domain_ihi,domain_jhi,domain_khi)]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            AMREX_D_TERM(bool extdir_or_ho_ilo = (d_bcrec[1].lo(0) == BCType::ext_dir) or
                                                 (d_bcrec[1].lo(0) == BCType::hoextrap);,
                         bool extdir_or_ho_jlo = (d_bcrec[1].lo(1) == BCType::ext_dir) or
                                                 (d_bcrec[1].lo(1) == BCType::hoextrap);,
                         bool extdir_or_ho_klo = (d_bcrec[1].lo(2) == BCType::ext_dir) or
                                                 (d_bcrec[1].lo(2) == BCType::hoextrap););

            AMREX_D_TERM(bool extdir_or_ho_ihi = (d_bcrec[1].hi(0) == BCType::ext_dir) or
                                                 (d_bcrec[1].hi(0) == BCType::hoextrap);,
                         bool extdir_or_ho_jhi = (d_bcrec[1].hi(1) == BCType::ext_dir) or
                                                 (d_bcrec[1].hi(1) == BCType::hoextrap);,
                         bool extdir_or_ho_khi = (d_bcrec[1].hi(2) == BCType::ext_dir) or
                                                 (d_bcrec[1].hi(2) == BCType::hoextrap););

            // Initialize to zero just in case
            fy(i,j,k,0) = 0.0;
            fy(i,j,k,1) = 0.0;

            if (flag(i,j,k).isConnected(0,-1,0))
            {
               Real xf = fcy(i,j,k,0); // local (x,z) of centroid of y-face we are extrapolating to
#if (AMREX_SPACEDIM == 3)
               Real zf = fcy(i,j,k,1);
#endif

               AMREX_D_TERM(Real delta_x = xf  - ccc(i,j,k,0);,
                            Real delta_y = 0.5 + ccc(i,j,k,1);,
                            Real delta_z = zf  - ccc(i,j,k,2););

               const Real u_mns = vel(i,j-1,k,0);
               const Real u_pls = vel(i,j  ,k,0);
               const Real v_mns = vel(i,j-1,k,1);
               const Real v_pls = vel(i,j  ,k,1);

               Real cc_umax = amrex::max(u_pls, u_mns);
               Real cc_umin = amrex::min(u_pls, u_mns);
               Real cc_vmax = amrex::max(v_pls, v_mns);
               Real cc_vmin = amrex::min(v_pls, v_mns);

               // Compute slopes of both components of vel
               const auto& slopes_eb_hi_u = amrex_calc_slopes_extdir_eb(i,j,k,0,vel,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));
               const auto& slopes_eb_hi_v = amrex_calc_slopes_extdir_eb(i,j,k,1,vel,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

#if (AMREX_SPACEDIM == 3)
               Real upls = u_pls + delta_x * slopes_eb_hi_u[0]
                                 - delta_y * slopes_eb_hi_u[1]
                                 + delta_z * slopes_eb_hi_u[2];
               Real vpls = v_pls + delta_x * slopes_eb_hi_v[0]
                                 - delta_y * slopes_eb_hi_v[1]
                                 + delta_z * slopes_eb_hi_v[2];
#else
               Real upls = u_pls + delta_x * slopes_eb_hi_u[0]
                                 - delta_y * slopes_eb_hi_u[1];
               Real vpls = v_pls + delta_x * slopes_eb_hi_v[0]
                                 - delta_y * slopes_eb_hi_v[1];
#endif

               upls = amrex::max(amrex::min(upls, cc_umax), cc_umin);
               vpls = amrex::max(amrex::min(vpls, cc_vmax), cc_vmin);

               AMREX_D_TERM(delta_x = xf  - ccc(i,j-1,k,0);,
                            delta_y = 0.5 - ccc(i,j-1,k,1);,
                            delta_z = zf  - ccc(i,j-1,k,2););

               // Compute slopes of both components of vel
               const auto& slopes_eb_lo_u = amrex_calc_slopes_extdir_eb(i,j-1,k,0,vel,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));
               // Compute slopes of both components of vel
               const auto& slopes_eb_lo_v = amrex_calc_slopes_extdir_eb(i,j-1,k,1,vel,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

#if (AMREX_SPACEDIM == 3)
               Real umns = umns + delta_x * slopes_eb_lo_u[0]
                                + delta_y * slopes_eb_lo_u[1]
                                + delta_z * slopes_eb_lo_u[2];
               Real vmns = vmns + delta_x * slopes_eb_lo_v[0]
                                + delta_y * slopes_eb_lo_v[1]
                                + delta_z * slopes_eb_lo_v[2];
#else
               Real umns = umns + delta_x * slopes_eb_lo_u[0]
                                + delta_y * slopes_eb_lo_u[1];
               Real vmns = vmns + delta_x * slopes_eb_lo_v[0]
                                + delta_y * slopes_eb_lo_v[1];
#endif

               umns = amrex::max(amrex::min(umns, cc_umax), cc_umin);
               vmns = amrex::max(amrex::min(vmns, cc_vmax), cc_vmin);

               Real v_val(0);
               Real u_val(0.5*(upls+umns));

               if ( vmns >= 0.0 or vpls <= 0.0 ) {
                  Real avg = 0.5 * ( vpls + vmns );

                  if (avg >= small_vel) {
                    u_val = umns;
                    v_val = vmns;
                  }
                  else if (avg <= -small_vel) {
                    u_val = upls;
                    v_val = vpls;
                  }
               }

               if (j == domain_jlo && (d_bcrec[1].lo(1) == BCType::ext_dir)) {
                   u_val = u_mns;
                   v_val = v_mns;
               } 
               else if (j == domain_jhi+1 && (d_bcrec[1].hi(1) == BCType::ext_dir)) {
                   u_val = u_pls;
                   v_val = v_pls;
               }

               fy(i,j,k,0) = u_val*v_val;
               fy(i,j,k,1) = v_val*v_val;
            }
        });
    }
    else
    {
        amrex::ParallelFor(Box(vbx),
        [fy,vel,flag,fcy,ccc] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Initialize to zero just in case
            fy(i,j,k,0) = 0.0;
            fy(i,j,k,1) = 0.0;

            if (flag(i,j,k).isConnected(0,-1,0))
            {
               Real xf = fcy(i,j,k,0); // local (x,z) of centroid of y-face we are extrapolating to
#if (AMREX_SPACEDIM == 3)
               Real zf = fcy(i,j,k,1);
#endif

               AMREX_D_TERM(Real delta_x = xf  - ccc(i,j,k,0);,
                            Real delta_y = 0.5 + ccc(i,j,k,1);,
                            Real delta_z = zf  - ccc(i,j,k,2););

               const Real u_mns = vel(i,j-1,k,0);
               const Real u_pls = vel(i,j  ,k,0);
               const Real v_mns = vel(i,j-1,k,1);
               const Real v_pls = vel(i,j  ,k,1);

               Real cc_umax = amrex::max(u_pls, u_mns);
               Real cc_umin = amrex::min(u_pls, u_mns);
               Real cc_vmax = amrex::max(v_pls, v_mns);
               Real cc_vmin = amrex::min(v_pls, v_mns);

               // Compute slopes of both components of vel
               const auto slopes_eb_hi_u = amrex_calc_slopes_eb(i,j,k,0,vel,ccc,flag);
               const auto slopes_eb_hi_v = amrex_calc_slopes_eb(i,j,k,1,vel,ccc,flag);

#if (AMREX_SPACEDIM == 3)
               Real upls = u_pls + delta_x * slopes_eb_hi_u[0]
                                 - delta_y * slopes_eb_hi_u[1]
                                 + delta_z * slopes_eb_hi_u[2];
               Real vpls = v_pls + delta_x * slopes_eb_hi_v[0]
                                 - delta_y * slopes_eb_hi_v[1]
                                 + delta_z * slopes_eb_hi_v[2];
#else
               Real upls = u_pls + delta_x * slopes_eb_hi_u[0]
                                 - delta_y * slopes_eb_hi_u[1];
               Real vpls = v_pls + delta_x * slopes_eb_hi_v[0]
                                 - delta_y * slopes_eb_hi_v[1];
#endif

               upls = amrex::max(amrex::min(upls, cc_umax), cc_umin);
               vpls = amrex::max(amrex::min(vpls, cc_vmax), cc_vmin);

               AMREX_D_TERM(delta_x = xf  - ccc(i,j-1,k,0);,
                            delta_y = 0.5 - ccc(i,j-1,k,1);,
                            delta_z = zf  - ccc(i,j-1,k,2););

               // Compute slopes of both components of vel
               const auto& slopes_eb_lo_u = amrex_calc_slopes_eb(i,j-1,k,0,vel,ccc,flag);
               const auto& slopes_eb_lo_v = amrex_calc_slopes_eb(i,j-1,k,1,vel,ccc,flag);

#if (AMREX_SPACEDIM == 3)
               Real umns = u_mns + delta_x * slopes_eb_lo_u[0]
                                 + delta_y * slopes_eb_lo_u[1]
                                 + delta_z * slopes_eb_lo_u[2];
               Real vmns = v_mns + delta_x * slopes_eb_lo_v[0]
                                 + delta_y * slopes_eb_lo_v[1]
                                 + delta_z * slopes_eb_lo_v[2];
#else
               Real umns = u_mns + delta_x * slopes_eb_lo_u[0]
                                 + delta_y * slopes_eb_lo_u[1];
               Real vmns = v_mns + delta_x * slopes_eb_lo_v[0]
                                 + delta_y * slopes_eb_lo_v[1];
#endif
                                          
               umns = amrex::max(amrex::min(umns, cc_umax), cc_umin);
               vmns = amrex::max(amrex::min(vmns, cc_vmax), cc_vmin);

               Real v_val(0);
               Real u_val(0.5*(upls+umns));

               if ( vmns >= 0.0 or vpls <= 0.0 ) 
               {
                  Real avg = 0.5 * ( vpls + vmns );

                  if (avg >= small_vel) {
                    u_val = umns;
                    v_val = vmns;
                  }
                  else if (avg <= -small_vel) {
                    u_val = upls;
                    v_val = vpls;
                  }
               }

               fy(i,j,k,0) = u_val*v_val;
               fy(i,j,k,1) = v_val*v_val;
            }
        });
    }

#if (AMREX_SPACEDIM == 3)
    // ****************************************************************************
    // Predict to z-faces
    // ****************************************************************************
    extdir_lohi = has_extdir_or_ho(h_bcrec.data(), ncomp, static_cast<int>(Direction::z));
    has_extdir_or_ho_lo = extdir_lohi.first;
    has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_klo >= wbx.smallEnd(2)-1) or
        (has_extdir_or_ho_hi and domain_khi <= wbx.bigEnd(2)))
    {
        amrex::ParallelFor(Box(wbx),
        [w,vel,flag,ccc,d_bcrec,
         AMREX_D_DECL(fcx,fcy,fcz),
         domain_ilo,domain_ihi,domain_jlo,domain_jhi,domain_klo,domain_khi]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real w_val(0);

            bool extdir_or_ho_ilo = (d_bcrec[2].lo(0) == BCType::ext_dir) or
                                    (d_bcrec[2].lo(0) == BCType::hoextrap);
            bool extdir_or_ho_ihi = (d_bcrec[2].hi(0) == BCType::ext_dir) or
                                    (d_bcrec[2].hi(0) == BCType::hoextrap);

            bool extdir_or_ho_jlo = (d_bcrec[2].lo(1) == BCType::ext_dir) or
                                    (d_bcrec[2].lo(1) == BCType::hoextrap);
            bool extdir_or_ho_jhi = (d_bcrec[2].hi(1) == BCType::ext_dir) or
                                    (d_bcrec[2].hi(1) == BCType::hoextrap);

            bool extdir_or_ho_klo = (d_bcrec[2].lo(2) == BCType::ext_dir) or
                                    (d_bcrec[2].lo(2) == BCType::hoextrap);
            bool extdir_or_ho_khi = (d_bcrec[2].hi(2) == BCType::ext_dir) or
                                    (d_bcrec[2].hi(2) == BCType::hoextrap);

            if (flag(i,j,k).isConnected(0,0,-1))
            {
               Real xf = fcz(i,j,k,0); // local (x,y) of centroid of z-face we are extrapolating to
               Real yf = fcz(i,j,k,1);

               Real delta_x = xf  - ccc(i,j,k,0);
               Real delta_y = yf  - ccc(i,j,k,1);
               Real delta_z = 0.5 + ccc(i,j,k,2);

               const Real vel_mns = vel(i,j,k-1,2);
               const Real vel_pls = vel(i,j,k,2);

               Real cc_wmax = amrex::max(vel_pls, vel_mns);
               Real cc_wmin = amrex::min(vel_pls, vel_mns);

               // Compute slopes of component "2" of vel
               const auto& slopes_eb_hi = amrex_calc_slopes_extdir_eb(i,j,k,2,vel,ccc,
                                          AMREX_D_DECL(fcx,fcy,fcz), flag,
                                          AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                          AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                          AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                          AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

               Real wpls = vel_pls + delta_x * slopes_eb_hi[0]
                                   + delta_y * slopes_eb_hi[1]
                                   - delta_z * slopes_eb_hi[2];

               wpls = amrex::max(amrex::min(wpls, cc_wmax), cc_wmin);

               delta_x = xf  - ccc(i,j,k-1,0);
               delta_y = yf  - ccc(i,j,k-1,1);
               delta_z = 0.5 - ccc(i,j,k-1,2);

               // Compute slopes of component "2" of vel
               const auto& slopes_eb_lo = amrex_calc_slopes_extdir_eb(i,j,k-1,2,vel,ccc,
                                          AMREX_D_DECL(fcx,fcy,fcz), flag,
                                          AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                          AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                          AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                          AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

               Real wmns = vel_mns + delta_x * slopes_eb_lo[0]
                                   + delta_y * slopes_eb_lo[1]
                                   + delta_z * slopes_eb_lo[2];

               wmns = amrex::max(amrex::min(wmns, cc_wmax), cc_wmin);

               if ( wmns >= 0.0 or wpls <= 0.0 ) {
                  Real avg = 0.5 * ( wpls + wmns );

                  if (avg >= small_vel) {
                    w_val = wmns;
                  }
                  else if (avg <= -small_vel) {
                    w_val = wpls;
                  }
               }

                if (k == domain_klo && (d_bcrec[2].lo(2) == BCType::ext_dir)) {
                    w_val = vel_mns;
                }
                else if (k == domain_khi+1 && (d_bcrec[2].hi(2) == BCType::ext_dir)) {
                    w_val = vel_pls;
                }
            }

            w(i,j,k) = w_val;
        });
    }
    else
    {
        amrex::ParallelFor(Box(wbx),
        [w,vel,flag,fcz,ccc] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real w_val(0);

            if (flag(i,j,k).isConnected(0,0,-1))
            {
               Real xf = fcz(i,j,k,0); // local (x,y) of centroid of z-face we are extrapolating to
               Real yf = fcz(i,j,k,1);

               Real delta_x = xf  - ccc(i,j,k,0);
               Real delta_y = yf  - ccc(i,j,k,1);
               Real delta_z = 0.5 + ccc(i,j,k,2);

               const Real vel_mns = vel(i,j,k-1,2);
               const Real vel_pls = vel(i,j,k,2);

               Real cc_wmax = amrex::max(vel_pls, vel_mns);
               Real cc_wmin = amrex::min(vel_pls, vel_mns);

               // Compute slopes of component "2" of vel
               const auto slopes_eb_hi = amrex_calc_slopes_eb(i,j,k,2,vel,ccc,flag);

               Real wpls = vel_pls + delta_x * slopes_eb_hi[0]
                                   + delta_y * slopes_eb_hi[1]
                                   - delta_z * slopes_eb_hi[2];

               wpls = amrex::max(amrex::min(wpls, cc_wmax), cc_wmin);

               delta_x = xf  - ccc(i,j,k-1,0);
               delta_y = yf  - ccc(i,j,k-1,1);
               delta_z = 0.5 - ccc(i,j,k-1,2);

               // Compute slopes of component "2" of vel
               const auto& slopes_eb_lo = amrex_calc_slopes_eb(i,j,k-1,2,vel,ccc,flag);

               Real wmns = vel_mns + delta_x * slopes_eb_lo[0]
                                   + delta_y * slopes_eb_lo[1]
                                   + delta_z * slopes_eb_lo[2];

               wmns = amrex::max(amrex::min(wmns, cc_wmax), cc_wmin);

               if ( wmns >= 0.0 or wpls <= 0.0 ) {
                  Real avg = 0.5 * ( wpls + wmns );

                  if (avg >= small_vel) {
                    w_val = wmns;
                  }
                  else if (avg <= -small_vel) {
                    w_val = wpls;
                  }
               }
            }

            w(i,j,k) = w_val;
        });
    }
#endif
}
#endif
