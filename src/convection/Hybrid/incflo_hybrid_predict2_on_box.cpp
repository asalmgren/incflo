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
hybrid::predict_vels_with_forces ( Box const& bx,
                                   AMREX_D_DECL(Box const& ubx, 
                                                Box const& vbx, 
                                                Box const& wbx),
                                   AMREX_D_DECL(Array4<Real> const& u, 
                                                Array4<Real> const& v,
                                                Array4<Real> const& w), 
                                   Array4<Real const> const& vel,
                                   Array4<Real const> const& vel_forces,
                                   Array4<Real const> const& dudt,
                                   Vector<BCRec> const& h_bcrec,
                                          BCRec  const* d_bcrec,
                                   Real dt, Geometry& geom )
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
        amrex::ParallelFor(ubx, [vel,u,vel_forces,dudt,domain_ilo,domain_ihi,d_bcrec,dt]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            bool extdir_or_ho_ilo = (d_bcrec[0].lo(0) == BCType::ext_dir) or
                                    (d_bcrec[0].lo(0) == BCType::hoextrap);
            bool extdir_or_ho_ihi = (d_bcrec[0].hi(0) == BCType::ext_dir) or
                                    (d_bcrec[0].hi(0) == BCType::hoextrap);

            int order = 2;

            Real upls = vel(i  ,j,k,0) - 0.5 * amrex_calc_xslope_extdir(
                 i  ,j,k,0,order,vel,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi) 
                 + 0.5 * dt * (vel_forces(i,j,k,0) + dudt(i,j,k,0)) ;
            Real umns = vel(i-1,j,k,0) + 0.5 * amrex_calc_xslope_extdir(
                 i-1,j,k,0,order,vel,extdir_or_ho_ilo, extdir_or_ho_ihi, domain_ilo, domain_ihi) 
                 + 0.5 * dt * (vel_forces(i-1,j,k,0) + dudt(i-1,j,k,0)) ;

            Real u_val = 0.;

            if (umns >= 0.0 or upls <= 0.0) {
                
                Real avg = 0.5 * (upls + umns);

                if (avg >= small_vel) {
                    u_val = umns;
                }
                else if (avg <= -small_vel){
                    u_val = upls;
                }
            }

            if (i == domain_ilo && (d_bcrec[0].lo(0) == BCType::ext_dir)) {
                u_val = vel(i-1,j,k,0);
            } else if (i == domain_ihi+1 && (d_bcrec[0].hi(0) == BCType::ext_dir)) {
                u_val = vel(i  ,j,k,0);
            }

            u(i,j,k) = u_val;
        });
    }
    else
    {
        amrex::ParallelFor(ubx, [vel,u,vel_forces,dudt,dt]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            Real upls = vel(i  ,j,k,0) - 0.5 * amrex_calc_xslope(i  ,j,k,0,order,vel) 
                                       + 0.5 * dt * (vel_forces(i,j,k,0) + dudt(i,j,k,0));
            Real umns = vel(i-1,j,k,0) + 0.5 * amrex_calc_xslope(i-1,j,k,0,order,vel) 
                                       + 0.5 * dt * (vel_forces(i-1,j,k,0) + dudt(i-1,j,k,0));

            Real u_val(0);

            if (umns >= 0.0 or upls <= 0.0) {
                
                Real avg = 0.5 * (upls + umns);
                
                if (avg >= small_vel) {
                    u_val = umns;
                }
                else if (avg <= -small_vel){
                    u_val = upls;
                }
            }

            u(i,j,k) = u_val;
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
        amrex::ParallelFor(vbx, [vel,v,vel_forces,dudt,domain_jlo,domain_jhi,d_bcrec,dt]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            bool extdir_or_ho_jlo = (d_bcrec[1].lo(1) == BCType::ext_dir) or
                                    (d_bcrec[1].lo(1) == BCType::hoextrap);
            bool extdir_or_ho_jhi = (d_bcrec[1].hi(1) == BCType::ext_dir) or
                                    (d_bcrec[1].hi(1) == BCType::hoextrap);

            int order = 2;
    
            Real vpls = vel(i,j  ,k,1) - 0.5 * amrex_calc_yslope_extdir(
                 i,j,k,1,order,vel,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi) 
                                       + 0.5 * dt * (vel_forces(i,j,k,1) + dudt(i,j,k,1));
            Real vmns = vel(i,j-1,k,1) + 0.5 * amrex_calc_yslope_extdir(
                 i,j-1,k,1,order,vel,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi) 
                                       + 0.5 * dt * (vel_forces(i,j-1,k,1) + dudt(i,j-1,k,1));

            Real v_val(0);

            if (vmns >= 0.0 or vpls <= 0.0) {

                Real avg = 0.5 * (vpls + vmns);

                if (avg >= small_vel) {
                    v_val = vmns;
                }
                else if (avg <= -small_vel){
                    v_val = vpls;
                }
            }

            if (j == domain_jlo && (d_bcrec[1].lo(1) == BCType::ext_dir)) {
                v_val = vel(i,j-1,k,1);
            } else if (j == domain_jhi+1 && (d_bcrec[1].hi(1) == BCType::ext_dir)) {
                v_val = vel(i,j  ,k,1);
            }

            v(i,j,k) = v_val;
        });
    }
    else
    {
        amrex::ParallelFor(vbx, [vel,v,vel_forces,dudt,dt]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            Real vpls = vel(i,j  ,k,1) - 0.5 * amrex_calc_yslope(i,j  ,k,1,order,vel) 
                                       + 0.5 * dt * (vel_forces(i,j,k,1) + dudt(i,j,k,1));
            Real vmns = vel(i,j-1,k,1) + 0.5 * amrex_calc_yslope(i,j-1,k,1,order,vel) 
                                       + 0.5 * dt * (vel_forces(i,j-1,k,1) + dudt(i,j-1,k,1));

            Real v_val(0);

            if (vmns >= 0.0 or vpls <= 0.0) {

                Real avg = 0.5 * (vpls + vmns);

                if (avg >= small_vel) {
                    v_val = vmns;
                }
                else if (avg <= -small_vel){
                    v_val = vpls;
                }
            }

            v(i,j,k) = v_val;
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
        amrex::ParallelFor(wbx, [vel,vel_forces,dudt,domain_klo,domain_khi,w,d_bcrec,dt]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            bool extdir_or_ho_klo = (d_bcrec[2].lo(2) == BCType::ext_dir) or
                                    (d_bcrec[2].lo(2) == BCType::hoextrap);
            bool extdir_or_ho_khi = (d_bcrec[2].hi(2) == BCType::ext_dir) or
                                    (d_bcrec[2].hi(2) == BCType::hoextrap);

            const Real vel_pls = vel(i,j,k,2);
            const Real vel_mns = vel(i,j,k-1,2);

            int order = 2;

            Real wpls = vel_pls - 0.5 * amrex_calc_zslope_extdir(
                 i,j,k  ,2,order,vel,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi)
                                + 0.5 * dt * (vel_forces(i,j,k,2) + dudt(i,j,k,2)) ;
            Real wmns = vel_mns + 0.5 * amrex_calc_zslope_extdir(
                 i,j,k-1,2,order,vel,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi)
                                + 0.5 * dt * (vel_forces(i,j,k-1,2) + dudt(i,j,k-1,2)) ;

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
                w_val = vel_mns;
            } else if (k == domain_khi+1 && (d_bcrec[2].hi(2) == BCType::ext_dir)) {
                w_val = vel_pls;
            }

            w(i,j,k) = w_val;
        });
    }
    else
    {
        amrex::ParallelFor(wbx, [vel,w,vel_forces,dudt,dt]
        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int order = 2;

            Real wpls = vel(i,j,k  ,2) - 0.5 * amrex_calc_zslope(i,j,k  ,2,order,vel)
                                       + 0.5 * dt * (vel_forces(i,j,k,2) + dudt(i,j,k,2)) ;
            Real wmns = vel(i,j,k-1,2) + 0.5 * amrex_calc_zslope(i,j,k-1,2,order,vel)
                                       + 0.5 * dt * (vel_forces(i,j,k-1,2) + dudt(i,j,k-1,2)) ;
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
