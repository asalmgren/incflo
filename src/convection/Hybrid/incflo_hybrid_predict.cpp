#include <Hybrid.H>
#include <AMReX_Slopes_K.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void 
hybrid::predict_vels_on_faces ( AMREX_D_DECL(MultiFab& u_mac, 
                                             MultiFab& v_mac,
                                             MultiFab& w_mac), 
                               MultiFab const& vel,
                               MultiFab const& vel_forces,
                               Vector<BCRec> const& h_bcrec,
                                   BCRec  const* d_bcrec,
#ifdef AMREX_USE_EB
                               EBFArrayBoxFactory const* ebfact,
#endif
                               Real dt, Geometry& geom)
{
    Real l_dt = dt; 

#ifdef AMREX_USE_EB
    auto const& flags = ebfact->getMultiEBCellFlagFab();
    auto const& fcent = ebfact->getFaceCent();
    auto const& ccent = ebfact->getCentroid();
    auto const& vfrac = ebfact->getVolFrac();
#endif

    // Temporary to hold div(uu) computed in Step 1 to use as source term in Step 2
    MultiFab dudt(vel.boxArray(), vel.DistributionMap(), AMREX_SPACEDIM, 2);
    dudt.setVal(0.);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        // Turn off tiling due to overlap needed to make (dvdt)
        for (MFIter mfi(vel, false); mfi.isValid(); ++mfi)
        {
            AMREX_D_TERM(Box const& ubx = mfi.nodaltilebox(0);,
                         Box const& vbx = mfi.nodaltilebox(1);,
                         Box const& wbx = mfi.nodaltilebox(2););
            AMREX_D_TERM(Box const& ubx_grown = mfi.grownnodaltilebox(0,1);,
                         Box const& vbx_grown = mfi.grownnodaltilebox(1,1);,
                         Box const& wbx_grown = mfi.grownnodaltilebox(2,1););
            Array4<Real const> const& vel_arr  = vel.const_array(mfi);
            Array4<Real      > const& dudt_arr = dudt.array(mfi);

            Box const& bx = mfi.tilebox();
            Box gbx = bx;
            gbx.grow(1);

            Box tmpbox = amrex::surroundingNodes(gbx);

            FArrayBox tmpfab(tmpbox, AMREX_SPACEDIM*AMREX_SPACEDIM);
            tmpfab.setVal(0.);
            Elixir eli = tmpfab.elixir();

            AMREX_D_TERM(Array4<Real> const& fx = tmpfab.array(0);,
                         Array4<Real> const& fy = tmpfab.array(AMREX_SPACEDIM);,
                         Array4<Real> const& fz = tmpfab.array(2*AMREX_SPACEDIM););

#ifdef AMREX_USE_EB

           Array4<Real const> AMREX_D_DECL(apx, apy, apz);
           Array4<Real const> AMREX_D_DECL(fcx, fcy, fcz);

           EBCellFlagFab const& flagfab = flags[mfi];

           bool regular = (flagfab.getType(amrex::grow(bx,2)) == FabType::regular);
           if (!regular) {
               AMREX_D_TERM(apx = ebfact->getAreaFrac()[0]->const_array(mfi);,
                            apy = ebfact->getAreaFrac()[1]->const_array(mfi);,
                            apz = ebfact->getAreaFrac()[2]->const_array(mfi););
            }
            Array4<EBCellFlag const> const& flag_arr = flagfab.const_array();
            auto const typ = flagfab.getType(amrex::grow(bx,2));
            if (typ == FabType::covered)
            {
                // We have already zeroed out fx,fy,fz above
            }
            else if (typ == FabType::singlevalued)
            {
                AMREX_D_TERM(fcx = fcent[0]->const_array(mfi);,
                             fcy = fcent[1]->const_array(mfi);,
                             fcz = fcent[2]->const_array(mfi););

                Array4<Real const> ccent_arr = ccent.const_array(mfi);
                Array4<Real const> vfrac_arr = vfrac.const_array(mfi);
                
                // 
                // STEP 1:  Create fluxes on faces by predicting to faces and upwinding
                // 
                hybrid::predict_vels_on_faces_eb(AMREX_D_DECL(ubx_grown,vbx_grown,wbx_grown), 
                                                 AMREX_D_DECL(fx, fy, fz),
                                                 vel_arr,flag_arr,AMREX_D_DECL(fcx,fcy,fcz),ccent_arr,
                                                 h_bcrec,d_bcrec,geom);

                // 
                // STEP 2:  Compute div(uu) by taking the divergence of the fluxes from above
                // 
                hybrid::compute_convective_rate_eb(gbx, AMREX_SPACEDIM, dudt_arr, AMREX_D_DECL(fx, fy, fz),
                                                   flag_arr, vfrac_arr, AMREX_D_DECL(apx, apy, apz), geom);

                // 
                // STEP 3:  Predict to faces using u_t = -div(uu) + F 
                // 
                AMREX_D_TERM(Array4<Real> const& u = u_mac.array(mfi);,
                             Array4<Real> const& v = v_mac.array(mfi);,
                             Array4<Real> const& w = w_mac.array(mfi););
                Array4<Real const> const& vf_arr   = vel_forces.const_array(mfi);
                hybrid::predict_vels_with_forces_eb(bx,AMREX_D_DECL(ubx,vbx,wbx),
                                                    AMREX_D_DECL(u,v,w),vel_arr,vf_arr,dudt_arr,
                                                    flag_arr,AMREX_D_DECL(fcx,fcy,fcz),ccent_arr,
                                                    h_bcrec,d_bcrec,l_dt,geom);
            }
            else
#endif
            {
                // 
                // STEP 1:  Create fluxes on faces by predicting to faces and upwinding
                // 
                hybrid::predict_vels_on_faces(AMREX_D_DECL(ubx_grown,vbx_grown,wbx_grown),
                                              AMREX_D_DECL(fx,fy,fz),
                                              vel_arr,h_bcrec,d_bcrec,geom);
                // 
                // STEP 2:  Compute div(uu) by taking the divergence of the fluxes from above
                // 
                hybrid::compute_convective_rate(gbx, AMREX_SPACEDIM, dudt_arr, AMREX_D_DECL(fx, fy, fz), geom);
                // const auto dxinv = geom.InvCellSizeArray();
                // amrex_compute_divergence(bx, dudt_arr, AMREX_D_DECL(fx, fy, fz), dxinv);

                // 
                // STEP 3:  Predict to faces using u_t = -div(uu) + F 
                // 
                AMREX_D_TERM(Array4<Real> const& u = u_mac.array(mfi);,
                             Array4<Real> const& v = v_mac.array(mfi);,
                             Array4<Real> const& w = w_mac.array(mfi););
                Array4<Real const> const& vf_arr   = vel_forces.const_array(mfi);
                hybrid::predict_vels_with_forces(bx,AMREX_D_DECL(ubx,vbx,wbx),
                                                 AMREX_D_DECL(u,v,w),vel_arr,vf_arr,dudt_arr,
                                                 h_bcrec,d_bcrec,l_dt,geom);
            }
        } // MFIter
    }
}

