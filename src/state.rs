//! Contains definitions, and init + update code for our primary state.

use lin_alg::f64::Vec3;

use crate::{
    Axis,
    basis_wfs::Basis,
    dirac::BasisSpinor,
    GRID_MAX_CHARGE,
    GRID_MAX_GRADIENT,
    GRID_MAX_RENDER,
    GRID_N_CHARGE_DEFAULT,
    GRID_N_GRADIENT_DEFAULT, GRID_N_RENDER_DEFAULT, grid_setup::{self, Arr2dReal, Arr3d, Arr3dReal, new_data, new_data_2d_real, new_data_real}, presets::Preset, RENDER_L, RENDER_SPINOR,
    SPACING_FACTOR_DEFAULT, StateUi, SurfaceDesc, SurfaceToRender,
    types::{ComputationDevice, SurfacesPerElec, SurfacesShared}, ui::procedures, wf_ops::{self, DerivCalc, Q_PROT, Spin},
};
use crate::core_calcs::potential;

pub struct State {
    /// Computation device for evaluating the very expensive charge potential computation.
    pub dev_charge: ComputationDevice,
    /// Computation device for evaluating psi. (And psi''?)
    pub dev_psi: ComputationDevice,
    pub deriv_calc: DerivCalc,
    /// Eg, Nuclei (position, charge amt), per the Born-Oppenheimer approximation. Charges over space
    /// due to electrons are stored in `Surfaces`.
    pub nucleii: Vec<(Vec3, f64)>,
    /// The electric force acting on the nucleus, from electrons and nuclei
    /// todo: Combine with charges_fixed as a third Tuple term, or make it a struct.
    pub net_force_on_nuc: Vec<Vec3>,
    /// Charges from electrons, over 3d space. Each value is the charge created by a single electron. Computed from <ψ|ψ>.
    /// This is not part of `SurfacesPerElec` since we use all values at once (or all except one)
    /// when calculating the potential. (easier to work with API)
    pub charges_from_electron: Vec<Arr3dReal>,
    /// Also stored here vice part of per-elec structs due to borrow-limiting on struct fields.
    /// We use this to calculate charge.
    // pub V_from_elecs: Vec<Arr3dReal>,
    pub V_from_elecs: Vec<Arr2dReal>,
    /// Surfaces that are not electron-specific.
    pub surfaces_shared: SurfacesShared,
    /// Computed surfaces, per electron. These span 3D space, are are quite large in memory. Contains various
    /// data including the grid spacing, psi, psi'', V etc.
    /// Vec iterates over the different electrons.
    pub surfaces_per_elec: Vec<SurfacesPerElec>,
    /// Wave functions, with weights. Per-electron. (Outer Vec iterates over electrons; inner over
    /// bases per-electron)
    pub bases: Vec<Vec<Basis>>,
    pub bases_spinor: Vec<Vec<BasisSpinor>>,
    /// Similar to `bases_evaluated`, but on the charge grid. We don't need diffs for this.
    /// Outer is per-electron. Inner is per-basis
    pub psi_charge: Vec<Vec<Arr3d>>,
    // /// Amount to nudge next; stored based on sensitivity of previous nudge. Per-electron.
    // pub nudge_amount: Vec<f64>,
    pub surface_descs_per_elec: Vec<SurfaceDesc>,
    pub surface_descs_combined: Vec<SurfaceDesc>,
    pub grid_n_render: usize,
    /// This charge grid is generally denser than the main grid. This allows more fidelity for
    /// modelling electron charge, without evaluating the wave function at too many points.
    pub grid_n_charge: usize,
    /// For our gradient vector field display.
    pub grid_n_gradient: usize,
    pub grid_range_render: (f64, f64),
    pub grid_range_charge: (f64, f64),
    pub grid_range_gradient: (f64, f64),
    /// 1.0 is an evenly-spaced grid. A higher value spreads out the grid; high values
    /// mean increased non-linearity, with higher spacing farther from the center.
    /// This only (currently) applies to the main grid, with a uniform grid set for
    /// charge density.
    pub sample_factor_render: f64,
    // /// When finding and initializing basis, this is the maximum n quantum number.
    // pub max_basis_n: u16,
    pub num_elecs: usize,
    pub ui: StateUi,
    pub presets: Vec<Preset>,
    /// These are rendered.
    pub charge_density_balls: Vec<Vec3>,
}

impl State {
    pub fn new(
        // num_elecs: usize,
        dev_psi: ComputationDevice,
        dev_charge: ComputationDevice,
    ) -> Self {
        println!("Initializing state...");

        // todo: Adjustment to poreset
        let nuclei = Vec::new();
        let net_force_on_nuc = Vec::new();
        let bases_per_elec = Vec::new();
        let bases_per_elec_spinor = Vec::new();
        let charges_from_electron = Vec::new();
        let V_from_elecs = Vec::new();
        let psi_charge = Vec::new();
        let surfaces_per_elec = Vec::new();

        let num_elecs = 0;

        // todoFigure out why you get incorrect answers if these 2 grids don't line up.
        // todo: FOr now, you can continue with matching them if you wish.
        let grid_range_render = (-GRID_MAX_RENDER, GRID_MAX_RENDER);
        let grid_range_charge = (-GRID_MAX_CHARGE, GRID_MAX_CHARGE);
        let grid_range_gradient = (-GRID_MAX_GRADIENT, GRID_MAX_GRADIENT);

        // let spacing_factor = 1.6;
        // Currently, must be one as long as used with elec-elec charge.
        let spacing_factor = SPACING_FACTOR_DEFAULT;

        let grid_n_render = GRID_N_RENDER_DEFAULT;
        let grid_n_charge = GRID_N_CHARGE_DEFAULT;
        let grid_n_gradient = GRID_N_GRADIENT_DEFAULT;

        let psi_pp_calc = DerivCalc::Numeric;

        let surfaces_shared = SurfacesShared::new(
            grid_range_render,
            grid_range_charge,
            grid_range_gradient,
            spacing_factor,
            grid_n_render,
            grid_n_charge,
            grid_n_gradient,
            Axis::Z,
        );

        println!("Initializing from grid...");
        // let (charges_electron, V_from_elecs, psi_charge, surfaces_shared, surfaces_per_elec) =
        //     init_from_grid(
        //         &dev_psi,
        //         &dev_charge,
        //         (grid_min_render, grid_max_render),
        //         (grid_min_charge, grid_max_charge),
        //         spacing_factor,
        //         grid_n,
        //         grid_n_charge,
        //         &bases_per_elec,
        //         &bases_per_elec_spinor,
        //         &nuclei,
        //         num_elecs,
        //         psi_pp_calc,
        //         Axis::Z,
        //     );

        println!("Grid init complete.");

        let mut surface_descs_per_elec = vec![
            SurfaceDesc::new(SurfaceToRender::V, true),
            SurfaceDesc::new(SurfaceToRender::Psi, false),
            SurfaceDesc::new(SurfaceToRender::PsiIm, false),
            SurfaceDesc::new(SurfaceToRender::ChargeDensity, false),
            SurfaceDesc::new(SurfaceToRender::PsiPpCalc, false),
            SurfaceDesc::new(SurfaceToRender::PsiPpCalcIm, false),
            SurfaceDesc::new(SurfaceToRender::PsiPpMeas, false),
            SurfaceDesc::new(SurfaceToRender::PsiPpMeasIm, false),
            SurfaceDesc::new(SurfaceToRender::ElecVFromPsi, false),
            SurfaceDesc::new(SurfaceToRender::TotalVFromPsi, true),
            SurfaceDesc::new(SurfaceToRender::VDiff, false), // todo: Experiment
            // SurfaceDesc::new(SurfaceToRender::VPElec, false),
            SurfaceDesc::new(SurfaceToRender::H, false),
            SurfaceDesc::new(SurfaceToRender::HIm, false),
            // SurfaceDesc::new(SurfaceToRender::ElecFieldGradient, false),
        ];

        if RENDER_L {
            surface_descs_per_elec.append(&mut vec![
                SurfaceDesc::new(SurfaceToRender::LSq, false),
                SurfaceDesc::new(SurfaceToRender::LSqIm, false),
                SurfaceDesc::new(SurfaceToRender::LZ, false),
                SurfaceDesc::new(SurfaceToRender::LZIm, false),
                // todo: These likely temp to verify.
                // SurfaceDesc::new("dx", false),
                // SurfaceDesc::new("dy", false),
                // SurfaceDesc::new("dz", false),
                // SurfaceDesc::new("d2x", false),
                // SurfaceDesc::new("d2y", false),
                // SurfaceDesc::new("d2z", false),
            ])
        }

        if RENDER_SPINOR {
            surface_descs_per_elec.append(&mut vec![
                SurfaceDesc::new(SurfaceToRender::PsiSpinor0, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinor1, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinor2, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinor3, false),
                // Calculated, to compare to the trial.
                SurfaceDesc::new(SurfaceToRender::PsiSpinorCalc0, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinorCalc0, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinorCalc0, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinorCalc0, false),
            ])
        }

        // todo: Come back to this, and add appropriate SurfaceToRender variants next time you use this.
        let surface_descs_combined = vec![
            //     SurfaceDesc::new("V", true),
            //     SurfaceDesc::new("ψ_α", false),
            //     SurfaceDesc::new("ψ_β", false),
            //     SurfaceDesc::new("ψ_α im", false),
            //     SurfaceDesc::new("ψ_β im", false),
            //     SurfaceDesc::new("ρ_α", false),
            //     SurfaceDesc::new("ρ_β", false),
            //     SurfaceDesc::new("ρ", true),
            //     SurfaceDesc::new("ρ spin", true),
        ];

        println!("State init complete.");

        // todo: Sort out how and where to handle presets
        let presets = vec![
            Preset::make_h(),
            Preset::make_h_anion(),
            Preset::make_h2(),
            Preset::make_h2_cation(),
            Preset::make_he(),
            Preset::make_li(),
            Preset::make_li_test(),
            Preset::make_li_h(),
        ];

        let mut result = Self {
            dev_charge,
            dev_psi,
            deriv_calc: psi_pp_calc,
            nucleii: nuclei,
            net_force_on_nuc,
            charges_from_electron,
            V_from_elecs,
            bases: bases_per_elec,
            bases_spinor: bases_per_elec_spinor,
            psi_charge,
            surfaces_shared,
            surfaces_per_elec,
            surface_descs_per_elec,
            surface_descs_combined,
            grid_n_render,
            grid_n_charge,
            grid_n_gradient,
            grid_range_render,
            grid_range_charge,
            grid_range_gradient,
            sample_factor_render: spacing_factor,
            // max_basis_n,
            num_elecs,
            ui: Default::default(),
            presets,
            charge_density_balls: Vec::new(),
        };

        // todo: Maybe this is required to set up bases, to set up the first per-electron surface?
        // todo: Our UI code expects there to be at least one per-electron surface.
        result.set_preset(0);
        result.init_from_grid();

        result
    }

    /// Run this whenever n changes. Ie, at init, or when n changes in the GUI.
    /// (todo: Other caess?0-
    pub fn init_from_grid(&mut self) {
        let grid_n = self.grid_n_render; // Code shortener; we use this frequently.

        let arr_real_2d = new_data_2d_real(grid_n);

        // let sfcs_one_elec =
        // SurfacesPerElec::new(self.bases[0].len(), grid_n, self.grid_n_charge, Spin::Alpha);
        // SurfacesPerElec::new(0, grid_n, self.grid_n_charge, Spin::Alpha);

        // todo: Reconcile how elecs are managed. Per-nuc? Not?
        self.surfaces_per_elec = Vec::new();
        for i in 0..self.num_elecs {
            self.surfaces_per_elec.push(SurfacesPerElec::new(
                self.bases[i].len(),
                grid_n,
                self.grid_n_charge,
                Spin::Alpha,
            ));
        }

        self.surfaces_shared = SurfacesShared::new(
            self.grid_range_render,
            self.grid_range_charge,
            self.grid_range_gradient,
            self.sample_factor_render,
            grid_n,
            self.grid_n_charge,
            self.grid_n_gradient,
            self.ui.hidden_axis,
        );

        grid_setup::update_grid_posits_2d(
            &mut self.surfaces_shared.grid_posits,
            self.grid_range_render,
            self.sample_factor_render,
            // todo: Don't reset to 0.
            0., // z_displayed: Initialize.
            grid_n,
            self.ui.hidden_axis,
        );

        grid_setup::update_grid_posits(
            &mut self.surfaces_shared.grid_posits_charge,
            self.grid_range_charge,
            1.,
            self.grid_n_charge,
        );

        potential::update_V_from_nuclei(
            &mut self.surfaces_shared.V_from_nuclei,
            &self.nucleii,
            &self.surfaces_shared.grid_posits,
        );

        // These must be initialized from wave functions later.
        self.psi_charge = Vec::new();
        self.charges_from_electron = Vec::new();
        self.V_from_elecs = Vec::new();

        for i_elec in 0..self.num_elecs {
            self.charges_from_electron
                .push(new_data_real(self.grid_n_charge));
            self.V_from_elecs.push(arr_real_2d.clone());

            // todo: Call procedures::update_bases_weights etc here.
            let sfcs = &mut self.surfaces_per_elec[i_elec];

            // Assigning vars prevents multiple-borrow-mut vars.
            let psi = &mut sfcs.psi_per_basis;
            let psi_pp = &mut sfcs.derivs_per_basis;
            let spinor = &mut sfcs.spinor_per_basis;
            let spinor_derivs = &mut sfcs.spinor_derivs_per_basis;

            wf_ops::wf_from_bases(
                &self.dev_psi,
                psi,
                // Some(psi_pp),
                psi_pp,
                &self.bases[i_elec],
                &self.surfaces_shared.grid_posits,
                self.deriv_calc,
            );

            // todo: Put back A/R
            // wf_ops::wf_from_bases_spinor(
            //     dev_psi,
            //     spinor,
            //     Some(spinor_derivs),
            //     &bases_per_elec_spinor[i_elec],
            //     &surfaces_shared.grid_posits,
            // );

            let psi = &mut sfcs.psi;
            let charge_density_2d = &mut sfcs.charge_density_2d;
            let psi_pp = &mut sfcs.derivs;
            let spinor = &mut sfcs.spinor;
            let spinor_derivs = &mut sfcs.spinor_derivs;

            let weights: Vec<f64> = self.bases[i_elec].iter().map(|b| b.weight()).collect();

            wf_ops::mix_bases(
                psi,
                charge_density_2d,
                psi_pp,
                &sfcs.psi_per_basis,
                &sfcs.derivs_per_basis,
                &weights,
            );

            // todo: 2D conversion: When do we update psi charge density?

            // wf_ops::mix_bases_charge(
            //     psi,
            //     charge_density,
            //     &sfcs.psi_per_basis,
            //     &weights,
            // );

            wf_ops::mix_bases_spinor(
                spinor,
                None, // todo
                Some(spinor_derivs),
                &sfcs.spinor_per_basis,
                Some(&sfcs.spinor_derivs_per_basis),
                &weights,
            );

            wf_ops::update_eigen_vals(
                &mut sfcs.V_elec_eigen,
                &mut sfcs.V_total_eigen,
                &mut sfcs.V_diff,
                &mut sfcs.psi_pp_calculated,
                &sfcs.psi,
                &sfcs.derivs,
                // &sfcs.psi_pp_div_psi_evaluated,
                &sfcs.V_acting_on_this,
                sfcs.E,
                &self.surfaces_shared.V_from_nuclei,
                &self.surfaces_shared.grid_posits,
                &mut sfcs.psi_fm_H,
                &mut sfcs.psi_fm_L2,
                &mut sfcs.psi_fm_Lz,
            );

            // todo: Put back A/R
            // wf_ops::update_eigen_vals_spinor(
            //     &mut sfcs.spinor_calc,
            //     spinor_derivs,
            //     [-0.5; 4], // todo temp
            //     &sfcs.V_acting_on_this,
            // );

            let mut psi_charge = Vec::new();

            for _ in 0..self.bases[i_elec].len() {
                // Handle the charge-grid-evaluated psi.
                psi_charge.push(new_data(self.grid_n_charge));
            }

            // wf_ops::wf_from_bases(
            wf_ops::wf_from_bases_charge(
                &self.dev_psi,
                &mut psi_charge,
                &self.bases[i_elec],
                &self.surfaces_shared.grid_posits_charge,
            );

            procedures::create_elec_charge(
                &mut self.charges_from_electron[i_elec],
                &psi_charge,
                &weights,
                self.grid_n_charge,
            );

            self.psi_charge.push(psi_charge);

            // todo: Create electron V here
            // potential::create_V_from_elecs(
            //     dev,
            //     &mut V_from_elecs[i_elec],
            //     &state.surfaces_shared.grid_posits,
            //     &state.surfaces_shared.grid_posits_charge,
            //     &charges_other_elecs,
            //     state.grid_n_render,
            //     state.grid_n_charge,
            //     state.ui.create_2d_electron_V,
            // );

            for electron in &mut self.surfaces_per_elec {
                // todo: Come back to A/R
                potential::update_V_acting_on_elec(
                    &mut electron.V_acting_on_this,
                    &self.surfaces_shared.V_from_nuclei,
                    // &[], // Not ready to apply V from other elecs yet.
                    // &new_data_real(grid_n_sample), // Not ready to apply V from other elecs yet.
                    &new_data_2d_real(grid_n), // Not ready to apply V from other elecs yet.
                    grid_n,
                );
            }
        }
        // todo: Put back A/R
        // wf_ops::update_combined(&mut surfaces_shared, &surfaces_per_elec, grid_n_sample);
    }

    /// Replace nuclei and electron data with that from a preset.
    pub fn set_preset(&mut self, preset: usize) {
        // Reset relevant state variables.
        self.nucleii = Vec::new();
        self.net_force_on_nuc = Vec::new();
        self.bases = Vec::new();
        self.bases_spinor = Vec::new();

        // todo: Sort out how to handle electron distro across nuclei.

        self.num_elecs = self.presets[preset].elecs.len();

        for nuc in &self.presets[preset].nuclei {
            self.nucleii
                .push((nuc.posit, Q_PROT * nuc.num_protons as f64));
        }

        // todo: YOu will need to re-think how you manage electrons, as distributed across nuclei.
        for bases in &self.presets[preset].elecs {
            let mut bases_this_elec = Vec::new();

            for basis in bases {
                // for (i, nuc) in self.presets[preset].nuclei.iter().enumerate() {
                // let mut sto_ = sto.clone();

                // sto_.nuc_id = i;
                // sto_.posit = nuc.posit;
                bases_this_elec.push(basis.clone());
                // }
            }

            self.bases.push(bases_this_elec);
        }

        self.net_force_on_nuc = vec![Vec3::new_zero(); self.nucleii.len()];
    }
}
