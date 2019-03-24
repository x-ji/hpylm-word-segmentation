use def::*;
use hpylm::{init_hyperparameters_at_depth_if_needed, sum_auxiliary_variables_recursively, HPYLM};
use pyp::*;
use rand::distributions::{Beta, Gamma};
use rand::prelude::*;

pub struct WHPYLM {
    pub root: PYP<u64>,
    pub depth: usize,
    pub g_0: f64,
    pub d_array: Vec<f64>,
    pub theta_array: Vec<f64>,
    /*
      These variables are related to the sampling process as described in the Teh technical report, expressions (40) and (41)

      Note that they do *not* directly correspond to the alpha, beta parameters of a Beta distribution, nor the shape and scale parameters of a Gamma distribution.
    */
    a_array: Vec<f64>,
    b_array: Vec<f64>,
    alpha_array: Vec<f64>,
    beta_array: Vec<f64>,
}

impl WHPYLM {
    pub fn new(order: usize) -> Self {
        Self {
            root: PYP::new(0),
            depth: 0.max(order - 1),
            d_array: vec![HPYLM_INITIAL_D; order],
            theta_array: vec![HPYLM_INITIAL_THETA; order],
            a_array: vec![HPYLM_A; order],
            b_array: vec![HPYLM_B; order],
            alpha_array: vec![HPYLM_ALPHA; order],
            beta_array: vec![HPYLM_BETA; order],
            g_0: 0.0,
        }
    }
}

impl HPYLM<u64> for WHPYLM {
    fn get_num_nodes(&self) -> usize {
        return self.root.get_num_nodes() + 1;
    }
    fn get_num_tables(&self) -> usize {
        return self.root.get_num_tables() + 1;
    }
    fn get_num_customers(&self) -> usize {
        return self.root.get_num_customers() + 1;
    }
    fn get_pass_counts(&self) -> usize {
        return self.root.get_pass_counts() + 1;
    }
    fn get_stop_counts(&self) -> usize {
        return self.root.get_stop_counts() + 1;
    }

    fn sample_hyperparameters(&mut self) {
        let max_depth: usize = self.d_array.len() - 1;
        let mut sum_log_x_u_array = vec![0.0; max_depth + 1];
        let mut sum_y_ui_array = vec![0.0; max_depth + 1];
        let mut sum_one_minus_y_ui_array = vec![0.0; max_depth + 1];
        let mut sum_one_minus_z_uwkj_array = vec![0.0; max_depth + 1];

        self.depth = 0;
        sum_auxiliary_variables_recursively(
            &self.root,
            &mut sum_log_x_u_array,
            &mut sum_y_ui_array,
            &mut sum_one_minus_y_ui_array,
            &mut sum_one_minus_z_uwkj_array,
            &mut self.d_array,
            &mut self.theta_array,
            &mut self.a_array,
            &mut self.b_array,
            &mut self.alpha_array,
            &mut self.beta_array,
            &mut self.depth,
        );

        init_hyperparameters_at_depth_if_needed(
            &mut self.d_array,
            &mut self.theta_array,
            &mut self.a_array,
            &mut self.b_array,
            &mut self.alpha_array,
            &mut self.beta_array,
            self.depth,
        );

        for u in 0..self.depth {
            let dist1 = Beta::new(
                self.a_array[u] + sum_one_minus_y_ui_array[u],
                self.b_array[u] + sum_one_minus_z_uwkj_array[u],
            );
            self.d_array[u] = dist1.sample(&mut rand::thread_rng());

            let dist2 = Gamma::new(
                self.alpha_array[u] + sum_y_ui_array[u],
                1.0 / (self.beta_array[u] - sum_log_x_u_array[u]),
            );
            self.theta_array[u] = dist2.sample(&mut rand::thread_rng());
        }

        let excessive_length = max_depth - self.depth;
        for _ in 0..excessive_length {
            self.d_array.pop();
            self.theta_array.pop();
            self.a_array.pop();
            self.b_array.pop();
            self.alpha_array.pop();
            self.beta_array.pop();
        }
    }
}
