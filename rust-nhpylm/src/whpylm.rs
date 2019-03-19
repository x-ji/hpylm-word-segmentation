use def::*;
use hpylm::HPYLM;
use pyp::*;

pub struct WHPYLM {
  root: PYP<u64>,
  depth: usize,
  g_0: f64,
  d_array: Vec<f64>,
  theta_array: Vec<f64>,
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
  fn new(order: usize) -> Self {
    let whpylm = Self {
      root: PYP::new(0),
      depth: 0.max(order - 1),
      d_array: vec![HPYLM_INITIAL_d; order],
      theta_array: vec![HPYLM_INITIAL_theta; order],
      a_array: vec![HPYLM_a; order],
      b_array: vec![HPYLM_b; order],
      alpha_array: vec![HPYLM_alpha; order],
      beta_array: vec![HPYLM_beta; order],
      g_0: 0.0,
    };

    whpylm
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
  fn init_hyperparameters_at_depth_if_needed(&mut self, depth: usize) {
    if self.d_array.len() <= depth {
      while self.d_array.len() <= depth {
        self.d_array.push(HPYLM_INITIAL_d);
      }
    }
    if self.theta_array.len() <= depth {
      while self.theta_array.len() <= depth {
        self.theta_array.push(HPYLM_INITIAL_theta);
      }
    }
    if self.a_array.len() <= depth {
      while self.a_array.len() <= depth {
        self.a_array.push(HPYLM_a);
      }
    }
    if self.b_array.len() <= depth {
      while self.b_array.len() <= depth {
        self.b_array.push(HPYLM_b);
      }
    }
    if self.alpha_array.len() <= depth {
      while self.alpha_array.len() <= depth {
        self.alpha_array.push(HPYLM_alpha);
      }
    }
    if self.beta_array.len() <= depth {
      while self.beta_array.len() <= depth {
        self.beta_array.push(HPYLM_beta);
      }
    }
  }
  fn sum_auxiliary_variables_recursively(
    &mut self,
    node: &PYP<u64>,
    sum_log_x_u_array: &mut Vec<f64>,
    sum_y_ui_array: &mut Vec<f64>,
    sum_one_minus_y_ui_array: &mut Vec<f64>,
    sum_one_minus_z_uwkj_array: &mut Vec<f64>,
    bottom: &mut usize,
  ) {
    for child in node.children.values() {
      let depth = child.depth;
      if depth > *bottom {
        *bottom = depth;
      }
      self.init_hyperparameters_at_depth_if_needed(depth);

      let d = self.d_array[depth];
      let theta = self.theta_array[depth];
      sum_log_x_u_array[depth] += node.sample_log_x_u(theta);
      sum_y_ui_array[depth] += node.sample_summed_y_ui(d, theta, false);
      sum_one_minus_y_ui_array[depth] += node.sample_summed_y_ui(d, theta, true);
      sum_one_minus_z_uwkj_array[depth] += node.sample_summed_one_minus_z_uwkj(d);

      self.sum_auxiliary_variables_recursively(
        child,
        sum_log_x_u_array,
        sum_y_ui_array,
        sum_one_minus_y_ui_array,
        sum_one_minus_z_uwkj_array,
        bottom,
      );
    }
  }

  fn sample_hyperparameters(&mut self) {
    let max_depth: usize = self.d_array.len() - 1;
    let mut sum_log_x_u_array = vec![0.0; max_depth + 1];
    let mut sum_y_ui_array = vec![0.0; max_depth + 1];
    let mut sum_one_minus_y_ui_array = vec![0.0; max_depth + 1];
    let mut sum_one_minus_z_uwkj_array = vec![0.0; max_depth + 1];

    sum_log_x_u_array[0] = self.root.sample_log_x_u(self.theta_array[0]);
    sum_y_ui_array[0] += self
      .root
      .sample_summed_y_ui(self.d_array[0], self.theta_array[0], false);
    sum_one_minus_y_ui_array[0] +=
      self
        .root
        .sample_summed_y_ui(self.d_array[0], self.theta_array[0], true);
    sum_one_minus_z_uwkj_array[0] += self.root.sample_summed_one_minus_z_uwkj(self.d_array[0]);

    self.depth = 0;
    self.sum_auxiliary_variables_recursively(
      &self.root,
      &mut sum_log_x_u_array,
      &mut sum_y_ui_array,
      &mut sum_one_minus_y_ui_array,
      &mut sum_one_minus_z_uwkj_array,
      &mut self.depth,
    )
  }
}