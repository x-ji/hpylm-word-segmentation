use def::*;
use pyp::*;

pub trait HPYLM<T> {
  fn get_num_nodes(&self) -> usize;
  fn get_num_tables(&self) -> usize;
  fn get_num_customers(&self) -> usize;
  fn get_pass_counts(&self) -> usize;
  fn get_stop_counts(&self) -> usize;
  fn sample_hyperparameters(&mut self);
}

pub fn init_hyperparameters_at_depth_if_needed(
  d_array: &mut Vec<f64>,
  theta_array: &mut Vec<f64>,
  a_array: &mut Vec<f64>,
  b_array: &mut Vec<f64>,
  alpha_array: &mut Vec<f64>,
  beta_array: &mut Vec<f64>,
  depth: usize,
) {
  if d_array.len() <= depth {
    while d_array.len() <= depth {
      d_array.push(HPYLM_INITIAL_D);
    }
  }
  if theta_array.len() <= depth {
    while theta_array.len() <= depth {
      theta_array.push(HPYLM_INITIAL_THETA);
    }
  }
  if a_array.len() <= depth {
    while a_array.len() <= depth {
      a_array.push(HPYLM_A);
    }
  }
  if b_array.len() <= depth {
    while b_array.len() <= depth {
      b_array.push(HPYLM_B);
    }
  }
  if alpha_array.len() <= depth {
    while alpha_array.len() <= depth {
      alpha_array.push(HPYLM_ALPHA);
    }
  }
  if beta_array.len() <= depth {
    while beta_array.len() <= depth {
      beta_array.push(HPYLM_BETA);
    }
  }
}

pub fn sum_auxiliary_variables_recursively<T>(
  node: &PYP<T>,
  sum_log_x_u_array: &mut Vec<f64>,
  sum_y_ui_array: &mut Vec<f64>,
  sum_one_minus_y_ui_array: &mut Vec<f64>,
  sum_one_minus_z_uwkj_array: &mut Vec<f64>,
  d_array: &mut Vec<f64>,
  theta_array: &mut Vec<f64>,
  a_array: &mut Vec<f64>,
  b_array: &mut Vec<f64>,
  alpha_array: &mut Vec<f64>,
  beta_array: &mut Vec<f64>,
  bottom: &mut usize,
) where
  T: std::cmp::Eq,
  T: std::hash::Hash,
  T: Copy,
{
  for child in node.children.values() {
    let depth = child.depth;
    if depth > *bottom {
      *bottom = depth;
    }

    init_hyperparameters_at_depth_if_needed(
      d_array,
      theta_array,
      a_array,
      b_array,
      alpha_array,
      beta_array,
      depth,
    );

    let d = d_array[depth];
    let theta = theta_array[depth];
    sum_log_x_u_array[depth] += node.sample_log_x_u(theta);
    sum_y_ui_array[depth] += node.sample_summed_y_ui(d, theta, false);
    sum_one_minus_y_ui_array[depth] += node.sample_summed_y_ui(d, theta, true);
    sum_one_minus_z_uwkj_array[depth] += node.sample_summed_one_minus_z_uwkj(d);

    sum_auxiliary_variables_recursively(
      child,
      sum_log_x_u_array,
      sum_y_ui_array,
      sum_one_minus_y_ui_array,
      sum_one_minus_z_uwkj_array,
      d_array,
      theta_array,
      a_array,
      b_array,
      alpha_array,
      beta_array,
      bottom,
    );
  }
}
