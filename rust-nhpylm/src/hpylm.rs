use pyp::*;

pub trait HPYLM<T> {
  fn get_num_nodes(&self) -> usize;
  fn get_num_tables(&self) -> usize;
  fn get_num_customers(&self) -> usize;
  fn get_pass_counts(&self) -> usize;
  fn get_stop_counts(&self) -> usize;
  fn init_hyperparameters_at_depth_if_needed(&mut self, depth: usize);
  fn sum_auxiliary_variables_recursively(
    &mut self,
    node: &PYP<T>,
    sum_log_x_u_array: &mut Vec<f64>,
    sum_y_ui_array: &mut Vec<f64>,
    sum_one_minus_y_ui_array: &mut Vec<f64>,
    sum_one_minus_z_uwkj_array: &mut Vec<f64>,
    bottom: &mut usize,
  );
  // fn sum_auxiliary_variables (
  //   &mut self
  // );
  fn sample_hyperparameters(&mut self);
}

// pub struct HPYLM<T> {
//   root:
// }
