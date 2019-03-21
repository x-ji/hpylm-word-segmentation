// Seems that I'll have to prepend self:: because this is a library?
use def::*;
use either::*;
use rand::distributions::{Bernoulli, Beta, Distribution, WeightedIndex};
use rand::prelude::*;
use rand::Rng;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

pub fn init_hyperparameters_at_depth_if_needed(
  depth: usize,
  d_array: &mut Vec<f64>,
  theta_array: &mut Vec<f64>,
) {
  if depth >= d_array.len() {
    while d_array.len() <= depth {
      d_array.push(HPYLM_INITIAL_D);
    }
    while theta_array.len() <= depth {
      theta_array.push(HPYLM_INITIAL_THETA);
    }
  }
}

#[derive(Clone)]
pub struct PYP<T> {
  pub children: HashMap<T, PYP<T>>,
  // https://stackoverflow.com/questions/36167160/how-do-i-express-mutually-recursive-data-structures-in-safe-rust
  parent: Option<*mut PYP<T>>,
  pub tablegroups: HashMap<T, Vec<usize>>,
  pub ntables: usize,
  pub ncustomers: usize,
  pub stop_count: usize,
  pub pass_count: usize,
  pub depth: usize,
  pub context: T,
}

impl<T> PYP<T>
where
  T: std::cmp::Eq,
  T: std::hash::Hash,
  T: Copy,
{
  pub fn new(context: T) -> Self {
    Self {
      children: HashMap::new(),
      parent: None,
      tablegroups: HashMap::new(),
      ntables: 0,
      ncustomers: 0,
      stop_count: 0,
      pass_count: 0,
      depth: 0,
      context: context,
    }
  }

  pub fn need_to_remove_from_parent(&self) -> bool {
    if self.parent == None {
      return false;
    } else if self.children.is_empty() && self.tablegroups.is_empty() {
      return true;
    } else {
      return false;
    }
  }

  pub fn get_num_tables_serving_dish(&self, dish: T) -> usize {
    let tablegroup = self.tablegroups.get(&dish);
    match tablegroup {
      None => 0,
      Some(t) => t.into_iter().sum(),
    }
  }

  pub fn find_child_pyp(&mut self, dish: T, generate_if_not_found: bool) -> Option<&mut PYP<T>> {
    if self.children.contains_key(&dish) {
      return self.children.get_mut(&dish);
    }

    if !generate_if_not_found {
      return None;
    }

    let mut child = PYP::new(dish);
    child.parent = Some(self);
    child.depth = self.depth + 1;
    self.children.insert(dish, child);
    // Fucking hell this actually worked!
    // return Some(self.children.get(&dish).unwrap());
    return self.children.get_mut(&dish);
  }

  pub fn add_customer_to_table(
    &mut self,
    dish: T,
    table_index: usize,
    g0_or_parent_p_ws: Either<f64, &Vec<f64>>,
    d_array: &mut Vec<f64>,
    theta_array: &mut Vec<f64>,
    table_index_in_root: &mut usize,
  ) -> bool {
    if !self.tablegroups.contains_key(&dish) {
      return PYP::add_customer_to_new_table(
        self,
        dish,
        g0_or_parent_p_ws,
        d_array,
        theta_array,
        table_index_in_root,
      );
    } else {
      let mut tablegroup = self.tablegroups.get_mut(&dish).unwrap();
      tablegroup[table_index] += 1;
      self.ncustomers += 1;
      return true;
    }
  }

  pub fn add_customer_to_new_table(
    &mut self,
    dish: T,
    g0_or_parent_p_ws: Either<f64, &Vec<f64>>,
    d_array: &mut Vec<f64>,
    theta_array: &mut Vec<f64>,
    table_index_in_root: &mut usize,
  ) -> bool {
    self._add_customer_to_new_table(dish);

    match self.parent {
      None => return true,
      Some(p) => unsafe {
        return (*p).add_customer(
          dish,
          g0_or_parent_p_ws,
          d_array,
          theta_array,
          false,
          table_index_in_root,
        );
      },
    }
  }

  pub fn _add_customer_to_new_table(&mut self, dish: T) {
    match self.tablegroups.entry(dish) {
      Entry::Vacant(e) => {
        e.insert(vec![1]);
      }
      Entry::Occupied(mut e) => {
        e.get_mut().push(1);
      }
    }

    self.ntables += 1;
    self.ncustomers += 1;
  }

  pub fn remove_customer_from_table(
    &mut self,
    dish: T,
    table_index: usize,
    table_index_in_root: &mut usize,
  ) -> bool {
    let is_empty = {
      let mut tablegroup = self.tablegroups.get_mut(&dish).unwrap();
      tablegroup[table_index] -= 1;
      self.ncustomers -= 1;

      // If there are no customers anymore at this table, we need to remove this table.
      if tablegroup[table_index] == 0 {
        match self.parent {
          None => {}
          Some(p) => unsafe {
            (*p).remove_customer(dish, false, table_index_in_root);
          },
        }

        tablegroup.remove(table_index);
        self.ntables -= 1;
      }

      tablegroup.is_empty()
    };

    if is_empty {
      // Does this work if I'm still borrowing the value? I guess so since the borrowed value only goes out scope when the function itself finishes. Let's see then.
      self.tablegroups.remove(&dish);
    }

    return true;
  }

  pub fn add_customer(
    &mut self,
    dish: T,
    g0_or_parent_p_ws: Either<f64, &Vec<f64>>,
    d_array: &mut Vec<f64>,
    theta_array: &mut Vec<f64>,
    update_beta_count: bool,
    index_of_table_in_root: &mut usize,
  ) -> bool {
    init_hyperparameters_at_depth_if_needed(self.depth, d_array, theta_array);
    let d_u = d_array[self.depth];
    let theta_u = theta_array[self.depth];
    let parent_p_w: f64 = match g0_or_parent_p_ws {
      Left(g0) => match self.parent {
        None => g0,
        Some(p) => unsafe { (*p).compute_p_w(dish, g0, d_array, theta_array) },
      },
      Right(parent_p_ws) => parent_p_ws[self.depth],
    };

    if !self.tablegroups.contains_key(&dish) {
      self.add_customer_to_new_table(
        dish,
        g0_or_parent_p_ws,
        d_array,
        theta_array,
        index_of_table_in_root,
      );
      if update_beta_count == true {
        self.increment_stop_count();
      }
      return true;
    } else {
      let mut sum: f64 = 0.0;
      // Apparently we'll have to make a clone here otherwise we'd be borrowing `self` as a whole.
      let tablegroup = self.tablegroups.get(&dish).unwrap().clone();
      for k in 0..tablegroup.len() {
        // Man comparing two floats surely is convoluted...
        let temp: f64 = tablegroup[k] as f64 - d_u;
        sum += temp.max(0.0);
      }
      let t_u = self.ntables as f64;
      sum += (theta_u + d_u * t_u) * parent_p_w;

      let mut rng = rand::thread_rng();
      let normalizer = 1.0 / sum;
      let bernoulli: f64 = rng.gen();
      let mut stack = 0.0;
      for k in 0..tablegroup.len() {
        let temp: f64 = tablegroup[k] as f64 - d_u;
        stack += temp.max(0.0) * normalizer;
        if bernoulli <= stack {
          self.add_customer_to_table(
            dish,
            k,
            g0_or_parent_p_ws,
            d_array,
            theta_array,
            index_of_table_in_root,
          );
          if update_beta_count {
            self.increment_stop_count();
          }
          if self.depth == 0 {
            *index_of_table_in_root = k;
          }

          return true;
        }
      }

      self.add_customer_to_new_table(
        dish,
        g0_or_parent_p_ws,
        d_array,
        theta_array,
        index_of_table_in_root,
      );

      if update_beta_count {
        self.increment_stop_count();
      }
      if self.depth == 0 {
        *index_of_table_in_root = tablegroup.len();
      }

      return true;
    }
  }

  pub fn remove_customer(
    &mut self,
    dish: T,
    update_beta_count: bool,
    index_of_table_in_root: &mut usize,
  ) -> bool {
    let tablegroup = self.tablegroups.get(&dish).unwrap().clone();

    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(&tablegroup).unwrap();
    let index_to_remove = dist.sample(&mut rng);
    self.remove_customer_from_table(dish, index_to_remove, index_of_table_in_root);
    if update_beta_count == true {
      self.decrement_stop_count();
    }
    if self.depth == 0 {
      *index_of_table_in_root = index_to_remove;
    }

    return true;

    // As we use rand::WeightedIndex, there's no need for the manual sampling method anymore.
    // let sum = tablegroup.into_iter().sum();
    // let normalizer = 1.0 / sum;
    // let bernoulli = rng.gen();
    // let mut stack = 0.0;
    // for k in 0..tablegroup.len() {
    //   stack += tablegroup[k] * normalizer;
    //   if bernoulli <= stack {
    //     self.remove_customer_from_table(dish, k, index_of_table_in_root);
    //     if (update_beta_count == true) {
    //       self.decrement_stop_count();
    //     }
    //     if (self.depth == 0) {
    //       *index_of_table_in_root = k;
    //     }
    //     return true;
    //   }
    // }

    // self.remove_customer_from_table(dish, tablegroup.len() - 1, index_of_table_in_root);
  }

  pub fn compute_p_w(
    &mut self,
    dish: T,
    g_0: f64,
    d_array: &mut Vec<f64>,
    theta_array: &mut Vec<f64>,
  ) -> f64 {
    init_hyperparameters_at_depth_if_needed(self.depth, d_array, theta_array);
    let d_u = d_array[self.depth];
    let theta_u = theta_array[self.depth];
    let t_u = self.ntables as f64;
    let c_u = self.ncustomers as f64;
    match self.tablegroups.entry(dish) {
      Entry::Vacant(e) => {
        let coeff: f64 = (theta_u + d_u * t_u) / (theta_u + c_u);
        match self.parent {
          None => g_0 * coeff,
          Some(p) => unsafe { (*p).compute_p_w(dish, g_0, d_array, theta_array) * coeff },
        }
      }
      Entry::Occupied(e) => {
        let parent_p_w = match self.parent {
          None => g_0,
          Some(p) => unsafe { (*p).compute_p_w(dish, g_0, d_array, theta_array) },
        };
        let tablegroup = e.get();
        let c_uw: usize = tablegroup.into_iter().sum();
        let t_uw = tablegroup.len() as f64;
        let first_term: f64 = (c_uw as f64 - d_u * t_uw).max(0.0) / (theta_u + c_u);
        let second_coeff: f64 = (theta_u + d_u * t_u) / (theta_u + c_u);
        return first_term + second_coeff * parent_p_w;
      }
    }
  }

  pub fn compute_p_w_with_parent_p_w(
    &mut self,
    dish: T,
    parent_p_w: f64,
    d_array: &mut Vec<f64>,
    theta_array: &mut Vec<f64>,
  ) -> f64 {
    init_hyperparameters_at_depth_if_needed(self.depth, d_array, theta_array);
    let d_u = d_array[self.depth];
    let theta_u = theta_array[self.depth];
    let t_u = self.ntables as f64;
    let c_u = self.ncustomers as f64;
    match self.tablegroups.entry(dish) {
      Entry::Vacant(e) => {
        let coeff: f64 = (theta_u + d_u * t_u) / (theta_u + c_u);
        return parent_p_w * coeff;
      }
      Entry::Occupied(e) => {
        let tablegroup = e.get();
        let c_uw: usize = tablegroup.into_iter().sum();
        let t_uw = tablegroup.len() as f64;
        let first_term: f64 = (c_uw as f64 - d_u * t_uw).max(0.0) / (theta_u + c_u);
        let second_coeff: f64 = (theta_u + d_u * t_u) / (theta_u + c_u);
        return first_term + second_coeff * parent_p_w;
      }
    }
  }

  /* The following methods are specifically related to the character variant of PYP */

  pub fn stop_probability(&self, beta_stop: f64, beta_pass: f64, recursive: bool) -> f64 {
    let prob = (self.stop_count as f64 + beta_stop)
      / (self.stop_count as f64 + self.pass_count as f64 + beta_stop + beta_pass);
    if !recursive {
      return prob;
    } else {
      match self.parent {
        None => return prob,
        Some(p) => unsafe {
          return prob * (*p).pass_probability(beta_stop, beta_pass, recursive);
        },
      }
    }
  }

  pub fn pass_probability(&self, beta_stop: f64, beta_pass: f64, recursive: bool) -> f64 {
    let prob = (self.stop_count as f64 + beta_pass)
      / (self.stop_count as f64 + self.pass_count as f64 + beta_stop + beta_pass);
    if !recursive {
      return prob;
    } else {
      match self.parent {
        None => return prob,
        Some(p) => unsafe {
          return prob * (*p).pass_probability(beta_stop, beta_pass, recursive);
        },
      }
    }
  }

  pub fn increment_stop_count(&mut self) {
    self.stop_count += 1;
    match self.parent {
      None => {}
      Some(p) => unsafe { (*p).decrement_pass_count() },
    }
  }

  pub fn decrement_stop_count(&mut self) {
    self.stop_count -= 1;
    match self.parent {
      None => {}
      Some(p) => unsafe { (*p).decrement_pass_count() },
    }
  }

  pub fn increment_pass_count(&mut self) {
    self.pass_count += 1;
    match self.parent {
      None => {}
      Some(p) => unsafe { (*p).increment_pass_count() },
    }
  }

  pub fn decrement_pass_count(&mut self) {
    self.pass_count -= 1;
    match self.parent {
      None => {}
      Some(p) => unsafe { (*p).decrement_pass_count() },
    }
  }

  pub fn remove_from_parent(&self) -> bool {
    match self.parent {
      None => false,
      Some(p) => unsafe {
        (*p).delete_child_node(self.context);
        true
      },
    }
  }

  pub fn delete_child_node(&mut self, dish: T) {
    // let child = { self.find_child_pyp(dish, false) };
    // Yeah this is great. Think through the perspectives then. Why you need the variables in the first place, etc.
    // The thing about Rust then is that you won't need manual memory management. In C++ you'd need to write a line `delete child`, which here is not necessary. Thus the system like this.
    let has_this_child = { self.find_child_pyp(dish, false).is_some() };
    if has_this_child {
      self.children.remove(&dish);
    } else if self.children.len() == 0 && self.tablegroups.len() == 0 {
      self.remove_from_parent();
    }
  }

  pub fn get_max_depth(&self, base: usize) -> usize {
    let mut max_depth = base;
    for child in self.children.values() {
      let depth = child.get_max_depth(base + 1);
      if depth > max_depth {
        max_depth = depth;
      }
    }
    max_depth
  }

  pub fn get_num_nodes(&self) -> usize {
    let mut count = self.children.len();
    for child in self.children.values() {
      count += child.get_num_nodes();
    }
    count
  }

  pub fn get_num_tables(&self) -> usize {
    let mut count = self.ntables;
    // for tablegroup in self.tablegroups.values() {
    //   count += tablegroup.len();
    // }
    for child in self.children.values() {
      count += child.get_num_tables();
    }
    count
  }

  pub fn get_num_customers(&self) -> usize {
    let mut count = self.ncustomers;
    for child in self.children.values() {
      count += child.get_num_customers();
    }
    count
  }

  pub fn get_pass_counts(&self) -> usize {
    let mut count = self.pass_count;
    for child in self.children.values() {
      count += child.get_pass_counts();
    }
    count
  }

  pub fn get_stop_counts(&self) -> usize {
    let mut count = self.stop_count;
    for child in self.children.values() {
      count += child.get_stop_counts();
    }
    count
  }

  // Don't think this is actually used.
  // pub fn get_all_pyps_at_depth(&self, depth: usize, acc: &mut Vec<&PYP<T>>) {
  //   if self.depth == depth {
  //     acc.push(self);
  //   }
  // }

  /* The functions below are related to hyperparameter (d, θ) sampling, based on the algorithm given in the Teh Technical Report

  There are 3 auxiliary variables defined, x_**u**, y_**u**i, z**u**wkj.

  The following methods sample them. */

  pub fn sample_log_x_u(&self, theta_u: f64) -> f64 {
    if self.ncustomers >= 2 {
      let dist = Beta::new(theta_u + 1.0, self.ncustomers as f64 - 1.0);
      // Prevent underflow.
      let sample = dist.sample(&mut rand::thread_rng()) + 1e-8;
      return sample.ln();
    } else {
      return 0.0;
    }
  }

  pub fn sample_summed_y_ui(&self, d_u: f64, theta_u: f64, is_one_minus: bool) -> f64 {
    if self.ntables >= 2 {
      let mut sum = 0;
      for i in 1..self.ntables - 1 {
        let denom = theta_u + d_u * i as f64;
        let prob = theta_u / denom;
        let dist = Bernoulli::new(prob);
        let mut y_ui = dist.sample(&mut rand::thread_rng());
        if is_one_minus {
          y_ui = !y_ui;
        }
        match y_ui {
          true => sum += 1,
          false => sum += 0,
        }
      }
      return sum as f64;
    } else {
      return 0.0;
    }
  }

  pub fn sample_summed_one_minus_z_uwkj(&self, d_u: f64) -> f64 {
    let mut sum = 0;
    for tablegroup in self.tablegroups.values() {
      for customercount in tablegroup {
        if customercount >= &2 {
          for j in 1..customercount - 1 {
            let prob = (j - 1) as f64 / (j as f64 - d_u);
            let dist = Bernoulli::new(prob);
            let result = if dist.sample(&mut rand::thread_rng()) {
              1
            } else {
              0
            };
            sum += 1 - result;
          }
        }
      }
    }
    return sum as f64;
  }
}
