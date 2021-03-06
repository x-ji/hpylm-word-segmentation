use def::*;
use pyp::*;

use either::*;
use hpylm::{init_hyperparameters_at_depth_if_needed, sum_auxiliary_variables_recursively, HPYLM};
use rand::distributions::WeightedIndex;
use rand::distributions::{Beta, Gamma};
use rand::prelude::*;

pub struct CHPYLM {
    pub root: PYP<char>,
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

    /* Fields specific to CHPYLM */
    pub beta_stop: f64,
    pub beta_pass: f64,
    pub max_depth: usize,
    parent_p_w_cache: Vec<f64>,
    path_nodes: Vec<Option<*mut PYP<char>>>,
}

impl CHPYLM {
    pub fn new(g_0: f64, max_depth: usize, beta_stop: f64, beta_pass: f64) -> Self {
        let root = PYP::new(BOW);
        Self {
            root: root,
            depth: 0,
            d_array: Vec::new(),
            theta_array: Vec::new(),
            a_array: Vec::new(),
            b_array: Vec::new(),
            alpha_array: Vec::new(),
            beta_array: Vec::new(),
            g_0: g_0,
            beta_stop: beta_stop,
            beta_pass: beta_pass,
            max_depth: max_depth,
            parent_p_w_cache: vec![0.0; max_depth],
            path_nodes: vec![None; max_depth],
        }
    }

    pub fn add_customer_at_index_n(
        &mut self,
        characters: &Vec<char>,
        n: usize,
        depth: usize,
        with_cache: bool,
    ) -> bool {
        let char_n = characters[n];
        let mut root_table_index = 0;
        let node = if with_cache {
            self.find_node_by_tracing_back_context_path_nodes_cache(characters, n, depth)
                .unwrap()
        } else {
            self.find_node_by_tracing_back_context_parent_p_w(characters, n, depth)
                .unwrap()
        };
        unsafe {
            return (*node).add_customer(
                char_n,
                Right(&self.parent_p_w_cache),
                &mut self.d_array,
                &mut self.theta_array,
                true,
                &mut root_table_index,
            );
        }
    }

    pub fn remove_customer_at_index_n(
        &mut self,
        characters: &Vec<char>,
        n: usize,
        depth: usize,
    ) -> bool {
        let char_n = characters[n];
        let mut root_table_index = 0;
        let node = self
            .find_node_by_tracing_back_context_removal(characters, n, depth, false, false)
            .unwrap();

        unsafe {
            (*node).remove_customer(char_n, true, &mut root_table_index);

            if (*node).need_to_remove_from_parent() {
                (*node).remove_from_parent();
            }
        }
        return true;
    }

    fn find_node_by_tracing_back_context_removal(
        &mut self,
        characters: &Vec<char>,
        n: usize,
        depth_of_n: usize,
        generate_if_not_found: bool,
        return_cur_node_if_not_found: bool,
    ) -> Option<*mut PYP<char>> {
        if n < depth_of_n {
            return None;
        }

        let mut cur_node = &mut self.root as *mut PYP<char>;
        for d in 1..depth_of_n + 1 {
            let context = characters[n - d];
            unsafe {
                let child = (*cur_node).find_child_pyp(context, generate_if_not_found);
                match child {
                    None => {
                        if return_cur_node_if_not_found {
                            return Some(cur_node);
                        } else {
                            return None;
                        }
                    }
                    Some(c) => cur_node = c,
                }
            }
        }
        return Some(cur_node);
    }

    fn find_node_by_tracing_back_context_parent_p_w(
        &mut self,
        characters: &Vec<char>,
        n: usize,
        depth_of_n: usize,
    ) -> Option<*mut PYP<char>> {
        if n < depth_of_n {
            return None;
        }

        let char_n = characters[n];

        let mut cur_node = &mut self.root as *mut PYP<char>;
        let mut parent_p_w = self.g_0;
        self.parent_p_w_cache[0] = parent_p_w;

        unsafe {
            for d in 1..depth_of_n + 1 {
                let p_w = (*cur_node).compute_p_w_with_parent_p_w(
                    char_n,
                    parent_p_w,
                    &mut self.d_array,
                    &mut self.theta_array,
                );
                let context = characters[n - d];
                let child = (*cur_node).find_child_pyp(context, true);
                parent_p_w = p_w;
                cur_node = child.unwrap();
            }
        }
        return Some(cur_node);
    }

    fn find_node_by_tracing_back_context_path_nodes_cache(
        &mut self,
        characters: &Vec<char>,
        n: usize,
        depth_of_n: usize,
    ) -> Option<*mut PYP<char>> {
        if n < depth_of_n {
            return None;
        }
        let mut cur_node = &mut self.root as *mut PYP<char>;
        unsafe {
            for d in 0..depth_of_n {
                match self.path_nodes[d + 1] {
                    None => {
                        let context = characters[n - d - 1];
                        let child = (*cur_node).find_child_pyp(context, true);
                        cur_node = child.unwrap();
                    }
                    Some(node) => cur_node = node,
                }
            }
        }
        return Some(cur_node);
    }

    pub fn compute_p_w(&mut self, characters: &Vec<char>) -> f64 {
        return self.compute_log_p_w(characters).exp();
    }

    pub fn compute_log_p_w(&mut self, characters: &Vec<char>) -> f64 {
        let char = characters[0];
        let mut log_p_w = 0.0 as f64;

        // I'm not sure if this scenario ever happens.
        if char != BOW {
            log_p_w += self
                .root
                .compute_p_w(char, self.g_0, &mut self.d_array, &mut self.theta_array)
                .ln();
        }

        for n in 1..characters.len() {
            log_p_w += self.compute_p_w_given_h(characters, 0, n - 1).ln();
        }

        log_p_w
    }

    pub fn compute_p_w_given_h(
        &mut self,
        characters: &Vec<char>,
        context_begin: usize,
        context_end: usize,
    ) -> f64 {
        let target_char = characters[context_end + 1];
        return self.compute_p_w_given_h_with_target(
            target_char,
            characters,
            context_begin,
            context_end,
        );
    }

    pub fn compute_p_w_given_h_with_target(
        &mut self,
        target_char: char,
        characters: &Vec<char>,
        context_begin: usize,
        context_end: usize,
    ) -> f64 {
        let mut cur_node = &mut self.root as *mut PYP<char>;
        // let cur_node = &mut self.root;
        let mut parent_pass_probability = 1.0 as f64;
        let mut p = 0.0 as f64;
        let mut parent_p_w = self.g_0;
        let mut p_stop = 1.0 as f64;
        let mut depth = 0;
        let mut end_reached = false;

        unsafe {
            while p_stop > CHPYLM_EPSILON {
                if end_reached {
                    p_stop = self.beta_stop / (self.beta_pass + self.beta_stop)
                        * parent_pass_probability;
                    p += parent_p_w * p_stop;
                    parent_pass_probability *= self.beta_pass / (self.beta_pass + self.beta_stop);
                } else {
                    let p_w = (*cur_node).compute_p_w_with_parent_p_w(
                        target_char,
                        parent_p_w,
                        &mut self.d_array,
                        &mut self.theta_array,
                    );
                    p_stop = (*cur_node).stop_probability(self.beta_stop, self.beta_pass, false)
                        * parent_pass_probability;
                    p += p_w * p_stop;
                    parent_pass_probability *=
                        (*cur_node).pass_probability(self.beta_stop, self.beta_pass, false);
                    parent_p_w = p_w;

                    if depth + 1 >= context_end - context_begin + 1 {
                        end_reached = true;
                    } else {
                        let cur_context_char = characters[context_end - depth];
                        match (*cur_node).find_child_pyp(cur_context_char, false) {
                            None => end_reached = true,
                            Some(child) => cur_node = child,
                        }
                    }
                }
                depth += 1;
            }
        }
        p
    }

    pub fn sample_depth_at_index_n(&mut self, characters: &Vec<char>, n: usize) -> usize {
        // The first character is always the BOW, with depth 0.
        if n == 0 {
            return 0;
        }

        let mut sampling_table = vec![0.0; n + 1];
        let char_n = characters[n];
        // let mut sum: f64 = 0.0;
        let parent_p_w = self.g_0;
        let mut parent_pass_probability = 1.0 as f64;
        self.parent_p_w_cache[0] = parent_p_w;
        // let mut sampling_table_size = 0;
        let mut cur_node: Option<*mut PYP<char>> = Some(&mut self.root as *mut PYP<char>);

        unsafe {
            for index in 0..n + 1 {
                match cur_node {
                    None => {
                        let p_stop = self.beta_stop / (self.beta_pass + self.beta_stop)
                            * parent_pass_probability;
                        let p = parent_p_w * p_stop;
                        self.parent_p_w_cache[index + 1] = parent_p_w;
                        sampling_table[index] = p;
                        self.path_nodes[index] = None;
                        // sampling_table_size += 1;
                        // sum += p;
                        parent_pass_probability *=
                            self.beta_pass / (self.beta_pass + self.beta_stop);
                        if p_stop < CHPYLM_EPSILON {
                            break;
                        }
                    }
                    Some(node) => {
                        let p_w = (*node).compute_p_w_with_parent_p_w(
                            char_n,
                            parent_p_w,
                            &mut self.d_array,
                            &mut self.theta_array,
                        );
                        let p_stop = self.beta_stop / (self.beta_pass + self.beta_stop)
                            * parent_pass_probability;
                        let p = p_w * p_stop * parent_pass_probability;
                        let parent_p_w = p_w;
                        self.parent_p_w_cache[index + 1] = parent_p_w;
                        sampling_table[index] = p;
                        self.path_nodes[index] = cur_node;
                        // sampling_table_size += 1;
                        parent_pass_probability *=
                            self.beta_pass / (self.beta_pass + self.beta_stop);
                        // sum += p;
                        if p_stop < CHPYLM_EPSILON {
                            break;
                        }
                        if index < n {
                            let context_char = characters[n - index - 1];
                            // TODO: This is a bit awkward. Might refactor this function.
                            cur_node = match (*node).find_child_pyp(context_char, false) {
                                None => None,
                                Some(n) => Some(n as *mut PYP<char>),
                            };
                        }
                    }
                }
            }
        }
        let dist = WeightedIndex::new(&sampling_table).unwrap();
        dist.sample(&mut rand::thread_rng())
    }
}

impl HPYLM<char> for CHPYLM {
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
