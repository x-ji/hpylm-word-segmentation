use std::collections::HashMap;

pub struct PYP<T> {
  children: HashMap<T, PYP<T>>,
  // https://stackoverflow.com/questions/36167160/how-do-i-express-mutually-recursive-data-structures-in-safe-rust
  parent: Option<*const PYP<T>>,
  tablegroups: HashMap<T, Vec<usize>>,
  ntables: usize,
  ncustomers: usize,
  stop_count: usize,
  pass_count: usize,
  depth: usize,
  context: T,
}

impl<T> PYP<T>
where
  T: std::cmp::Eq,
  T: std::hash::Hash,
  T: Copy,
{
  fn new(context: T) -> Self {
    Self {
      children: HashMap::new(),
      parent: None,
      tablegroups: HashMap::new(),
      ntables: 0,
      ncustomers: 0,
      stop_count: 0,
      pass_count: 0,
      depth: std::usize::MAX,
      context: context,
    }
  }

  fn need_to_remove_from_parent(&self) -> bool {
    if self.parent == None {
      return false;
    } else if self.children.is_empty() && self.tablegroups.is_empty() {
      return true;
    } else {
      return false;
    }
  }

  fn get_num_tables_serving_dish(&self, dish: T) -> usize {
    let tablegroup = self.tablegroups.get(&dish);
    match tablegroup {
      None => 0,
      Some(t) => t.into_iter().sum(),
    }
  }

  fn find_child_pyp(&mut self, dish: T, generate_if_not_found: bool) -> Option<&PYP<T>> {
    if self.children.contains_key(&dish) {
      return self.children.get(&dish);
    }

    if !generate_if_not_found {
      return None;
    }

    let mut child = PYP::new(dish);
    child.parent = Some(self);
    child.depth = self.depth + 1;
    self.children.insert(dish, child);
    // Fucking hell this actually worked!
    return self.children.get(&dish);
  }
}
