pub mod sampling {
    pub fn greedy(logits: &[f32]) -> Option<u32> {
        debug_assert!(!logits.is_empty());
        let mut best_idx = 0_usize;
        let mut best_val = logits[0];
        for (idx, &val) in logits.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        Some(best_idx as u32)
    }
}
