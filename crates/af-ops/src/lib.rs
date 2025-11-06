pub mod sampling {
    use core::f32;

    use rand::distributions::{Distribution, WeightedIndex};

    /// Always choose the most likely next token
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

    pub fn sample_temperature(logits: &[f32], temperature: f32) -> Option<u32> {
        if logits.is_empty() {
            return None;
        }
        if temperature < 1e-6 {
            return greedy(logits);
        }

        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();

        if !sum_exp.is_finite() || sum_exp < 1e-10 {
            return greedy(logits);
        }

        let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();
        let mut rng = rand::thread_rng();
        let dist = WeightedIndex::new(&probs).ok()?;
        let token = dist.sample(&mut rng);

        Some(token as u32)
    }

    pub fn sample_top_p(logits: &[f32], p: f32) -> Option<u32> {
        if logits.is_empty() {
            return None;
        }

        if p <= 0.0 || p > 1.0 {
            return None;
        }

        if p == 1.0 {
            return sample_temperature(logits, 1.0);
        }

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        if !sum_exp.is_finite() || sum_exp < 1e-10 {
            return greedy(logits);
        }
        let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();

        let mut indexed_probs: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(idx, &prob)| (idx, prob))
            .collect();

        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut nucleus_size = 0;

        for (_, prob) in indexed_probs.iter() {
            cumsum += prob;
            nucleus_size += 1;
            if cumsum >= p {
                break;
            }
        }

        let nucleus = &indexed_probs[..nucleus_size];

        let nucleus_probs: Vec<f32> = nucleus.iter().map(|(_, prob)| *prob).collect();
        let nucleus_sum: f32 = nucleus_probs.iter().sum();
        let normalized_probs: Vec<f32> = nucleus_probs.iter().map(|p| p / nucleus_sum).collect();

        let mut rng = rand::thread_rng();
        let dist = WeightedIndex::new(&normalized_probs).ok()?;
        let sampled_idx = dist.sample(&mut rng);

        Some(nucleus[sampled_idx].0 as u32)
    }

    pub fn sample_top_k(logits: &[f32], k: usize) -> Option<u32> {
        if logits.is_empty() || k == 0 {
            return None;
        }

        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &logit)| (idx, logit))
            .collect();

        indexed_logits.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_k = &indexed_logits[..k];
        let max_logit = top_k
            .iter()
            .map(|(_, logit)| *logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = top_k
            .iter()
            .map(|(_, logit)| (logit - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_vals.iter().sum();

        if !sum_exp.is_finite() || sum_exp < 1e-10 {
            return Some(top_k[0].0 as u32); // 回退到最大值
        }

        let probs: Vec<f32> = exp_vals.iter().map(|x| x / sum_exp).collect();
        let mut rng = rand::thread_rng();
        let dist = WeightedIndex::new(&probs).ok()?;
        let sampled_idx = dist.sample(&mut rng);

        Some(top_k[sampled_idx].0 as u32)
    }
}
