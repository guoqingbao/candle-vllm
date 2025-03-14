#[cfg(feature = "cuda")]
use crate::backend::custom_ops::sort::ArgSortOp; //Use our custom sort kernel, fix kernel crash on A100
use crate::candle::D;
use crate::candle::{DType, Error, Result, Tensor};
use rand::{distributions::Distribution, SeedableRng};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::sync::Arc;
use std::sync::Mutex;
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: Arc<Mutex<rand::rngs::StdRng>>,
    pub sampling: Sampling,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng: Arc::new(Mutex::new(rng)),
            sampling,
        }
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&self, logits: &Tensor) -> Result<Vec<u32>> {
        let next_tokens = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        Ok(next_tokens)
    }

    fn sample_multinomial(&self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let mut rng = self.rng.lock().unwrap();
        let next_token = distr.sample(&mut *rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&self, logits: &Tensor, top_p: f32) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let asort = logits.arg_sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let asort = logits.to_device(&candle_core::Device::Cpu)?.arg_sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = logits.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let indices: Vec<u32> = asort[b].to_vec();
                let mut prs: Vec<f32> = sorted[b].to_vec();
                // Clamp smaller probabilities to zero.
                let mut cumsum = 0.;
                for index in &indices {
                    if cumsum >= top_p {
                        prs[*index as usize] = 0.0;
                    } else {
                        cumsum += prs[*index as usize];
                    }
                }
                // Sample with clamped probabilities.
                self.sample_multinomial(&prs).unwrap()
            })
            .collect();
        Ok(vec_ret)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&self, logits: &Tensor, top_k: usize) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let (sorted, asort) = logits.sort(false)?;
        #[cfg(feature = "gcu")]
        let (sorted, asort) = candle_nn::ops::topk(logits, top_k)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_par_iter()
            .map(|b| {
                #[cfg(feature = "gcu")]
                let indices: Vec<u32> = asort[b].to_vec();
                #[cfg(feature = "gcu")]
                let prs: Vec<f32> = sorted[b].to_vec();
                #[cfg(not(feature = "gcu"))]
                let indices: Vec<u32> = asort[b][0..top_k].to_vec();
                #[cfg(not(feature = "gcu"))]
                let prs: Vec<f32> = sorted[b][0..top_k].to_vec();
                let index = self.sample_multinomial(&prs).unwrap();
                indices[index as usize] as u32
            })
            .collect();
        Ok(vec_ret)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(&self, logits: &Tensor, top_k: usize, top_p: f32) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let (sorted, asort) = logits.sort(false)?;
        #[cfg(feature = "gcu")]
        let (sorted, asort) = candle_nn::ops::topk(logits, top_k)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_par_iter()
            .map(|b| {
                #[cfg(feature = "gcu")]
                let indices: Vec<u32> = asort[b].to_vec();
                #[cfg(feature = "gcu")]
                let mut prs: Vec<f32> = sorted[b].to_vec();
                #[cfg(not(feature = "gcu"))]
                let indices: Vec<u32> = asort[b][0..top_k].to_vec();
                #[cfg(not(feature = "gcu"))]
                let mut prs: Vec<f32> = sorted[b][0..top_k].to_vec();
                let sum_p = prs.iter().sum::<f32>();
                let index = if top_p <= 0.0 || top_p >= sum_p {
                    self.sample_multinomial(&prs).unwrap()
                } else {
                    let mut cumsum = 0.;
                    for i in 0..prs.len() {
                        if cumsum >= top_p {
                            prs[i] = 0.0;
                        } else {
                            cumsum += prs[i];
                        }
                    }
                    // Sample with clamped probabilities.
                    self.sample_multinomial(&prs).unwrap()
                };
                indices[index as usize] as u32
            })
            .collect();
        Ok(vec_ret)
    }

    pub fn sample(&self, logits: &Tensor) -> Result<Vec<u32>> {
        let logits = logits.to_dtype(DType::F32)?;
        let batch = logits.layout().dims()[0];
        let prs = |temperature: f64| -> Result<Tensor> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            Ok(prs)
        };

        let next_tokens = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(&logits)?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?.to_vec2()?;
                (0..batch)
                    .into_iter()
                    .map(|b| self.sample_multinomial(&prs[b]).unwrap())
                    .collect()
            }
            Sampling::TopP { p, temperature } => {
                let prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    let prs = prs.to_vec2()?;
                    (0..batch)
                        .into_iter()
                        .map(|b| self.sample_multinomial(&prs[b]).unwrap())
                        .collect()
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let prs = prs(*temperature)?;
                self.sample_topk(&prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let prs = prs(*temperature)?;
                self.sample_topk_topp(&prs, *k, *p as f32)?
            }
        };
        Ok(next_tokens)
    }

    pub fn apply_batch_repeat_penalty(
        &self,
        logits: &Tensor,
        penalties: Vec<f32>,
        context: Vec<Vec<u32>>,
    ) -> Result<Tensor> {
        let device = logits.device();
        let batch = logits.layout().dims()[0];
        let logits_len = logits.layout().dims()[1];
        let logits: Vec<Vec<f32>> = logits.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
        let vec_ret: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let mut logits = logits[b].to_vec();
                let mut already_seen = std::collections::HashSet::new();
                if penalties[b] != 1.0 && penalties[b] != 0. && context[b].len() > 1 {
                    for token_id in &context[b] {
                        if already_seen.contains(&token_id) {
                            continue;
                        }
                        already_seen.insert(token_id);
                        if let Some(logit) = logits.get_mut(*token_id as usize) {
                            if *logit >= 0. {
                                *logit /= penalties[b]
                            } else {
                                *logit *= penalties[b]
                            }
                        }
                    }
                }
                logits
            })
            .collect();

        let logits = vec_ret.into_iter().flatten().collect();
        Tensor::from_vec(logits, (batch, logits_len), device)
    }
}
