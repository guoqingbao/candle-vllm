#[cfg(feature = "gcu")]
use crate::backend::custom_ops::moe::{TopKLastDimOp, TopKOutput};
use crate::openai::distributed::shard;
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, VarBuilder};
use crate::openai::models::linear::Linear;
use crate::openai::models::{Config, MoEConfig};
use attention_rs::moe;
use candle::{DType, Module, Result, Tensor, D};
use candle_core as candle;
use candle_nn::var_builder::Shard;
use std::rc::Rc;

pub struct FusedMoe {
    gate: Linear,
    gate_experts: Tensor,
    up_experts: Tensor,
    down_experts: Tensor,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
}

impl FusedMoe {
    pub fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>, _dtype: DType) -> Result<Self> {
        let moe_cfg = if let Some(MoEConfig::QwenMoE(moe_cfg)) = &cfg.moe_config {
            moe_cfg.clone()
        } else {
            candle::bail!("Expected QwenMoEConfig")
        };

        let num_experts = moe_cfg.num_experts.unwrap();
        let ws = vb.pp("gate").get_with_hints_dtype(
            (num_experts, cfg.hidden_size),
            "weight",
            Shard::default(),
            DType::F32,
        )?;
        let gate = Linear::new(ws, None);

        let experts_vb = vb.pp("experts");
        let mut gate_experts = Vec::with_capacity(num_experts);
        let mut up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        for i in 0..num_experts {
            let experts_vb = experts_vb.pp(format!("{}", i).as_str());
            // n x k format
            let gate_expert = experts_vb.pp("gate_proj").get_with_hints(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                "weight",
                shard(0, comm.rank(), comm.world_size()),
            )?;
            let up_expert = experts_vb.pp("up_proj").get_with_hints(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                "weight",
                shard(0, comm.rank(), comm.world_size()),
            )?;
            let down_expert = experts_vb.pp("down_proj").get_with_hints(
                (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                "weight",
                shard(1, comm.rank(), comm.world_size()),
            )?;
            gate_experts.push(gate_expert);
            up_experts.push(up_expert);
            down_experts.push(down_expert);
        }

        let gate_experts = Tensor::stack(&gate_experts, 0)?;
        let up_experts = Tensor::stack(&up_experts, 0)?;
        let down_experts = Tensor::stack(&down_experts, 0)?;
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
        })
    }

    pub fn forward(&self, xs: &Tensor, _: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let router_logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let TopKOutput {
            values: mut scores,
            indices,
        } = routing_weights.topk(self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }

        let ys = {
            let xs = xs.reshape((num_tokens, 1, hidden_dim))?;
            let gate = moe::moe_gemm(&xs, &self.gate_experts, &indices)?;
            let up = moe::moe_gemm(&xs, &self.up_experts, &indices)?;
            let down_inputs = (up * gate.apply(&self.act)?)?;
            moe::moe_gemm(&down_inputs, &self.down_experts, &indices)?.to_dtype(DType::F32)?
        };

        let mut ys = ys
            .broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .t()?
            .contiguous()?
            .sum(D::Minus1)?
            .reshape((num_tokens, hidden_dim))?
            .to_dtype(original_dtype)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}
