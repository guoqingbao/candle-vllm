// Original implementation:
// https://github.com/guoqingbao/vllm.rs/blob/main/src/utils/graph.rs
use crate::InputMetadata;
use candle_core::gcu_backend::ubridge::gcu_device::driv;
use candle_core::gcu_backend::ubridge::prelude::tops::error::ToResult;
use candle_core::gcu_backend::GcuDevice;
use candle_core::{DType, Device, Result, Tensor};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use tqdm::tqdm;

#[allow(dead_code)]
pub struct CudaGraph {
    cu_graph: driv::topsGraph_t,
    cu_graph_exec: driv::topsGraphExec_t,
    stream: driv::topsStream_t,
}

impl CudaGraph {
    pub fn begin_capture(
        stream: driv::topsStream_t,
        mode: driv::topsStreamCaptureMode,
    ) -> Result<()> {
        unsafe {
            driv::topsStreamBeginCapture(stream, mode)
                .to_result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("topsStreamBeginCapture failed: {e:?}"))
                })
        }
    }

    pub fn end_capture(
        stream: driv::topsStream_t,
        flags: driv::topsGraphInstantiateFlags,
    ) -> Result<CudaGraph> {
        let mut graph = MaybeUninit::uninit();
        let cu_graph = unsafe {
            driv::topsStreamEndCapture(stream, graph.as_mut_ptr())
                .to_result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("topsStreamEndCapture failed: {e:?}"))
                })?;
            graph.assume_init()
        };

        let mut graph_exec = MaybeUninit::uninit();
        let cu_graph_exec = unsafe {
            driv::topsGraphInstantiateWithFlags(
                graph_exec.as_mut_ptr(),
                cu_graph,
                flags as u32 as u64,
            )
            .to_result()
            .map_err(|e| {
                candle_core::Error::Msg(format!("topsGraphInstantiateWithFlags failed: {e:?}"))
            })?;
            graph_exec.assume_init()
        };
        Ok(CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream,
        })
    }

    pub fn capture_status(stream: driv::topsStream_t) -> Result<driv::topsStreamCaptureStatus> {
        let mut status = driv::topsStreamCaptureStatus::topsStreamCaptureStatusNone;
        unsafe {
            driv::topsStreamIsCapturing(stream, &mut status)
                .to_result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("topsStreamIsCapturing failed: {e:?}"))
                })?;
        }
        Ok(status)
    }

    pub fn launch(&self) -> Result<()> {
        unsafe {
            driv::topsGraphLaunch(self.cu_graph_exec, self.stream)
                .to_result()
                .map_err(|e| candle_core::Error::Msg(format!("topsGraphLaunch failed: {e:?}")))
        }
    }
}

pub trait CudaGraphModule {
    fn start_capture(&mut self, bs: usize) -> Result<()>;
    fn end_capture(&mut self) -> Result<()>;
    fn replay(&self, bs: usize) -> Result<()>;
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor>;
    fn report_graph_pool_usage(&self) -> Result<()>;
}

pub struct CudaGraphHandle {
    graph: Arc<CudaGraph>,
}

impl CudaGraphHandle {
    pub fn new(graph: Arc<CudaGraph>) -> Self {
        Self { graph }
    }

    pub fn replay(&self) -> Result<()> {
        self.graph
            .launch()
            .map_err(|e| candle_core::Error::Msg(format!("CUDA Graph launch failed: {:?}", e)))?;
        Ok(())
    }
}

pub struct CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
    ) -> Result<Tensor>,
{
    module: M,
    captured_graphs: BTreeMap<usize, CudaGraphHandle>,
    capturing: bool,
    current_bs: Option<usize>,
    device: Arc<GcuDevice>,
    pub pool_handle: RwLock<Option<i64>>,
    captured_bs: Vec<usize>,
}

impl<M> CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
    ) -> Result<Tensor>,
{
    pub fn new(module: M, device: Arc<GcuDevice>) -> Self {
        Self {
            module,
            captured_graphs: BTreeMap::new(),
            capturing: false,
            current_bs: None,
            device,
            pool_handle: RwLock::new(None),
            captured_bs: Vec::new(),
        }
    }

    fn sync_stream(&self) -> Result<()> {
        let stream = self
            .device
            .stream_inner()
            .expect("unable to obtain stream!");
        unsafe {
            driv::topsStreamSynchronize(stream)
                .to_result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("topsStreamSynchronize failed: {e:?}"))
                })
        }
    }

    fn create_capture_pool(&self) -> Result<driv::topsMemPool_t> {
        let mut pool: driv::topsMemPool_t = ptr::null_mut();
        unsafe {
            driv::topsDeviceGetDefaultMemPool(&mut pool, self.device.gcu_device().id as i32)
                .to_result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("topsDeviceGetDefaultMemPool failed: {e:?}"))
                })?;

            let handle = pool as *mut std::ffi::c_void as usize as i64;
            *self.pool_handle.write() = Some(handle);

            let threshold: u64 = u64::MAX;
            driv::topsMemPoolSetAttribute(
                pool,
                driv::topsMemPoolAttr::topsMemPoolAttrReleaseThreshold,
                &threshold as *const _ as _,
            )
            .to_result()
            .map_err(|e| {
                candle_core::Error::Msg(format!("topsMemPoolSetAttribute failed: {e:?}"))
            })?;
        }
        Ok(pool)
    }

    fn set_capture_mem_pool(&self) -> Result<()> {
        if self.pool_handle.read().is_some() {
            return Ok(());
        }
        let stream = self
            .device
            .stream_inner()
            .expect("unable to obtain stream!");
        unsafe {
            let status = CudaGraph::capture_status(stream)?;
            if status != driv::topsStreamCaptureStatus::topsStreamCaptureStatusActive {
                let pool = self.create_capture_pool()?;
                driv::topsDeviceSetMemPool(self.device.gcu_device().id as i32, pool)
                    .to_result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("topsDeviceSetMemPool failed: {e:?}"))
                    })?;
            }
        }

        Ok(())
    }

    /// Reads a usize attribute from the given GCU memory pool.
    fn get_mem_pool_attribute(
        pool: driv::topsMemPool_t,
        attr: driv::topsMemPoolAttr,
    ) -> Result<usize> {
        let mut value: usize = 0;
        unsafe {
            driv::topsMemPoolGetAttribute(
                pool,
                attr,
                &mut value as *mut _ as *mut std::ffi::c_void,
            )
            .to_result()
            .map_err(|e| {
                candle_core::Error::Msg(format!("topsMemPoolGetAttribute failed: {e:?}"))
            })?;
        }
        Ok(value)
    }

    /// Returns peak memory used (in bytes) from a given CUDA memory pool.
    pub fn get_peak_memory_usage(pool: driv::topsMemPool_t) -> Result<usize> {
        Self::get_mem_pool_attribute(pool, driv::topsMemPoolAttr::topsMemPoolAttrReservedMemHigh)
    }

    /// Returns current memory usage (in bytes) from a given CUDA memory pool.
    pub fn get_current_memory_usage(pool: driv::topsMemPool_t) -> Result<usize> {
        Self::get_mem_pool_attribute(pool, driv::topsMemPoolAttr::topsMemPoolAttrUsedMemCurrent)
    }

    /// Retrieves the default GCU memory pool for a device.
    pub fn get_current_mem_pool(&self) -> Result<driv::topsMemPool_t> {
        if self.pool_handle.read().is_some() {
            let pool_handle = self.pool_handle.read().unwrap();
            let pool: driv::topsMemPool_t = pool_handle as usize as driv::topsMemPool_t;
            Ok(pool)
        } else {
            candle_core::bail!("Memory pool for graph is not init!")
        }
    }
}

impl<M> CudaGraphModule for CudaGraphWrapper<M>
where
    M: for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
    ) -> Result<Tensor>,
{
    fn start_capture(&mut self, bs: usize) -> Result<()> {
        self.capturing = true;
        self.current_bs = Some(bs);
        self.sync_stream()?;
        self.set_capture_mem_pool()?;
        let stream = self
            .device
            .stream_inner()
            .expect("unable to obtain stream!");
        CudaGraph::begin_capture(
            stream,
            driv::topsStreamCaptureMode::topsStreamCaptureModeRelaxed,
        )?;
        Ok(())
    }

    fn end_capture(&mut self) -> Result<()> {
        self.capturing = false;
        let bs = self.current_bs.take().unwrap();
        let stream = self
            .device
            .stream_inner()
            .expect("unable to obtain stream!");
        let graph = CudaGraph::end_capture(
            stream,
            driv::topsGraphInstantiateFlags::topsGraphInstantiateFlagAutoFreeOnLaunch,
        )?;
        self.captured_graphs
            .insert(bs, CudaGraphHandle::new(Arc::new(graph)));

        self.captured_bs.push(bs);
        self.captured_bs.sort_unstable(); // keep it sorted for binary search
        self.sync_stream()?;
        Ok(())
    }

    fn replay(&self, bs: usize) -> Result<()> {
        if let Some(&next_bs) = self.captured_bs.iter().find(|&&x| x >= bs) {
            if let Some(graph) = self.captured_graphs.get(&next_bs) {
                self.sync_stream()?;
                graph.replay()?;
                self.sync_stream()
            } else {
                candle_core::bail!("No suitable graph is found for batch size {}!", next_bs)
            }
        } else {
            candle_core::bail!("Batch size {} is not captured in graph!", bs)
        }
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        (self.module)(input_ids, positions, kv_caches, input_metadata)
    }

    fn report_graph_pool_usage(&self) -> Result<()> {
        let pool = self.get_current_mem_pool()?;
        let peak = Self::get_peak_memory_usage(pool)?;
        let current = Self::get_current_memory_usage(pool)?;
        tracing::info!(
            "Default pool usage: {:.2} MB (current), {:.2} MB (peak)",
            current as f64 / 1e6,
            peak as f64 / 1e6
        );
        Ok(())
    }
}

pub struct GraphCaptureVars {
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub slot_mapping: Tensor,
    pub context_lens: Tensor,
    pub block_tables: Tensor,
    pub outputs: BTreeMap<usize, Tensor>,
}

pub struct GraphCapturer<M: CudaGraphModule> {
    pub model: M,
    pub graph_bs: Vec<usize>,
    pub graph_vars: Option<GraphCaptureVars>,
    pub max_num_seqs: usize,
    pub max_model_len: usize,
    pub block_size: usize,
    pub hidden_size: usize,
}

impl<M: CudaGraphModule> GraphCapturer<M> {
    pub fn new(
        model: M,
        max_num_seqs: usize,
        max_model_len: usize,
        block_size: usize,
        hidden_size: usize,
    ) -> Self {
        let graph_bs = (1..16)
            .collect::<Vec<_>>()
            .iter()
            .copied()
            .chain((16..=max_num_seqs.min(64)).step_by(16))
            .collect();
        println!("The following batches for capture: {:?}", graph_bs);

        Self {
            model,
            graph_bs,
            graph_vars: None,
            max_num_seqs,
            max_model_len,
            block_size,
            hidden_size,
        }
    }

    pub fn capture(
        &mut self,
        device: &Device,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
    ) -> Result<()> {
        let max_bs = self.graph_bs[self.graph_bs.len() - 1];
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;

        let input_ids = Tensor::zeros((max_bs,), DType::U32, device)?;
        let positions = Tensor::zeros((max_bs,), DType::I64, device)?;
        let slot_mapping = Tensor::zeros((max_bs,), DType::I64, device)?;
        let context_lens = Tensor::zeros((max_bs,), DType::U32, device)?;
        let block_tables = Tensor::zeros((max_bs, max_num_blocks), DType::U32, device)?;
        let mut outputs = BTreeMap::<usize, Tensor>::new();
        for i in tqdm(0..self.graph_bs.len()).desc(Some("Graph capturing")) {
            let bs = self.graph_bs[self.graph_bs.len() - i - 1];
            let input_ids_bs = input_ids.narrow(0, 0, bs)?;
            let positions_bs = positions.narrow(0, 0, bs)?;
            let input_metadata = InputMetadata {
                is_prefill: false,
                slot_mapping: slot_mapping.narrow(0, 0, bs)?,
                block_tables: Some(block_tables.narrow(0, 0, bs)?),
                context_lens: Some(context_lens.narrow(0, 0, bs)?),
                cu_seqlens_q: None,
                cu_seqlens_k: None,
                max_seqlen_q: 0,
                max_seqlen_k: 0,
                max_context_len: self.max_model_len,
            };
            self.model.start_capture(bs)?;
            let out =
                self.model
                    .forward(&input_ids_bs, &positions_bs, kv_caches, &input_metadata)?;
            self.model.end_capture()?;
            outputs.insert(bs, out);
        }
        let _ = self.model.report_graph_pool_usage();

        tracing::warn!("Captured batches {:?}", outputs.keys());

        self.graph_vars = Some(GraphCaptureVars {
            input_ids,
            positions,
            slot_mapping,
            context_lens,
            block_tables,
            outputs,
        });

        Ok(())
    }

    pub fn is_captured(&self, batch: usize) -> bool {
        self.graph_vars.is_some()
            && self
                .graph_vars
                .as_ref()
                .unwrap()
                .outputs
                .keys()
                .find(|&&x| x >= batch)
                .is_some()
    }

    pub fn replay(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        if input_metadata.is_prefill {
            candle_core::bail!("Graph replay is not used for prefill!")
        }
        let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;
        let input_batch = input_ids.dim(0)?;
        if let Some(graph_vars) = &self.graph_vars {
            if let Some(&batch) = graph_vars.outputs.keys().find(|&&x| x >= input_batch) {
                graph_vars.input_ids.zero_()?;
                graph_vars.input_ids.copy_(&input_ids, 0)?;
                graph_vars.positions.zero_()?;
                graph_vars.positions.copy_(&positions, 0)?;

                let s_mapping = input_metadata.slot_mapping.as_ref();
                graph_vars.slot_mapping.zero_()?;
                graph_vars.slot_mapping.copy_(&s_mapping, 0)?;

                let c_lens = input_metadata.context_lens.as_ref().unwrap();
                graph_vars.context_lens.zero_()?;
                graph_vars.context_lens.copy_(&c_lens, 0)?;

                let b_tables = input_metadata.block_tables.as_ref().unwrap();
                let padded_table = b_tables
                    .pad_with_zeros(1, 0, max_num_blocks - b_tables.dim(1)?)?
                    .contiguous()?;

                graph_vars.block_tables.zero_()?;
                graph_vars.block_tables.copy_(&padded_table, 0)?;

                let result = self.model.replay(batch);
                if result.is_err() {
                    eprintln!("Error when replaying graph {:?}", result);
                }

                graph_vars.outputs[&batch]
                    .narrow(0, 0, input_batch)?
                    .contiguous()
            } else {
                candle_core::bail!("Input batch {} is not captured!", input_batch)
            }
        } else {
            candle_core::bail!("Graph is not captured!")
        }
    }
}

unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

pub type ModelFn = dyn for<'a> Fn(
        &'a Tensor,
        &'a Tensor,
        Option<&'a Vec<(Tensor, Tensor)>>,
        &'a InputMetadata,
    ) -> Result<Tensor>
    + Send
    + Sync;

pub type CudaGraphFn = Box<
    dyn for<'a> Fn(
            &'a Tensor,
            &'a Tensor,
            Option<&'a Vec<(Tensor, Tensor)>>,
            &'a InputMetadata,
        ) -> Result<Tensor>
        + Send
        + Sync,
>;
