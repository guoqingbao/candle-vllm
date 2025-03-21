use crate::openai::distributed::{Comm, Id};
use crate::openai::sampling_params::SamplingParams;
use anyhow::Context;
use bincode;
use candle_core::{DType, Device};
use core::ffi::c_char;
use interprocess::local_socket::traits::{Listener, Stream};
use interprocess::local_socket::{GenericNamespaced, Name, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::collections::HashMap;
use std::env;
use std::io::Read;
use std::io::{BufRead, BufReader, Write};
use std::ops::Range;
use std::process::Command;
use std::str::FromStr;
use std::sync::Arc;
use std::time::SystemTime;
use tokenizers::Encoding;
use tracing::info;
pub(crate) const DAEMON_PAYLOAD: &str = "__CANDLE_VLLM_DAEMON_INTERNAL";
use lazy_static::lazy_static;
use std::sync::Mutex;

lazy_static! {
    static ref IS_DAEMON: Mutex<bool> = Mutex::new(false);
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(transparent)]
pub struct CommID(#[serde(with = "BigArray")] pub [c_char; 128]);

#[derive(Serialize, Deserialize, Debug)]
pub enum RankData {
    Init {
        id: CommID,
        rank: usize,
        device_id: usize,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskData {
    pub prompt: Encoding,
    pub request_id: String,
    pub created: SystemTime,
    pub sampling_params: SamplingParams,
    pub use_logprobs: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum MessageType {
    Start,
    Data(Vec<TaskData>),
    Close,
}

pub struct DaemonManager {
    daemon_streams: Option<Vec<LocalStream>>,
    main_stream: Option<LocalStream>,
}

impl DaemonManager {
    pub fn ipc_name() -> anyhow::Result<Name<'static>> {
        let printname = "candle_vllm_daemon.sock";
        Ok(printname.to_ns_name::<GenericNamespaced>()?)
    }

    pub fn is_daemon() -> bool {
        // std::env::var(DAEMON_PAYLOAD).is_ok()
        *IS_DAEMON.lock().unwrap()
    }

    //called by main process
    pub fn new(num_subprocess: usize) -> anyhow::Result<Self> {
        let name = DaemonManager::ipc_name()?;
        *IS_DAEMON.lock().unwrap() = false;
        let listener = ListenerOptions::new().name(name).create_sync()?;
        let mut streams = Vec::with_capacity(num_subprocess);
        for _ in 0..num_subprocess {
            let stream = listener.accept()?;
            streams.push(stream);
        }

        for stream in streams.iter_mut() {
            let mut reader = BufReader::new(stream);
            let mut message = String::new();
            reader.read_line(&mut message)?;
            if message.trim() == "ready" {
                info!("one daemon process connected!");
            }
        }

        Ok(Self {
            daemon_streams: Some(streams),
            main_stream: None,
        })
    }

    //called by daemon processes
    pub fn connect() -> anyhow::Result<Self> {
        *IS_DAEMON.lock().unwrap() = true;
        let name = DaemonManager::ipc_name()?;
        let mut stream = LocalStream::connect(name)?;
        stream.write_all(b"ready\n")?;
        Ok(Self {
            daemon_streams: None,
            main_stream: Some(stream),
        })
    }

    pub fn send_message(&mut self, message: &MessageType) -> std::io::Result<()> {
        assert!(
            !DaemonManager::is_daemon(),
            "must be called in the main process!"
        );
        assert!(self.daemon_streams.is_some(), "No daomon process found!");
        let serialized = bincode::serialize(message).expect("Serialization failed");
        let mut streams = self.daemon_streams.as_mut().unwrap();
        for mut stream in streams.iter() {
            stream.write_all(&(serialized.len() as u32).to_le_bytes())?;
            stream.write_all(&serialized)?;
        }
        Ok(())
    }

    pub fn receive_message(&mut self) -> std::io::Result<MessageType> {
        assert!(
            DaemonManager::is_daemon(),
            "must be called in the daemon processes!"
        );
        assert!(
            self.main_stream.is_some(),
            "not connected to the main process!"
        );
        let mut stream = self.main_stream.as_ref().unwrap();
        let mut length_buf = [0u8; 4];
        stream.read_exact(&mut length_buf)?;
        let length = u32::from_le_bytes(length_buf) as usize;

        let mut serialized = vec![0u8; length];
        stream.read_exact(&mut serialized)?;
        let message: MessageType =
            bincode::deserialize(&serialized).expect("Deserialization failed");
        Ok(message)
    }
    // pub fn notify_task(&mut self, message: MessageType) -> anyhow::Result<()> {
    //     assert!(!is_daemon(), "must be called in the main process!");
    //     let name = ipc_name()?;
    //     info!("prepare to send messages to {} subprocesses", num_subprocess);
    //     // let listener = ListenerOptions::new().name(name).create_sync()?;
    //     // let mut streams = Vec::new();
    //     // for _ in 0..num_subprocess {
    //     //     let conn = listener.accept()?;
    //     //     streams.push(conn);
    //     // }
    //     self.send_message(&message)?;
    //     info!("sending messages to {} subprocesses", num_subprocess);
    //     Ok(())
    // }
}

// pub fn wait_task() -> Option<MessageType> {
//     let name = ipc_name().unwrap();
//     // let conn = ListenerOptions::new().name(name).create_sync()?;
//     info!("prepare to receive messages from the main process");

//     // let mut conn = LocalSocketStream::connect(name).unwrap();
//     loop {
//         let mut stream = LocalStream::connect(name).unwrap();
//         let msg = match receive_message(&mut stream) {
//             Ok(MessageType::Start) => Some(MessageType::Start),
//             Ok(MessageType::Data(data)) => Some(MessageType::Data(data)),
//             Ok(MessageType::Close) => {
//                 info!("Received close signal");
//                 Some(MessageType::Close)
//             }
//             Err(e) => {
//                 // info!("Error receiving message: {}", e);
//                 continue;
//             }
//         };
//         info!("received messages from the main process");
//         return msg;
//     }
//     None
// }

pub fn init_subprocess(device_ids: Vec<usize>) -> anyhow::Result<(Id, usize, DaemonManager)> {
    // let name = DaemonManager::ipc_name()?;
    let mut id;
    let (local_rank, local_device_id, daemon_manager) =
        if let Ok(payload) = env::var(DAEMON_PAYLOAD) {
            let payload: RankData = serde_json::from_str(&payload)?;
            let RankData::Init {
                id: new_id,
                rank,
                device_id,
            } = payload;
            id = Id::uninit(new_id.0);

            let daemon_manager = DaemonManager::connect()?;
            // let mut stream = LocalStream::connect(name)?;
            // stream.write_all(b"ready\n")?;
            (rank, device_id, daemon_manager)
        } else {
            id = Id::new().unwrap();
            use rayon::iter::IndexedParallelIterator;
            use rayon::iter::IntoParallelRefIterator;
            use rayon::iter::ParallelIterator;

            let children: Vec<std::process::Child> = device_ids[1..]
                .par_iter()
                .enumerate()
                .map(|(rank, dev_id)| {
                    let exe_path = env::current_exe().expect("Failed to get current exe");
                    let args: Vec<String> = env::args().collect();
                    let mut cmd = Command::new(exe_path);
                    cmd.args(&args[1..]);

                    let data = RankData::Init {
                        id: CommID(*id.internal()),
                        rank: rank + 1,
                        device_id: *dev_id,
                    };

                    cmd.env(DAEMON_PAYLOAD, serde_json::to_string(&data).unwrap());
                    cmd.env("RUST_LOG", "info,warn");

                    cmd.stdout(std::process::Stdio::null());
                    cmd.stderr(std::process::Stdio::null());
                    cmd.stdin(std::process::Stdio::null());

                    cmd.spawn().expect("Failed to spawn process")
                })
                .collect();
            let daemon_manager = DaemonManager::new(device_ids.len() - 1)?;

            // When sending a message
            // daemon_manager.send_message(&MessageType::Start)?;

            // let listener = ListenerOptions::new().name(name).create_sync()?;
            // let mut ready_count = 0;

            // while ready_count < device_ids.len() - 1 {
            //     let stream = listener.accept()?;
            //     let mut reader = BufReader::new(stream);
            //     let mut message = String::new();
            //     reader.read_line(&mut message)?;
            //     if message.trim() == "ready" {
            //         ready_count += 1;
            //     }
            // }
            info!("All workers have received the ids!");

            (0, device_ids[0], daemon_manager)
        };

    // let comm = Rc::new(Comm::from_device(
    //     id,
    //     device,
    //     local_rank,
    //     device_ids.len(),
    // )?);

    Ok((id, local_rank, daemon_manager))
}
