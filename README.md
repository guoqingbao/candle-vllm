<p align="center">
    <img src="./res/candle_vllm_logo.png" alt="candle vLLM" width=55%/>
</p>

[![Continuous integration](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml)

Efficient, easy-to-use platform for inference and serving local LLMs including an OpenAI compatible API server.

## Features
- OpenAI compatible API server provided for serving LLMs.
- Highly extensible trait-based system to allow rapid implementation of new module pipelines,
- Streaming support in generation.
- Efficient management of key-value cache with PagedAttention.
- Continuous batching.
- `In-situ` quantization (and `In-situ` marlin format conversion)
- `GPTQ/Marlin` format quantization (4-bit)
- Support `Mac/Metal` devices
- Support `Multi-GPU` inference (both `multi-process` and  `multi-threaded` mode)
- Support `Multi-node` inference with MPI runner

## Develop Status

Currently, candle-vllm supports chat serving for the following models.

| Model ID | Model Type | Supported | Speed (A100, `BF16`) | Throughput (`BF16`, `bs=16`) | Quantized (A100, `Q4K` or `Marlin`) | Throughput (`GTPQ/Marlin`, `bs=16`) |
|--|--|--|--|--|--|--|
| #1 | **LLAMA** |✅|65 tks/s (LLaMa3.1 8B) | 553 tks/s (LLaMa3.1 8B) | 75 tks/s (LLaMa3.1 8B), **115 tks/s (LLaMa3.1 8B, Marlin)** |**968 tks/s (LLaMa3.1 8B)**|
| #2 | **Mistral** |✅|70 tks/s (7B)| 585 tks/s (7B) | 96 tks/s (7B), **115 tks/s (7B, Marlin)** |**981 tks/s (7B)**|
| #3 | **Phi (v1, v1.5, v2)** |✅|97 tks/s (2.7B, F32+BF16)|TBD|-|TBD|
| #4 | **Phi-3** |✅|107 tks/s (3.8B)| 744 tks/s (3.8B)|135 tks/s (3.8B)|TBD|
| #5 | **Yi** |✅|75 tks/s (6B)| 566 tks/s (6B) | 105 tks/s (6B)|TBD|
| #6 | **StableLM** |✅|99 tks/s (3B)|TBD|-|TBD|
| #7 | BigCode/StarCode |TBD|TBD|TBD |-|TBD|
| #8 | ChatGLM |TBD|TBD|TBD |-|TBD|
| #9 | **QWen2** |✅|148 tks/s (1.8B)|784 tks/s (1.8B) |-|TBD|
| #10 | **Google Gemma** |✅|130 tks/s (2B)|TBD |**73 tks/s (Gemma2-9B, Marlin)** |**587 tks/s (Gemma2-9B)**|
| #11 | **DeepSeek-R1-Distill-QWen** |TBD|TBD|TBD|**62 tks (QWen 14B)**|TBD|
| #12 | **DeepSeek-R1-Distill-LLaMa** |TBD|TBD|TBD|**108 tks (LLaMa3.1 8B)**|TBD|
| #13 | Moondream-2 (Multimodal LLM) |TBD|TBD|TBD |-|TBD|
| #14 | **DeepSeek V2/V3/R1** |✅|TBD|TBD |-|TBD|
| #15 | **QwQ-32B (GGUF)** |✅|TBD|TBD |36 tks/s (Q4K)|TBD|


## General Usage

### Build Candle-vLLM

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh #install rust, 1.83.0+ required
sudo apt install libssl-dev pkg-config -y
git clone git@github.com:EricLBuehler/candle-vllm.git
cd candle-vllm

#Make sure the CUDA Toolkit can be found in the system PATH
export PATH=$PATH:/usr/local/cuda/bin/

#single-node
cargo build --release --features cuda,nccl

#multinode
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin -y #install mpi
sudo apt install clang libclang-dev
cargo build --release --features cuda,nccl,mpi #build with mpi feature
```

### Build/Run Parameters

[`ENV_PARAM`] cargo run [`BUILD_PARAM`] -- [`PROGRAM_PARAM`] [`MODEL_ID/MODEL_WEIGHT_PATH`] [`MODEL_TYPE`] [`MODEL_PARAM`]

**Example:**
```shell
[RUST_LOG=warn] cargo run [--release --features cuda,nccl] -- [--multi-process --log --dtype bf16 --port 2000 --device-ids "0,1" --kvcache-mem-gpu 8192] [--weight-path /home/weights/QwQ32B-GPTQ-4Bit] [qwen2] [--quant gptq --temperature 0.7 --penalty 1.0 --top-k 40 --top-p 0.95]
```

`ENV_PARAM`: RUST_LOG=warn

`BUILD_PARAM`: --release --features cuda,nccl

`PROGRAM_PARAM`：--multi-process --log --dtype bf16 --port 2000 --device-ids "0,1" --kvcache-mem-gpu 8192

`MODEL_WEIGHT_PATH`: --weight-path /home/weights/QwQ32B-GPTQ-4Bit

`MODEL_TYPE`: qwen2

`MODEL_PARAM`: --quant gptq --temperature 0.7 --penalty 1.0 --top-k 40 --top-p 0.95

where, `MODEL_TYPE` in ["llama", "llama3", "mistral", "phi2", "phi3", "qwen2", "gemma", "yi", "stable-lm", "deep-seek"]

## Detailed Usage

Run **Uncompressed** models
```shell
target/release/candle-vllm --port 2000 --weight-path /home/DeepSeek-R1-Distill-Llama-8B/ llama3 --temperature 0. --penalty 1.0
```

Run **Marlin-compatible GPTQ models** (Suggested, fastest approach)
```shell
#model format (4-bit GPTQ, 128-group, desc_act=False)
target/release/candle-vllm --dtype bf16 --port 2000 --weight-path /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g qwen2 --quant gptq --temperature 0. --penalty 1.0

# If you don't have such model, you can convert Uncompressed model to Marlin-compatible format using the given script
python3 examples/convert_marlin.py --src /home/DeepSeek-R1-Distill-Qwen-14B/ --dst /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
```

Run **Marlin-compatible AWQ models**,
```shell
python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
target/release/candle-vllm --multi-process --dtype f16 --port 2000 --device-ids "0" --weight-path /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ llama3 --quant awq --temperature 0. --penalty 1.0
```

**Note:** Candle-vLLM will repack the GPTQ/AWQ model into Marlin format during model loading


Run **Marlin-format models**
```shell
# If you have Marlin-format model, run it with (--quant marlin)
target/release/candle-vllm --dtype bf16 --port 2000 --weight-path /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Marlin/ qwen2 --quant marlin --penalty 1.0 --temperature 0.
```

You may also run specific model using **Huggingface model-id**, e.g.,
```shell
target/release/candle-vllm --port 2000 --model-id meta-llama/Llama-2-7b-chat-hf llama
target/release/candle-vllm --port 2000 --model-id avoroshilov/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g --quant gptq --penalty 1.0 --temperature 0.
```

Run **QwQ-32B GGUF/GGML** models on **CUDA or Mac/Metal** devices
```shell
target/release/candle-vllm --port 2000 --model-id Qwen/QwQ-32B --dtype bf16 --weight-path ./ --weight-file qwq-32b-q4_k_m.gguf qwen2 --quant gguf --temperature 0. --penalty 1.0
```

Run **GGUF/GGML** models on **Mac/Metal** devices (assume gguf model downloaded in `/Users/Downloads`)
```shell
cargo run --release --features metal -- --port 2000 --model-id microsoft/Phi-3.5-mini-instruct --dtype bf16 --weight-path /Users/Downloads --weight-file Phi-3.5-mini-instruct-Q4_K_M.gguf phi3 --quant gguf --temperature 0. --penalty 1.0
```
**Note:** `dtype` in gguf/ggml mode is used for kv cache and attention, you may choose `f32` or `bf16`, while, `f16` is not recommended.


Run **Multi-process Multi-GPU**

```shell
target/release/candle-vllm --multi-process --port 2000 --device-ids "0,1" --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --temperature 0. --penalty 1.0
```

```shell
target/release/candle-vllm --multi-process --dtype bf16 --port 2000 --device-ids "0,1" --weight-path /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-Marlin/ llama3 --quant gptq --temperature 0. --penalty 1.0
```

Run **Multi-threaded Multi-GPU** (for debug purpose)
```shell
#simply remove the "--multi-process"
target/debug/candle-vllm --port 2000 --device-ids "0,1" --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --temperature 0. --penalty 1.0
```
**Note:** number of GPUs (`--device-ids`) used must be aligned to 2^n (e.g., 2, 4, or 8).

If you encountered problems under Multi-threaded Multi-GPU mode, you may:
```shell
export NCCL_P2P_DISABLE=1 # disable p2p cause this feature can cause illegal memory access in certain environments
```

Run **DeepSeek-R1 (671B/685B) on Lower GPU Memory Setups**, e.g., **single-node** with `8 x A100(48GB)`
```shell
python3 examples/convert_awq_marlin.py --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Marlin/ 
RUST_LOG=warn cargo run --release --features cuda,nccl -- --log --multi-process --dtype bf16 --port 2000 --device-ids "0,1,2,3,4,5,6,7" --weight-path /data/DeepSeek-R1-AWQ-Marlin/ deep-seek --quant awq --temperature 0. --penalty 1.0 --num-experts-offload-per-rank 11
```
**Note:** This setup offloads 11 experts per rank (a total of 88 out of 256 experts) to the CPU (around 125GB additional host memory required). During inference, these offloaded experts are swapped back into GPU memory as needed. If you have even less GPU memory, consider increasing the `--num-experts-offload-per-rank` parameter (up to a maximum of 32 experts per rank in this case).

Run **DeepSeek-R1 (671B/685B) on multi-node**, e.g., (`8 x A100(48GB)` x 2 nodes)
```shell
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin -y #install mpi
sudo apt install clang libclang-dev
cargo build --release --features cuda,nccl,mpi #build with mpi feature
python3 examples/convert_awq_marlin.py --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Marlin/ #convert awq deepseek to marlin-compatible format
#running multinode inference with mpi runner
sudo mpirun -np 16 -x RUST_LOG=info -hostfile ./hostfile --allow-run-as-root -bind-to none -map-by slot --mca plm_rsh_args "-p 22" --mca btl_tcp_if_include %NET_INTERFACE% target/release/candle-vllm --log --multi-process --dtype bf16 --port 2000 --device-ids "0,1,2,3,4,5,6,7" --weight-path /data/DeepSeek-R1-AWQ-Marlin/ deep-seek --quant awq --temperature 0. --penalty 1.0
```
**Note**: MPI Runner requires `identical` hardware and software configurations for all nodes, please ensure weights and candle-vllm binaries located in the identical folders in difference nodes. The the nodes need to be ssh (port 22 in this case) passwordless for each other (root user if `--allow-run-as-root`). `%NET_INTERFACE%` is the active network interface obtained through command 'ifconfig -a'. You may disable InfiniBand if it's not available in the nodes by insert env "-x NCCL_IB_DISABLE=1". Where, `hostfile` can be defined as:

Example (two nodes, each with 8 GPUs)
```
192.168.1.100 slots=8
192.168.1.101 slots=8
```

### Chat frontend (any frontend compatible with openai API, simple options available below):
#### Option 1: Chat with Chat.py (for simple tests)
Install API and chatbot dependencies (openai package is only used for local chat with candle-vllm)

```shell
python3 -m pip install openai rich click
```

Chat with the mini chatbot (plain text)
```shell
python3 examples/chat.py
```

Chat with the mini chatbot (live update with Markdown, may cause flick)
```shell
python3 examples/chat.py --live
```

Chat demo on GPU (A100, LLaMa3.1 8B)

<img src="res/LLaMa3.1-8B-Chatbot-A100.gif" width="75%" height="75%" >

Chat demo on Apple M4 (Phi3 3.8B)

<img src="res/Phi3-3.8B-Chatbot-Apple-M4.gif" width="75%" height="75%" >

#### Option 2: Chat with naive ChatUI (or popular dify frontend)
Install naive ChatUI and its dependencies:

```
git clone git@github.com:guoqingbao/candle-vllm-demo.git
cd candle-vllm-demo
apt install npm #install npm if needed
npm install n -g #update node js if needed
n stable #update node js if needed
npm i -g pnpm #install pnpm manager
pnpm install #install ChatUI dependencies
```

Launching the ChatUI:
```
pnpm run dev # run the ChatUI
```

#### Trouble shooting for Nodejs error
`ENOSPC: System limit for number of file watchers reached`
```
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
```

#### Option 3: Chat completion request with HTTP post

``` shell
curl -X POST "http://127.0.0.1:2000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -d '{
           "model": "llama7b",
           "messages": [
               {"role": "user", "content": "Explain how to best learn Rust."}
           ],
           "temperature": 0.7,
          "max_tokens": 128,
          "stop": {"Single":"</s>"}
       }'
```
Sample response:

```
{"id":"cmpl-53092967-c9cf-40e0-ae26-d7ac786d59e8","choices":[{"message":{"content":" Learning any programming language requires a combination of theory, practice, and dedication. Here are some steps and resources to help you learn Rust effectively:\n\n1. Start with the basics:\n\t* Understand the syntax and basic structure of Rust programs.\n\t* Learn about variables, data types, loops, and control structures.\n\t* Familiarize yourself with Rust's ownership system and borrowing mechanism.\n2. Read the Rust book:\n\t* The Rust book is an official resource that provides a comprehensive introduction to the language.\n\t* It covers topics such","role":"[INST]"},"finish_reason":"length","index":0,"logprobs":null}],"created":1718784498,"model":"llama7b","object":"chat.completion","usage":{"completion_tokens":129,"prompt_tokens":29,"total_tokens":158}}
```

#### Option 4: Chat completion with with openai package

In your terminal, install the `openai` Python package by running `pip install openai`. I use version `1.3.5`.

Then, create a new Python file and write the following code:
```python
import openai

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:2000/v1/"

completion = openai.chat.completions.create(
    model="llama",
    messages=[
        {
            "role": "user",
            "content": "Explain how to best learn Rust.",
        },
    ],
    max_tokens = 64,
)
print(completion.choices[0].message.content)
```
After the `candle-vllm` service is running, run the Python script and enjoy efficient inference with an OpenAI compatible API server!


## Batched requests

Install openai API first
```
python3 -m pip install openai
```

Run the benchmark test
``` shell
python3 examples/benchmark.py --batch 16 --max_tokens 1024
```
Refer to `examples/benchmark.py`

``` python
async def benchmark():
    model = "mistral7b"
    max_tokens = 1024
    # 16 requests
    prompts = ["Explain how to best learn Rust.", 
               "Please talk about deep learning in 100 words.", 
               "Do you know the capital city of China? Talk the details of you known.", 
               "Who is the best female actor in the world? Explain why.",
               "How to dealing with depression?",
               "How to make money in short time?",
               "What is the future trend of large language model?",
               "The famous tech companies in the world.",
               "Explain how to best learn Rust.", 
               "Please talk about deep learning in 100 words.", 
               "Do you know the capital city of China? Talk the details of you known.", 
               "Who is the best female actor in the world? Explain why.",
               "How to dealing with depression?",
               "How to make money in short time?",
               "What is the future trend of large language model?",
               "The famous tech companies in the world."]
    
    # send 16 chat requests at the same time
    tasks: List[asyncio.Task] = []
    for i in range(len(prompts)):
        tasks.append(
            asyncio.create_task(
                chat_completion(model, max_tokens, prompts[i]))
        )

    # obtain the corresponding stream object for each request
    outputs: List[Stream[ChatCompletionChunk]] = await asyncio.gather(*tasks)

    # tasks for streaming chat responses
    tasks_stream: List[asyncio.Task] = []
    for i in range(len(outputs)):
        tasks_stream.append(
            asyncio.create_task(
                stream_response(i, outputs[i]))
        )

    # gathering the response texts
    outputs: List[(int, str)] = await asyncio.gather(*tasks_stream)

    # print the results, you may find chat completion statistics in the backend server (i.e., candle-vllm)
    for idx, output in outputs:
        print("\n\n Response {}: \n\n {}".format(idx, output))


asyncio.run(benchmark())
```

## GPTQ/AWQ/Marlin 4-bit quantization
Candle-vllm now supports GPTQ/AWQ (Marlin kernel), you may supply the `quant` (marlin) parameter if you have `Marlin` format quantized weights, such as:

```shell
cargo run --release --features cuda -- --port 2000 --dtype f16 --weight-path /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-Marlin/ llama3 --quant marlin --temperature 0. --penalty 1.
```
or, convert existing AWQ 4bit model to marlin compatible format
```shell
python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
cargo run --release --features cuda,nccl -- --multi-process --dtype f16 --port 2000 --device-ids "0" --weight-path /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ llama3 --quant awq --temperature 0. --penalty 1.0
```

You may also use `GPTQModel` to transform a model to marlin-compatible format using the given script `examples/convert_marlin.py`. 

**Note:** for using Marlin fast kernel, only 4-bit GPTQ quantization supported at the moment. 

## In-situ quantization (or in-situ marlin conversion)

Candle-vllm now supports in-situ quantization, allowing the transformation of default weights (F32/F16/BF16) or `4-bit GPTQ/AWQ` weights into any GGML format (or `marlin format`) during model loading. This feature helps conserve GPU memory (or speedup inference performance through marlin kernel), making it more efficient for consumer-grade GPUs (e.g., RTX 4090). To use this feature, simply supply the quant parameter when running candle-vllm.

For unquantized models:

```
cargo run --release --features cuda -- --port 2000 --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --quant q4k
```

For quantized 4-bit GPTQ model:

```
cargo run --release --features cuda -- --port 2000 --weight-path /home/mistral_7b-int4/ mistral --quant marlin
```

Options for `quant` parameters: ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k", "marlin", "gguf", "ggml", "gptq", "awq"]

**Please note**:

1) It may takes few minutes to load F32/F16/BF16 models into quantized;

2) Marlin format in-situ conversion only support 4-bit GPTQ (with `sym=True`, `groupsize=128` or -1, `desc_act=False`) and 4-bit AWQ (after conversion using the given script);

3) Marlin format only supported in CUDA platform.

## Usage Help
For kvcache configuration, set `kvcache_mem_cpu` and `kvcache_mem_gpu`, default 4GB CPU memory and 4GB GPU memory for kvcache. 

For chat history settings, set `record_conversation` to `true` to let candle-vllm remember chat history. By `default`, candle-vllm `does not` record chat history; instead, the client sends both the messages and the contextual history to candle-vllm. If record_conversation is set to `true`, the client sends only new chat messages to candle-vllm, and candle-vllm is responsible for recording the previous chat messages. However, this approach requires per-session chat recording, which is not yet implemented, so the default approach `record_conversation=false` is recommended.

For chat streaming, the `stream` flag in chat request need to be set to `True`.

You may supply `penalty` and `temperature` to the model to **prevent potential repetitions**, for example:

```
cargo run --release --features cuda -- --port 2000 --weight-path /home/mistral_7b/ mistral --repeat-last-n 64 --penalty 1.1 --temperature 0.7
```

`--max-gen-tokens` parameter is used to control the maximum output tokens per chat response. The value will be set to 1/5 of max_sequence_len by default.

For `consumer GPUs`, it is suggested to run the models under GGML formats (or Marlin format), e.g.,

```
cargo run --release --features cuda -- --port 2000 --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --quant q4k
```

where `quant` is one of ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k", "awq", "gptq", "marlin", "gguf", "ggml"].

## Report issue
Installing `candle-vllm` is as simple as the following steps. If you have any problems, please create an
[issue](https://github.com/EricLBuehler/candle-lora/issues).


## Contributing
The following features are planned to be implemented, but contributions are especially welcome:
- Sampling methods:
  - Beam search ([huggingface/candle#1319](https://github.com/huggingface/candle/issues/1319))
- More pipelines (from `candle-transformers`)

## Resources
- Python implementation: [`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm` paper](https://arxiv.org/abs/2309.06180)
