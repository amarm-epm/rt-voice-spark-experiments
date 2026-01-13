# Local Realtime Voice Assistant on NVIDIA DGX Spark

## Research Report (2026)

---

## 1. Purpose of the document

This document is intended as **agent-ready research** for building a **fully local, real-time voice assistant** using an **NVIDIA DGX Spark** workstation (Grace Blackwell GB10).

Scope:

* Offline / on-device pipeline: Wake word (optional) → VAD → STT → LLM → TTS
* Only open-source / freely available models
* Practical deployment constraints on DGX Spark (OS, packaging, performance expectations)

---

## 2. DGX Spark quick facts (hardware + positioning)

DGX Spark is a compact desktop AI system built on **NVIDIA Grace Blackwell (GB10)**.

Key specs relevant for local AI:

* **Up to 1 PFLOP (FP4) AI performance**
* **128 GB coherent unified LPDDR5x system memory** (shared CPU/GPU address space)
* **20-core Arm CPU** (10 Cortex-X925 + 10 Cortex-A725)
* **4 TB NVMe M.2 storage (self-encrypting)**
* Connectivity: **10 GbE**, **ConnectX-7 Smart NIC** (high-speed), **Wi‑Fi 7**, Bluetooth 5.4
* Compact form factor: roughly 150mm × 150mm × 50.5mm

Practical implication:

* The 128 GB unified memory ceiling is a major differentiator vs typical consumer desktops, enabling **very large model inference** locally.

---

## 3. What “real-time voice assistant” means here

A real-time voice assistant is defined as a system that:

1. Accepts live microphone input
2. Detects speech boundaries (VAD)
3. Converts speech to text (STT)
4. Generates response (LLM)
5. Converts response to audio (TTS)
6. Plays back audio with optional **barge-in** (interrupt when user speaks)

Target UX latency:

* Ideal: < 500 ms end-to-end (cloud-grade)
* Realistic local goal: ~0.5–1.5 seconds end-to-end depending on model sizes

DGX Spark should reduce latency mainly by:

* Faster STT/LLM inference at a given model size
* Allowing larger batch/longer context without swapping

---

## 4. High-level architecture

Recommended event-driven pipeline:

```
Microphone
  ↓ (20–30 ms frames)
Audio Ring Buffer
  ↓
VAD (speech/silence)
  ↓
Streaming STT (partial + final)
  ↓
LLM (chat + tool calls)
  ↓
TTS (streaming or chunked)
  ↓
Playback + Barge-in
```

Engineering priorities:

* Keep audio I/O real-time (separate process/thread)
* Avoid blocking on STT/LLM/TTS
* Make STT return partial hypotheses continuously

---

## 5. STT (Speech-to-Text) choices on DGX Spark

### 5.1 Whisper ecosystem (pragmatic baseline)

Whisper-based STT remains a strong baseline for OSS voice.

Advantages:

* Mature tooling
* Large ecosystem
* Good multilingual accuracy

On DGX Spark:

* You can use GPU-accelerated inference paths (depending on the chosen runtime)
* For streaming, you still need chunking + VAD + partial transcript strategy

Best practice:

* Prefer smaller Whisper variants for very low latency, unless accuracy demands larger models

### 5.2 NVIDIA NeMo ASR (first-class on DGX Spark)

This is where DGX Spark changes the picture.

NeMo strengths for ASR:

* A large suite of speech AI models (ASR, punctuation, timestamps, diarization)
* Research and production-grade pipelines in the NVIDIA ecosystem

Why NeMo becomes relevant here:

* DGX Spark is designed around NVIDIA’s stack; NeMo workloads align with the platform
* GPU-centric ASR pipelines become practical and performant

Trade-offs:

* Heavier dependencies and operational complexity vs whisper.cpp-style deployment

Recommended approach:

* Prototype with Whisper first (fastest path)
* Run a comparative benchmark vs NeMo ASR models
* Choose based on accuracy/latency and packaging complexity

---

## 6. LLM choices on DGX Spark

### 6.1 Requirements

* Low first-token latency
* Enough memory for long context and large KV cache
* Stable local runtime
* Quantization optional (DGX Spark can handle larger models, but quant still helps)

### 6.2 Practical OSS runtimes

Choose one based on your comfort:

1. **TensorRT-LLM (NVIDIA)**

* Best for low-latency and throughput on NVIDIA platforms
* More setup complexity

2. **vLLM (OSS)**

* Strong ecosystem, good for serving
* Requires correct GPU stack integration

3. **llama.cpp**

* Extremely portable
* Useful for quick experiments and GGUF models
* May not use full GPU capabilities unless configured appropriately

Model sizing guidance (local voice assistant):

* 7B–14B: very strong for real-time dialog
* 30B–70B: can be feasible on DGX Spark but may impact latency depending on runtime

---

## 7. TTS choices on DGX Spark

### 7.1 Piper (simple OSS baseline)

* Very easy local deployment
* Fast inference
* Good enough for prototyping

### 7.2 NeMo TTS

* Potentially higher quality and more research options
* Heavier stack

Recommendation:

* Start with Piper
* Evaluate NeMo TTS only if voice quality is a core requirement

---

## 8. Wake word and interaction model

Start simple:

* Push-to-talk is the fastest path to a working system

Add later:

* Wake word detection (always-on) + false positive tuning

Barge-in:

* Highly recommended for “assistant feel”
* Requires careful echo handling (headphones or AEC)

---

## 9. Latency expectations on DGX Spark

DGX Spark should enable:

* Faster STT decoding (especially if GPU-accelerated)
* Faster LLM inference at larger sizes
* Larger context without memory pressure

A realistic target for a good local UX:

* 0.5–1.5 seconds end-to-end for typical prompts

Main latency drivers:

* STT finalization strategy (chunk size, beam)
* LLM decode speed (tokens/sec)
* TTS synthesis time (streaming vs full)

---

## 10. Is this “state of the art” for 2026 local voice?

Yes—DGX Spark is aligned with the current industry direction:

* Local / private AI nodes
* GPU-accelerated inference
* Full local pipelines without cloud

The main differentiator vs Mac-only approach:

* You can realistically adopt NVIDIA-native runtimes (NeMo, TensorRT-LLM)
* You can run larger models while maintaining interactive latency

---

## 11. Recommended next steps for an agent

Agent tasks (actionable):

1. **Environment confirmation**

   * Identify DGX Spark OS and driver stack
   * Confirm CUDA, TensorRT, container runtime availability

2. **STT benchmark**

   * Whisper baseline (streaming)
   * NeMo ASR baseline (streaming if available)
   * Compare: WER, latency, CPU/GPU utilization

3. **LLM benchmark**

   * Choose a target OSS model family (7B / 14B / 30B)
   * Compare runtimes: llama.cpp vs vLLM vs TensorRT-LLM
   * Metrics: first-token latency, tokens/sec, steady-state latency

4. **TTS benchmark**

   * Piper baseline
   * Optional NeMo TTS
   * Measure synthesis latency and perceived quality

5. **End-to-end prototype**

   * Implement push-to-talk dialog loop
   * Add barge-in
   * Measure end-to-end latency and user perceived responsiveness

---

## 12. Final conclusions

* DGX Spark is an excellent platform for a fully local voice assistant.
* Unlike Apple Silicon-only setups, DGX Spark makes **NeMo + TensorRT-LLM** a realistic and potentially best-in-class path.
* Recommended strategy: **start simple (Whisper + Piper + LLM runtime)**, then upgrade components (NeMo ASR, TensorRT-LLM) based on benchmark results.

---

