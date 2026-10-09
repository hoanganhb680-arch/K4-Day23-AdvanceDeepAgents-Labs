# Efficient Inference and Small Language Models: A Survey

## TL;DR
- Large language models are inefficient to serve because of model size, quadratic-complexity attention, and autoregressive decoding; efficient-inference surveys organize fixes into data-, model-, and system-level optimization [1].
- Small language models trade scale for practicality, targeting on-device, mobile, and edge deployment through architecture, training, and compression techniques [2].
- Post-training quantization is the most mature compression lever, with integer-only and scale-reparameterization methods recovering near-floating-point accuracy at low bit widths [3][5][6].
- Mixed-precision and hardware-aware formats are the active frontier for pushing LLMs to 4-bit and lower while guarding against outlier-induced degradation [7][9].
- Open problems include balancing accuracy against aggressive compression, and quantization-aware fine-tuning that merges compression with task adaptation [6][8].

## Background
The compute and memory cost of large language models remains the main barrier to deploying them in resource-constrained environments. Efficient-inference research identifies three root causes — large model size, quadratic attention cost, and autoregressive decoding — and builds a taxonomy spanning data-level, model-level, and system-level optimizations [1]. In parallel, the small-language-model (SLM) literature reframes the same goal as training models that are useful against hardware, data, bandwidth, or latency constraints, and surveys its own benchmark sets and metrics [2].

## Quantization and model compression
Quantization reduces model size and memory movement by representing weights and activations in low-bit integers. Integer-only ViT inference (I-ViT) showed that keeping the entire inference datapath in integers avoids floating-point overhead while remaining accurate [3]. LUT-GEMM replaced expensive quantized matrix multiplications with lookup tables to speed up token generation in generative language models [4]. Because quantization can distort weight distributions, reparameterization methods such as RepQ-ViT and RepQuant decouple quantization and inference and re-learn per-layer scales, substantially recovering accuracy for vision transformers and large transformers [5][6].

## Mixed-precision and fine-tuning
Uniform low-bit quantization struggles with outlier activations, so the newest work uses sensitivity-aware and mixed-precision schemes. MXSens analyzes that outliers range from rare extremes to frequent mild deviations and proposes sensitivity-aware mixed-precision quantization for LLM inference [7]. For mixture-of-experts models on the edge, DyMoE adapts precision per expert using expert-importance skew and depth sensitivity [9]. On the training side, quantization-augmented fine-tuning such as QEFT integrates compression directly into parameter-efficient fine-tuning to preserve quality after compression [8].

## Trends and open problems
The field is moving from static, uniform post-training quantization toward dynamic, sensitivity-aware, hardware-native approaches, and from compressing-the-model toward co-designing accuracy, memory, and latency. The outstanding challenges are handling outliers at very low bit widths without frequent dequantization, extending mixed-precision to sparse and mixture-of-experts architectures, and integrating compression with adaptation so fine-tuning does not undo quantization gains [7][9][6][8]. A central, still-unsettled question is how far "small" can be pushed before reasoning quality collapses, with the definition of small itself shifting over time as model families grow [2][1].

## References
[1] A Survey on Efficient Inference for Large Language Models. arxiv. https://arxiv.org/abs/2404.14294 (2024-04-22)
[2] A Survey on Small Language Models. web. https://aclanthology.org/2025.ranlp-1.93.pdf (2025)
[3] I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference. hf-search. https://huggingface.co/papers/2207.01405 (2022-07-04)
[4] LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models. hf-search. https://huggingface.co/papers/2206.09557 (2022-06-20)
[5] RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers. hf-search. https://huggingface.co/papers/2212.08254 (2022-12-16)
[6] RepQuant: Towards Accurate Post-Training Quantization of Large Transformer Models via Scale Reparameterization. hf-search. https://huggingface.co/papers/2402.05628 (2024-02-08)
[7] MXSens: Sensitivity-Aware Mixed-Precision Quantization for Efficient LLM Inference. hf-search. https://huggingface.co/papers/2607.17733 (2026-07-20)
[8] QEFT: Quantization for Efficient Fine-Tuning of LLMs. hf-search. https://huggingface.co/papers/2410.08661 (2024-10-11)
[9] DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge. hf-search. https://huggingface.co/papers/2603.19172 (2026-03-19)
