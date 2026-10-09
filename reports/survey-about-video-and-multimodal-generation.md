# Video and Multimodal Generation: A Survey

## TL;DR
- Diffusion models are the dominant paradigm for controllable, high-quality video synthesis; recent reviews taxonomy well over two hundred methods across generation, editing, and understanding [1][3].
- Scaling text-conditional diffusion to serve as a general "world simulator" was crystallized by OpenAI's Sora, which trains on spacetime patches of image and video latents [2].
- A parallel line extends large language models into multimodal video understanding, with dedicated surveys tracing the progression from image to long-video MM-LLMs [4].
- The field is converging on unified multimodal generation (image and text in one model) and agent-native generation with persistent memory and skills [7].
- Recurring open problems are long-video temporal consistency, evaluation of temporal compositionality, and the computational cost of training and inference [3][6].

## Background
Video generation has become a central pillar of AI-generated content because it carries richer semantic expressiveness than still images and unlocks applications from creative media to robotics. Among generative paradigms, denoising diffusion models rose to prominence for their controllability and visual fidelity, and are now the subject of structured surveys that propose unified taxonomies of architecture, temporal dynamics, and applications [1]. A complementary review frames the same landscape around the design principles of over two hundred methods and the datasets and metrics used to benchmark them [3].

## Diffusion architectures and world simulators
Most modern video generators are latent diffusion models that add a temporal dimension to the standard image denoiser: a video is compressed into a latent space and then decomposed into tokens (or patches) before a transformer or U-Net predicts the denoised sequence. OpenAI's Sora report pushed this direction the furthest, training a text-conditional diffusion model jointly on videos and images of variable durations, resolutions, and aspect ratios, with its largest model generating up to a minute of high-fidelity video; the authors explicitly frame scaling video generation as a path toward general-purpose simulators of the physical world [2]. On the reproducible side, open text-to-video and image-to-video models such as VideoCrafter1 showed that high-quality, cinematic generation is achievable outside proprietary labs [5].

## Multimodal understanding and generation
Alongside pure generation, multimodal large language models (MM-LLMs) attach a visual encoder to an LLM backbone to reason about video. A survey on long-video understanding traces how MM-LLMs evolved from images to short and then long videos, emphasizing the design of connectors that compress visual tokens and the challenges of fine-grained spatiotemporal detail and long-term dependency [4]. The generation and understanding threads are now merging: flow-matching and unified frameworks generate both modalities in a single model, and agent-native systems such as GEMS add persistent memory, domain skills, and multi-agent optimization on top of a generator, which improves quality across general and downstream tasks [7]. Recent "3D cinemagraph" work further turns static 3D world models into dynamic, endlessly looping scenes by using a vision-language model to infer plausible motion and lifting a reference video into multi-view video [8].

## Benchmarks and evaluation
Evaluation remains a weak point in the field. Standard metrics focus on per-frame quality and struggle to capture temporal coherence, so dedicated benchmarks have emerged for the temporal dimension. TC-Bench specifically measures temporal compositionality — how well a model handles compositional changes and transitions over time — and reports significant gaps in current text-to-video and image-to-video models [6]. Goal-directed video world models add a further requirement: the same machinery used for generation must also decide the next action from the generated state and recognize task completion, which exposes failures of open-loop generation under closed-loop procedural execution [9].

## Trends and open problems
The last two years have shifted generation from single-model demos toward unified, agentic, and physically grounded systems. The unresolved problems are consistent across surveys: generating longer videos with stable motion, faithfully rendering physical dynamics, evaluating temporal consistency beyond visual quality, and reducing the very large compute budgets of these models [3][6][1]. Disputed and rapidly moving areas include whether video models can genuinely serve as world simulators for planning rather than as visually impressive but physically approximate generators [2][9].

## References
[1] Video Diffusion Models: A Survey. arxiv. https://arxiv.org/abs/2405.03150 (2024-05-06)
[2] Video generation models as world simulators (Sora). web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[3] Video diffusion generation: comprehensive review and open problems. web. https://link.springer.com/article/10.1007/s10462-025-11331-6 (2025-08-20)
[4] From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensive Long Video Understanding. arxiv. https://arxiv.org/abs/2409.18938 (2024-09-26)
[5] VideoCrafter1: Open Diffusion Models for High-Quality Video Generation. hf-search. https://huggingface.co/papers/2310.19512 (2023-10-30)
[6] TC-Bench: Benchmarking Temporal Compositionality in Text-to-Video and Image-to-Video Generation. hf-search. https://huggingface.co/papers/2406.08656 (2024-06-12)
[7] GEMS: Agent-Native Multimodal Generation with Memory and Skills. hf-search. https://huggingface.co/papers/2603.28088 (2026-03-30)
[8] OuroWorld: Bringing Any 3D World Alive as Diverse, Endlessly Looping 3D Cinemagraphs. hf-daily. https://huggingface.co/papers/2610.12461 (2026-10-08)
[9] WorldGuide: Goal-Directed Video World Model for Procedural Task Execution. hf-daily. https://huggingface.co/papers/2610.12459 (2026-10-08)
