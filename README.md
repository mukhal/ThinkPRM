# Process Reward Models That Think 🧠

<div align="center">
  <a href="https://arxiv.org/abs/2504.16828">
    <img src="https://img.shields.io/badge/Paper-arXiv-red">
  </a>
  <a href="https://mukhal.github.io/projects/thinkprm/">
    <img src="https://img.shields.io/badge/Webpage-ThinkPRM-orange">
  </a>
  <a href="https://huggingface.co/datasets/launch/thinkprm-1K-verification-cots">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-green">
  </a>
  <a href="https://huggingface.co/collections/launch/thinkprm-681d7a7c5e8bfcfb5fc6d7bb">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collections-blue">
  </a>
</div>



<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a>
    <a href="#data-collection" style="text-decoration: none; font-weight: bold;">📄 Data Collection</a>

  </p>
  <p>
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div>

</div>

# 🎉News
- **[2025-07-3]** Gave a talk at Tsinghua university on ThinkPRM. Slides are [here](https://docs.google.com/presentation/d/1O7HIoQLFn6ACJgAwD-e_O8c643fQUSRqbjJjO4S4s2I/edit?usp=sharing). 
- **[2025-07-2]** We have released code for running test-time scaling experiments with ThinkPRM!.
- **[2025-04-23]** Our [paper](https://arxiv.org/abs/2504.16828) is released on arxiv.
- **[2025-04-24]** Our synthetic verification CoTs used to train ThinkPRM on are now on [huggingface](https://huggingface.co/datasets/launch/thinkprm-1K-verification-cots). 
- **[2025-04-25]** Our trained PRMs are released in two sizes: [1.5B](https://huggingface.co/launch/ThinkPRM-1.5B) and [14B](https://huggingface.co/launch/ThinkPRM-14B) finetuned from R1-Distill-Qwen models.
- **[2025-05-16]** To provide a balanced performance between 1.5B and 14B, we have upload [ThinkPRM-7B](https://huggingface.co/launch/ThinkPRM-7B), based on Deepseek-R1-Distill-Qwen-7B and finetuned on our data. 

# 📖Introduction

We introduce ThinkPRM, a collection of generative long CoT process reward models. Our verifiers are obtained by finetuning reasoning models over 1K synthetic verification CoTs---filtered based on only on 8K process labels from PRM800K. The resulting verifiers outperform LLM-as-a-judge, discriminative PRMs, on most in- and out-of-domain setups. ThinkPRM enables scaling up verifier  compute either in parallel or sequentially by thinking longer.

<div align="center">
<img src="https://github.com/user-attachments/assets/4fc1a558-4005-4f2f-8b10-0c0b4616592f" alt="image" width="600"/>
</div>

# 📑 Data Collection
ThinkPRM was trained on synthetic verification CoTs. This dataset contains 1,000 high-quality synthetic verification chains-of-thought (CoTs) designed for training generative Process Reward Models (PRMs), as used in the paper ["Process Reward Models that Think"](https://arxiv.org/abs/2504.16828). The goal was to create a data-efficient alternative to traditional PRM training which often requires extensive human annotation or expensive rollouts.

Each instance consists of a math problem, a corresponding multi-step solution prefix (sourced from PRM800K [Lightman et al., 2023]), and a detailed verification CoT generated by the [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview). The verification CoT critiques each step of the solution prefix and provides a step-level correctness judgment (`\boxed{correct}` or `\boxed{incorrect}`).

To ensure high-quality synthetic CoTs, only chains where all step-level judgments matched the ground-truth human annotations from the PRM800K dataset were retained. They were also filtered based on correct formatting and length constraints to avoid issues like excessive overthinking observed in unfiltered generation. The figure below summarizes the synthetic cots collection. Refer to our paper for more details on data collection. 


![image.png](https://cdn-uploads.huggingface.co/production/uploads/5f350fe67e5835433862161b/OBLqBFn2zJfKIvnEAK2D_.png)


The dataset was created to enable efficient training of powerful generative PRMs. The core idea is that fine-tuning strong reasoning models on carefully curated, synthetic verification CoTs can yield verifiers that outperform models trained on much larger, traditionally labeled datasets. The process-based filtering (matching gold step labels) was shown to be crucial for generating high-quality training data compared to outcome-based filtering.


# ✨Getting Started

## Installation
  We are going to create two virtual envrionments: one particularly to serve ThinkPRM using [sglang](https://github.com/sgl-project/sglang) and the other for running test-time scaling experiments. This is ncecessary to avoid dependency conflicts between sglang and other packages. We will be using uv so make sure you have it installed by following the [instructions](https://docs.astral.sh/uv/getting-started/installation).

```
uv python install 3.11
uv venv sglang-env
uv pip install "sglang[all]>=0.4.8.post1"
```

now let's setup the other main environment
```bash
uv venv
uv sync
```
   

## Running ThinkPRM: Use the provided recipes to run ThinkPRM with different models and configurations:

First of all we need to serve one of the ThinkPRM [models](https://huggingface.co/collections/launch/thinkprm-681d7a7c5e8bfcfb5fc6d7bb).

The available models are:
   - [launch/ThinkPRM-1.5B](https://huggingface.co/launch/ThinkPRM-1.5B)
   - [launch/ThinkPRM-7B](https://huggingface.co/launch/ThinkPRM-7B) 
   - [launch/ThinkPRM-14B](https://huggingface.co/launch/ThinkPRM-14B)


```bash
source sglang-env/bin/activate
uv run --active python -m sglang.launch_server --grammar-backend xgrammar --model-path launch/ThinkPRM-1.5B --port 31111 --host 127.0.0.1
```
This should start an Sglang server with ThinkPRM-1.5B ready to roll. 


Now we can run best-of-N or verifier-guided beam search as follows. Note that every experiment in the paper is described using a .yaml recipe file. These are located in [search-and-learn/recipes/](search-and-learn/recipes/)

Now, switch to a new terminal and cd into the project directory. To run best-of-N on MATH-500 with Llama-3.2-3B-Instruct as the generator, this correponds to the recipe in [search-and-learn/recipes/MATH-500/Llama-3.2-3B-Instruct/best_of_n/thinkprm.yaml](search-and-learn/recipes/MATH-500/Llama-3.2-3B-Instruct/best_of_n/thinkprm.yaml)


```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 uv run python search-and-learn/scripts/test_time_compute.py search-and-learn/recipes/MATH-500/Llama-3.2-3B-Instruct/best_of_n/thinkprm.yam
```
This should sample solutions from the generator model over MATH-500 dataset, score samples using ThinkPRM, and then at the end it will also calculate accuracy for different N values and store these in config.output_dir. 

### Parallel and Sequential Scaling with ThinkPRM 


<div align="center">
<img src="https://github.com/user-attachments/assets/7a7e88ac-0bd2-494d-8b54-00fb63054ee7" alt="image" width="800"/>
</div>

ThinkPRM supports two types of scaling:
- **Parallel Scaling**: Multiple PRM evaluations run simultaneously
- **Sequential Scaling**: Multiple thinking rounds in sequence

## Configuration Differences

### Parallel Scaling
```yaml
prm_temperature: 0.4    
prm_n: 4               # 4 parallel verification CoTs
```

### Sequential Scaling
```yaml
prm_temperature: 0.0    
prm_n: 1               # only supports a single chain
n_thinking_rounds: 1   # Number of sequential thinking rounds
```

**note**: we do not support both types at the same time. 
**note**: sequential scaling is substantially slower than parallel scaling.  

## Running the Experiments
As described in the paper, we scale verifier verifier compute with ThinkPRM in two ways: parallel by sampling K independent verification chains and aggregating scores over them, or sequentially via some form of budget forcing. We include two example recipes [here](search-and-learn/recipes/MATH-500/Llama-3.2-3B-Instruct/beam_search/thinkprm_parallel_scaling.yaml) and [here](search-and-learn/recipes/MATH-500/Llama-3.2-3B-Instruct/beam_search/thinkprm_sequential_scaling.yaml)


# Acknowledgements
This project would not be possible without the efforts from the open-source community, artpicularly repos like [search-and-learn](https://github.com/huggingface/search-and-learn), [vllm](https://github.com/vllm-project/vllm/) and the awesome [sglang](https://github.com/sgl-project/sglang).

# 🎈Citation
If you find ThinkPRM helpful, please cite us.

```bibtex
@article{khalifa2025,
      title={Process Reward Models That Think}, 
      author={Muhammad Khalifa and Rishabh Agarwal and Lajanugen Logeswaran and Jaekyeom Kim and Hao Peng and Moontae Lee and Honglak Lee and Lu Wang},
      year={2025},
      journal={arXiv preprint arXiv:2504.16828},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.16828}, 
}
```
