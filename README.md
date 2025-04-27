# Process Reward Models That Think

<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.16828)  [![Github](https://img.shields.io/badge/ThinkPRM-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/mukhal/thinkprm)  [![Data](https://img.shields.io/badge/Data-white?style=for-the-badge&logo=huggingface&logoColor=orange&color=yellow
)](https://huggingface.co/datasets/launch/thinkprm-1K-verification-cots)  [![ThinkPRM](https://img.shields.io/badge/ThinkPRM1.5B-white?style=for-the-badge&logo=huggingface&logoColor=orange&color=purple
)](https://huggingface.co/launch/ThinkPRM-1.5B)  [![ThinkPRM](https://img.shields.io/badge/ThinkPRM14B-white?style=for-the-badge&logo=huggingface&logoColor=orange&color=brown
)](https://huggingface.co/launch/ThinkPRM-14B)




<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a>
  </p>
  <p>
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div>

</div>

# 🎉News

- **[2025-04-23]** Our [paper](https://arxiv.org/abs/2504.16828) is released on arxiv.
- **[2025-04-24]** Our synthetic verification CoTs used to train ThinkPRM on are now on [huggingface](https://huggingface.co/datasets/launch/thinkprm-1K-verification-cots). 
- **[2025-04-25]** Our trained PRMs are released in two sizes: [1.5B](https://huggingface.co/launch/ThinkPRM-1.5B) and [14B](https://huggingface.co/launch/ThinkPRM-14B) finetuned from R1-Distill-Qwen models.

# 📖Introduction

We introduce ThinkPRM, a collection of generative long CoT process reward models. Our verifiers are obtained by finetuning reasoning models over 1K synthetic verification CoTs---filtered based on only on 8K process labels from PRM800K. The resulting verifiers outperform LLM-as-a-judge, discriminative PRMs, on most in- and out-of-domain setups. ThinkPRM enables scaling up verifier  compute either in parallel or sequentially by thinking longer.


# ✨Getting Started

*Code is coming soon.*


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
