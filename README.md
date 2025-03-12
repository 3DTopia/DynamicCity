<img src="https://dynamic-city.github.io/assets/images/logo.png" width="12.5%" align="left">

# DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes

<p align="center">
  <a href="https://bianhengwei.com/" target="_blank">Hengwei Bian</a><sup>1,2,*</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://ldkong.com/" target="_blank">Lingdong Kong</a><sup>1,3</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://www.infinitescript.com/about/" target="_blank">Haozhe Xie</a><sup>4</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=lSDISOcAAAAJ&amp;hl=en" target="_blank">Liang Pan</a><sup>1,†,‡</sup>
  <a href="https://mmlab.siat.ac.cn/yuqiao" target="_blank">Yu Qiao</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>4</sup>
  <br />
  <sup>1</sup>Shanghai AI Laboratory&nbsp;&nbsp;&nbsp;
  <sup>2</sup>Carnegie Mellon University&nbsp;&nbsp;&nbsp;
  <br />
  <sup>3</sup>National University of Singapore&nbsp;&nbsp;&nbsp;
  <sup>4</sup>S-Lab, Nanyang Technological University
  <br />
  <sup>*</sup>Work done during an internship at Shanghai AI Laboratory&nbsp;&nbsp;&nbsp;
  <sup>†</sup>Corresponding author&nbsp;&nbsp;&nbsp;
  <sup>‡</sup>Project lead
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.18084/" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%96-darkred">
  </a>

  <a href="https://dynamic-city.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-orange">
  </a>

  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=dynamic-city.DynamicCity">
  </a>
</p>

<img src="https://dynamic-city.github.io/assets/images/teaser_small.webp" alt="Teaser" width="100%">

Urban scene generation has been developing rapidly recently. However, existing methods primarily focus on generating
static and single-frame scenes, overlooking the inherently dynamic nature of real-world driving environments. In this
work, we introduce **DynamicCity**, a novel 4D occupancy generation framework capable of generating large-scale, 
high-quality dynamic 4D scenes with semantics. DynamicCity mainly consists of
two key models: 1. A VAE model for learning HexPlane as the compact 4D representation. Instead of using naive averaging
operations, DynamicCity employs a novel **Projection Module** to effectively compress 4D features into six 2D
feature maps for HexPlane construction, which significantly enhances HexPlane fitting quality (up to **12.56** mIoU
gain). Furthermore, we utilize an **Expansion & Squeeze Strategy** to reconstruct 3D feature volumes in parallel, which
improves both network training efficiency and reconstruction accuracy than naively querying each 3D point (up to **7.05**
mIoU gain, **2.06x** training speedup, and **70.84%** memory reduction). 2. A DiT-based diffusion model for HexPlane
generation. To make HexPlane feasible for DiT generation, a **Padded Rollout Operation** is proposed to reorganize all
six feature planes of the HexPlane as a squared 2D feature map. In particular, various conditions could be introduced in
the diffusion or sampling process, supporting **versatile 4D generation applications**, such as trajectory- and
command-driven generation, inpainting, and layout-conditioned generation. Extensive experiments on the CarlaSC and Waymo
datasets demonstrate that DynamicCity significantly outperforms existing state-of-the-art 4D occupancy generation methods
across multiple metrics. The code and models have been released to facilitate future research.

# Overview

<img src="https://dynamic-city.github.io/assets/images/pipeline.png" alt="Overview" width="100%">

Our **DynamicCity** framework consists of two key procedures: **(a)** Encoding HexPlane with an VAE architecture, 
and **(b)** 4D Scene Generation with HexPlane DiT.

## Updates

- **[February 2025]**: Code released.
- **[October 2024]**: Project page released.

## Outline

- [Installation](#gear-installation)
- [Data Preparation](#hotsprings-data-preparation)
- [Getting Started](#rocket-getting-started)
- [Dynamic Scene Generation](#cityscape-dynamic-scene-generation)

## :gear: Installation

```shell
conda create -n dyncity python=3.10 -y
conda activate dyncity
conda install pytorch==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install einops hydra-core matplotlib numpy omegaconf timm tqdm wandb -c conda-forge -y
pip install flash-attn --no-build-isolation
```

## :hotsprings: Data Preparation

Download the CarlaSC dataset from [here](https://umich-curly.github.io/CarlaSC.github.io/download/) and extract it into
the `./carlasc` directory.
Your repository should look like this:

```
DynamicCity
├── carlasc/
│   ├── Cartesian/
│   │   ├── Train/
│   │   │   ├── Town01_Heavy
│   │   │   ├── ...
│   │   ├── Test/
├── ...
```

## :rocket: Getting Started

You can obtain our checkpoints from [here](https://1drv.ms/f/c/29cb1a5046dedcc5/Eo1HiB44MCVEk0kJ5BNrZ-IBR6MUYd3iC4sSyl7hI7Iyxw?e=1sn7dk).
To use the pretrained models, simply unzip `ckpts.zip` as `./ckpts` and run `infer_dit.py`.

To train VAE on CarlaSC dataset, run the following command:

```shell
torchrun --nproc-per-node 8 train.py VAE carlasc name=DYNAMIC_CITY_VAE
```

After VAE is trained, save hexplane rollouts using:

```shell
torchrun --nproc-per-node 8 infer_vae.py -n DYNAMIC_CITY_VAE --save_rollout --best
```

Then, you can train your DiT using this command:

```shell
torchrun --nproc-per-node 8 train.py DiT carlasc name=DYNAMIC_CITY_DIT vae_name=DYNAMIC_CITY_VAE
```

Finally, use DiT to sample novel city scenes:

```shell
torchrun --nproc-per-node 8 infer_dit.py -d DYNAMIC_CITY_DIT --best_vae
```

## :cityscape: Dynamic Scene Generation

### Unconditional Generation

|                                                                                                                  |                                                                                                                  |                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| <img src="https://dynamic-city.github.io/assets/images/u_c_1.webp" alt="Unconditional Generation 1" width="240"> | <img src="https://dynamic-city.github.io/assets/images/u_c_2.webp" alt="Unconditional Generation 2" width="240"> | <img src="https://dynamic-city.github.io/assets/images/u_c_3.webp" alt="Unconditional Generation 3" width="240"> |
|                                                                                                                  |                                                                                                                  |                                                                                                                  |

### HexPlane Conditional Generation

|                                                                                                                  |                                                                                                                  |                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| <img src="https://dynamic-city.github.io/assets/images/h_c_1.webp" alt="Unconditional Generation 1" width="240"> | <img src="https://dynamic-city.github.io/assets/images/h_c_2.webp" alt="Unconditional Generation 2" width="240"> | <img src="https://dynamic-city.github.io/assets/images/h_c_3.webp" alt="Unconditional Generation 3" width="240"> |
|                                                                                                                  |                                                                                                                  |                                                                                                                  |

### Command & Trajectory-Driven Generation

|                                                                                                                  |                                                                                                                  |                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| <img src="https://dynamic-city.github.io/assets/images/r_c_1.webp" alt="Unconditional Generation 1" width="240"> | <img src="https://dynamic-city.github.io/assets/images/r_c_2.webp" alt="Unconditional Generation 2" width="240"> | <img src="https://dynamic-city.github.io/assets/images/r_c_3.webp" alt="Unconditional Generation 3" width="240"> |
|                                                                                                                  |                                                                                                                  |                                                                                                                  |

### Layout-Conditioned Generation

|                                                                                                                  |                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| <img src="https://dynamic-city.github.io/assets/images/l_c_1.webp" alt="Unconditional Generation 1" width="360"> | <img src="https://dynamic-city.github.io/assets/images/l_c_2.webp" alt="Unconditional Generation 2" width="360"> |
|                                                                                                                  |                                                                                                                  |

### Dynamic Scene Inpainting

|                                                                                                                  |                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| <img src="https://dynamic-city.github.io/assets/images/i_c_1.webp" alt="Unconditional Generation 1" width="360"> | <img src="https://dynamic-city.github.io/assets/images/i_c_2.webp" alt="Unconditional Generation 2" width="360"> |
|                                                                                                                  |                                                                                                                  |

## Citation

If you find this work helpful for your research, please kindly consider citing our papers:

```bibtex
@inproceedings{bian2025dynamiccity,
  title={DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes},
  author={Bian, Hengwei and Kong, Lingdong and Xie, Haozhe and Pan, Liang and Qiao, Yu and Liu, Ziwei},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025},
}
```
