---
layout: default
---

<div class="title-container">
  <img src="assets/images/logo.png" alt="Logo" class="logo">
  <h1>
    <span class="main-title"><span class="dynamic">Dynamic</span><span class="city">City</span>: Large-Scale LiDAR</span>
    <span class="main-title">Generation from Dynamic Scenes</span>
  </h1>
</div>

<div class="authors">
  <div class="author-names">
    <a href="https://bian-hengwei.github.io/" target="_blank">Hengwei Bian</a><sup>1,2,*</sup>,
    <a href="https://ldkong.com/" target="_blank">Lingdong Kong</a><sup>1,3</sup>,
    <a href="https://www.infinitescript.com/about/" target="_blank">Haozhe Xie</a><sup>4</sup>,
    <a href="https://scholar.google.com/citations?user=lSDISOcAAAAJ&hl=en/" target="_blank">Liang Pan</a><sup>1,†,‡</sup>,
    <a href="https://mmlab.siat.ac.cn/yuqiao" target="_blank">Yu Qiao</a><sup>1</sup>,
    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>4</sup>
  </div>
  
  <div class="affiliations">
    <sup>1</sup>Shanghai AI Laboratory &nbsp;&nbsp;
    <sup>2</sup>Carnegie Mellon University &nbsp;&nbsp;
    <sup>3</sup>National University of Singapore<br>
    <sup>4</sup>S-Lab, Nanyang Technological University
  </div>

  <div class="notes">
    <sup>*</sup>Work done during an internship at Shanghai AI Laboratory &nbsp;&nbsp;
    <sup>†</sup>Corresponding author &nbsp;&nbsp;
    <sup>‡</sup>Project lead
  </div>
</div>

<div class="links">
  <a href="https://arxiv.org/abs/2410.18084v1" target="_blank" rel="noopener noreferrer">
    <i class="fas fa-file-alt"></i> Arxiv
  </a>
  <a href="https://github.com/dynamic-city/DynamicCity" target="_blank" rel="noopener noreferrer">
    <i class="fab fa-github"></i> Code
  </a>
</div>

<div class="teaser-container">
  <img src="assets/images/teaser.webp" alt="Teaser Image" class="teaser-image">

  <p class="teaser-caption">
  We introduce a new LiDAR generation model that generates diverse 4D scenes of large spatial scales (80×80×6.4 m³) and long sequential modeling (up to 128 frames), enabling a diverse set of downstream applications.
  </p>
</div>

## Abstract

<div class="abstract">
LiDAR scene generation has been developing rapidly recently. However, existing methods primarily focus on generating static and single-frame scenes, overlooking the inherently dynamic nature of real-world driving environments. In this work, we introduce <span class="highlight-pink">Dynamic</span><span class="highlight-blue">City</span>, a novel 4D LiDAR generation framework capable of generating large-scale, high-quality LiDAR scenes that capture the temporal evolution of dynamic environments. DynamicCity mainly consists of two key models:

1. A VAE model for learning HexPlane as the compact 4D representation. Instead of using naive averaging operations, DynamicCity employs a novel <span class="highlight">Projection Module</span> to effectively compress 4D LiDAR features into six 2D feature maps for HexPlane construction, which significantly enhances HexPlane fitting quality (up to <span class="highlight">12.56</span> mIoU gain). Furthermore, we utilize an <span class="highlight">Expansion & Squeeze Strategy</span> to reconstruct 3D feature volumes in parallel, which improves both network training efficiency and reconstruction accuracy than naively querying each 3D point (up to <span class="highlight">7.05</span> mIoU gain, <span class="highlight">2.06x</span> training speedup, and <span class="highlight">70.84%</span> memory reduction).

2. A DiT-based diffusion model for HexPlane generation. To make HexPlane feasible for DiT generation, a <span class="highlight">Padded Rollout Operation</span> is proposed to reorganize all six feature planes of the HexPlane as a squared 2D feature map. In particular, various conditions could be introduced in the diffusion or sampling process, supporting <span class="highlight">versatile 4D generation applications</span>, such as trajectory- and command-driven generation, inpainting, and layout-conditioned generation.

Extensive experiments on the CarlaSC and Waymo datasets demonstrate that DynamicCity significantly outperforms existing state-of-the-art 4D LiDAR generation methods across multiple metrics. The code will be released to facilitate future research.
</div>

## Method

<div class="method-container">
  <img src="assets/images/pipeline.png" alt="Pipeline Image" class="method-image">

  <p class="method-caption">
  Our <span class="highlight-pink">Dynamic</span><span class="highlight-blue">City</span> framework consists of two key procedures: <strong>(a)</strong> Encoding HexPlane with an VAE architecture, and <strong>(b)</strong> 4D Scene Generation with HexPlane DiT.
  </p>
</div>

## Dynamic Scene Generation Results

### 1. Unconditional Generation
<div class="demo-section">
  <div class="video-row">
    <img src="assets/images/u_c_1.webp" alt="Unconditional Generation 1" class="video-small">
    <img src="assets/images/u_c_2.webp" alt="Unconditional Generation 2" class="video-small">
    <img src="assets/images/u_c_3.webp" alt="Unconditional Generation 3" class="video-small">
  </div>
  
  <div class="video-row">
    <img src="assets/images/u_w_1.webp" alt="Unconditional Generation 1" class="video-small">
    <img src="assets/images/u_w_2.webp" alt="Unconditional Generation 2" class="video-small">
    <img src="assets/images/u_w_3.webp" alt="Unconditional Generation 3" class="video-small">
  </div>
</div>

### 2. HexPlane Conditional Generation
<div class="demo-section">
  <div class="video-row">
    <img src="assets/images/h_c_1.webp" alt="HexPlane Conditional Generation 1" class="video-small">
    <img src="assets/images/h_c_2.webp" alt="HexPlane Conditional Generation 2" class="video-small">
    <img src="assets/images/h_c_3.webp" alt="HexPlane Conditional Generation 3" class="video-small">
  </div>
  
  <div class="video-row">
    <img src="assets/images/h_w_1.webp" alt="HexPlane Conditional Generation 1" class="video-small">
    <img src="assets/images/h_w_2.webp" alt="HexPlane Conditional Generation 2" class="video-small">
    <img src="assets/images/h_w_3.webp" alt="HexPlane Conditional Generation 3" class="video-small">
  </div>
</div>

### 3. Command/Trajectory-Driven Generation
<div class="demo-section">
  <div class="video-row">
    <img src="assets/images/r_c_1.webp" alt="Command-Driven Generation 1" class="video-small">
    <img src="assets/images/r_c_2.webp" alt="Command-Driven Generation 2" class="video-small">
    <img src="assets/images/r_c_3.webp" alt="Command-Driven Generation 3" class="video-small">
  </div>
</div>

### 4. Layout-Conditioned Generation
<div class="demo-section">
  <div class="video-row">
    <img src="assets/images/l_c_1.webp" alt="Layout-Conditioned Generation 1" class="video-normal">
    <img src="assets/images/l_c_2.webp" alt="Layout-Conditioned Generation 2" class="video-normal">
  </div>
  
  <div class="video-row">
    <img src="assets/images/l_w_1.webp" alt="Layout-Conditioned Generation 1" class="video-normal">
    <img src="assets/images/l_w_2.webp" alt="Layout-Conditioned Generation 2" class="video-normal">
  </div>
</div>

### 5. Dynamic Scene Inpainting
<div class="demo-section">
  <div class="video-row">
    <img src="assets/images/i_c_1.webp" alt="Dynamic Scene Inpainting 1" class="video-normal">
    <img src="assets/images/i_c_2.webp" alt="Dynamic Scene Inpainting 2" class="video-normal">
  </div>
</div>

## BibTeX

```bibtex
@article{bian2024dynamiccity,
  title={DynamicCity: Large-Scale LiDAR Generation from Dynamic Scenes},
  author={Bian, Hengwei and Kong, Lingdong and Xie, Haozhe and Pan, Liang and Qiao, Yu and Liu, Ziwei},
  journal={arXiv preprint arXiv:2410.18084},
  year={2024}
}
```
