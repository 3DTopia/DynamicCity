---
layout: default
---

<div class="title-container">
  <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/logo.png" alt="Logo" class="logo">
  <h1>
    <span class="main-title"><span class="dynamic">Dynamic</span><span class="city">City</span>: Large-Scale LiDAR</span>
    <span class="main-title">Generation from Dynamic Scenes</span>
  </h1>
</div>

{% include_relative authors.md %}
{% include_relative links.md %}

<div class="teaser-container">
  <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/teaser.webp" alt="Teaser Image" class="teaser-image">

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
  <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/pipeline.png" alt="Pipeline Image" class="method-image">

  <p class="method-caption">
  Our <span class="highlight-pink">Dynamic</span><span class="highlight-blue">City</span> framework consists of two key procedures: <strong>(a)</strong> Encoding HexPlane with an VAE architecture, and <strong>(b)</strong> 4D Scene Generation with HexPlane DiT.
  </p>
</div>

## Dynamic Scene Generation Results

### 1. Unconditional Generation
<div class="demo-section">
  <div class="video-row">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/u_c_1.webp" alt="Unconditional Generation 1" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/u_c_2.webp" alt="Unconditional Generation 2" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/u_c_3.webp" alt="Unconditional Generation 3" class="video-small">
  </div>
  
  <div class="video-row">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/u_w_1.webp" alt="Unconditional Generation 1" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/u_w_2.webp" alt="Unconditional Generation 2" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/u_w_3.webp" alt="Unconditional Generation 3" class="video-small">
  </div>
</div>

### 2. HexPlane Conditional Generation
<div class="demo-section">
  <div class="video-row">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/h_c_1.webp" alt="HexPlane Conditional Generation 1" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/h_c_2.webp" alt="HexPlane Conditional Generation 2" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/h_c_3.webp" alt="HexPlane Conditional Generation 3" class="video-small">
  </div>
  
  <div class="video-row">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/h_w_1.webp" alt="HexPlane Conditional Generation 1" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/h_w_2.webp" alt="HexPlane Conditional Generation 2" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/h_w_3.webp" alt="HexPlane Conditional Generation 3" class="video-small">
  </div>
</div>

### 3. Command/Trajectory-Driven Generation
<div class="demo-section">
  <div class="video-row">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/r_c_1.webp" alt="Command-Driven Generation 1" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/r_c_2.webp" alt="Command-Driven Generation 2" class="video-small">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/r_c_3.webp" alt="Command-Driven Generation 3" class="video-small">
  </div>
</div>

### 4. Layout-Conditioned Generation
<div class="demo-section">
  <div class="video-row">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/l_c_1.webp" alt="Layout-Conditioned Generation 1" class="video-normal">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/l_c_2.webp" alt="Layout-Conditioned Generation 2" class="video-normal">
  </div>
  
  <div class="video-row">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/l_w_1.webp" alt="Layout-Conditioned Generation 1" class="video-normal">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/l_w_2.webp" alt="Layout-Conditioned Generation 2" class="video-normal">
  </div>
</div>

### 5. Dynamic Scene Inpainting
<div class="demo-section">
  <div class="video-row">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/i_c_1.webp" alt="Dynamic Scene Inpainting 1" class="video-normal">
    <img src="https://raw.githubusercontent.com/dynamic-city/dynamic-city.github.io/main/assets/images/i_c_2.webp" alt="Dynamic Scene Inpainting 2" class="video-normal">
  </div>
</div>

{% include_relative bibtex.md %}