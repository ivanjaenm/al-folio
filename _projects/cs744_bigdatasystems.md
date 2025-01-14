---
layout: page_project
title: CS744 - Big Data Systems
description: Memory Efficient Low-rank Systems for Large Vision/Language Models
img: assets/img/projects/cs744/meme.jpg
importance: 4
category: work
github: https://github.com/ivanjaenm/Low-Rank-training-GaloLTE
---

## Abstract

Deep learning models are becoming bigger, with parameter count rapidly increasing, making it more and more infeasible to train all at once due to hardware constraints on most machines. In this work, we focused on exploring memory efficient systems for pre-training large foundation models through the use of low rank structures, such as LoRA [1]. More specifically, we primarily analyze recent strategies, such as LTE [2] and GaLore [3], which can achieve efficiency while maintaining model performance. We aimed to evaluate their performance in a small number of training nodes, and eventually combine their complementary advantages.

## Methods

- ### LTE (low rank weights)
LTE work [2] shows that during pre-training, the effective rank of the model parameters keeps increasing. They also show that choosing the rank equal to the smallest dimension of weight matrix achieves performance close to full-parameter training. However, training with smaller rank adapters fails to achieve this performance. Their key observation was that a matrix can be recreated by addition of lower-dimensional matrices. Therefore, we can train multiple low-rank matrices to reparametarize the higher-rank ones without compromising performance. This technique is called as multi-head LoRA (*mhlora*). Specifically, if there are $$N$$ low-rank heads, then the layer output $$h(x)$$ will be:

$$h_{mhlora}(x) = W x + \frac{s}{N} \sum_{n=1}{N}B_n A_n x$$




- ### GaLore (low rank gradients)
The regular pre-training weight update for a typical optimizer such as Adam, can be written down as:
\begin{equation}
W_T = W_0 + \eta \sum_{t=0}^{T-1} \tilde{G_t} = W_0 + \eta \sum^{T-1}_{t=0} \rho_t(G_t)
\end{equation}
Where $$\eta$$ is the learning rate, $$\tilde{G_t}$$ is the final processed gradient to be added to the weight matrix, and $$\rho_t$$ is an entry-wise stateful gradient regularizer (e.g., Adam). The state of $$\rho_t$$ is potentially memory-intensive. 
Instead, Gradient low-rank projection (GaLore) denotes the following gradient update rules:

\begin{equation}
W_T = W_0 + \eta \sum_{t=0}^{T-1}\tilde{G_t}
\end{equation} 

With $$\tilde{G_t} = P_t\rho_t(P_t^\top G_tQ_t)Q_t^\top$$, where $$P_t \in \mathbb{R}^{ m \times r}$$ and $$Q_t \in \mathbb{R}^{ r \times n}$$ are projection matrices.

## Difference between LoRA and GaLore

While both **GaLore** and **LoRA** use low-rank structures, they follow very different learning trajectories. For example, when $$r = min(m, n)$$, GaLore with $$\rho_t \equiv 1$$ follows the exact training trajectory of the original model, as $$\tilde{G_t} =P_t P^{\top}_t G_t Q_tQ^{\top}_t = G_t$$. On the other hand, when $$BA$$ reaches full rank, optimizing $$B$$ and $$A$$ simultaneously follows very different training trajectory from the original model.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/cs744/galore1.png" title="LTE process" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/cs744/galore2.png" title="Galore process" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    a) LTE reparametarizes a model weight matrix with multiple LoRA heads and train them independently for T iterations. b) In Galore, the learning happens through low-rank subspaces $\Delta W_{T_1}$ and $\Delta W_{T_2}$
</div>

## Experiments and Results

We ran our experiments in 4 CloudLab GPUs (NVIDIA P100, each with 12 GB of memory).

We trained a GPT-2 model [4] with multi-head LoRA architecture from scratch on CNN Dailymail dataset. The model has 512 context length and about 124 million parameters. We used weight-tying between the token embedding and final linear layer.

In the following picture we compare the training speed and number of trainable parameters in sequential and parallel-merge method. We observe that sequential multi-head takes 25% more time than parallel multi-head method. However, parallel multi-head training with rank 1 and 4 take about the same time as full parameter training, but with 24% less trainable parameters.

<div class="row">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/cs744/experiments1.png" title="parameter size vs iteration time" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Parameter size vs Iteration time
</div>

Figure below shows the test loss and number of parameters trained when different combinations of layers are replaced by multi-head LoRA layers. We used 4 heads in parallel-merge method and trained GPT-2 model for 500 iterations. Replacing attention layer with LoRA is most effective while replacing logit layer is least effective. Replacing MLP layers in decoder blocks reduces parameters most. However, it also degrades performance. As expected, replacing multiple types of layers have higher test loss compared to replacing single type of layer. Replacing logit layer with multi-head LoRA increased number of parameters because we used weight tying between embedding and logit layer in GPT-2 model.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/cs744/experiments2.png" title="parameter size vs iteration time" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Test loss and number of trainable parameters when different layers are replaced by multi-head LoRA layer. (A = Attention, M = MLP, L = Logit)
</div>

For Galore we experimented training a LLaMA-1B language model using the C4 common crawl dataset for a small number of epochs. We observed a 30% decrease in memory foot-print compared with the full training approach (see Figure below). Since the model hidden size used is about 2K, we employed a GaLore rank of 1024. Additionally, we found that varying the GaLore rank has little effect on accuracy or memory load. We are not reporting tables, because there was simply not
substancial change in results.

<div class="row">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/cs744/experiments3.png" title="Galore memory footprint" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Comparing memory footprint of using Adam optimizer vs GaLore optimize
</div>

We also evaluated the generality of low rank techniques across different domains by also exploring utility in vision tasks. A practical use case of using low rank adaptations to fine-tune a model is in the medical field using foundation models trained on natural images and adapting them to medical images for computer aided diagnosis. Therefore, our vision experiments provide evidence for the practical importance of these low rank systems

The dataset used is the 3x224x224 image size PathMNIST dataset from the larger MedMNIST dataset. This has been established as a benchmark for medical foundation models, especially with the 224 image size being a larger and newer addition. The two models tested were vit base patch16 224 and vit large patch16 224, as they were the largest pretrained models that we could find and use easily. We compared the memory load of varying LoRA rank (see Figure below). Varying the rank of regular LoRA had only small but noticeable affects of changing the memory load. All ranks, however, were significantly lighter on memory load and converged faster than full fine tuning. The base size and large size ViTs had the same memory saving trends, with large size just scaled up to about 11900 MB rather than 6100MB.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/cs744/experiments4.png" title="ViT experiments" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Evaluating ViT base patch16 224 using regular LoRA at differing ranks compared to full fine tuning for only 1 epoch. 
</div>

## Poster
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/cs744/poster.png" title="Poster" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Poster session
</div>

References
-----

1. [LoRA: Low-Rank Adaptation of Large Language Models, Hu et. al.](https://arxiv.org/pdf/2106.09685)
2. [Training Neural Networks from Scratch with Parallel Low-Rank Adapters, Huh et. al.](https://arxiv.org/pdf/2402.16828)
3. [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection, Zhao et. al.](https://arxiv.org/pdf/2403.03507)
4. [nanogpt: Minimal gpt implementation for educational purposes, Karpathy](https://github.com/karpathy/nanoGPT)

{% raw %}
{% endraw %}
