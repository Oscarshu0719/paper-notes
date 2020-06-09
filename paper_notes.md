# Paper notes

[Deep Residual Learning for Image Recognition](##Deep Residual Learning for Image Recognition)

[Generative Adversarial Nets](##Generative Adversarial Nets)

[Conditional Generative Adversarial Nets](##Conditional Generative Adversarial Nets)

[Image-to-Image Translation with Conditional Adversarial Networks](##Image-to-Image Translation with Conditional Adversarial Networks)

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](##Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks)

[Wasserstein GAN](##Wasserstein GAN)

[Least Squares Generative Adversarial Networks](##Least Squares Generative Adversarial Networks)

[BEGAN: Boundary Equilibrium Generative Adversarial Networks](##BEGAN: Boundary Equilibrium Generative Adversarial Networks)

[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](##Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network)

[Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](##Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks)

[Energy-Based Generative Adversarial Networks](##Energy-Based Generative Adversarial Networks)

[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](##Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks)

[Progressive Growing of GANs for Improved Quality, Stability, and Variation](##Progressive Growing of GANs for Improved Quality, Stability, and Variation)

[Conditional Image Synthesis with Auxiliary Classifier GANs](##Conditional Image Synthesis with Auxiliary Classifier GANs)

[Improving the Improved Training of Wasserstein GANs: A Consistency Term and Its Dual Effect](##Improving the Improved Training of Wasserstein GANs: A Consistency Term and Its Dual Effect)

[Spectral Normalization for Generative Adversarial Networks](##Spectral Normalization for Generative Adversarial Networks)

[Wasserstein Divergence for GANs](##Wasserstein Divergence for GANs)

[InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](##InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets)

[Autoencoding beyond pixels using a learned similarity metric](##Autoencoding beyond pixels using a learned similarity metric)

[Adversarial Feature Learning](##Adversarial Feature Learning)

[Triple Generative Adversarial Nets](##Triple Generative Adversarial Nets)

[Adversarial Ranking for Language Generation](##Adversarial Ranking for Language Generation)

[XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings](##XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings)

## Deep Residual Learning for Image Recognition

>   ResNet

1. On the **ImageNet** dataset, the residual nets with a depth of up to 152 layers, which are 8 times deeper than **VGG** nets but still having **lower complexity**.

4. Vanishing/exploding gradients problem has been largely addressed by **normalized initialization** and **intermediate normalization** layers.

5. Adding more layers to a suitably deep model leads to higher training error.

6. Our extremely deep residual nets are easy to optimize, but the counterpart "plain" nets (that simply stack layers) exhibit higher training error when the depth increases.

7. When deeper networks are able to start converging, degradation problem has been exposed with the network depth increasing, accuracy gets saturated (which might be
   unsurprising) and then degrades rapidly.

8. In this paper, we address the degradation problem by introducing a deep residual learning framework. Denoting the desired underlying mapping as $\mathcal{H}(x)$, and let the stacked nonlinear layers fit another mapping of $\mathcal{F}(x) := \mathcal{H}(x)−x$.![1.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/1.1.png)

9. The entire network can still be trained end-to-end by SGD with backpropagation, and can be easily implemented using common libraries (e.g., Caffe [19]) without modifying the solvers.

10. Building block: 
$$
   y = \mathcal{F}(x, \{W_i\}) + x
$$
   e.g., A network has two layers. $\mathcal{F} = W_2\sigma(W_1x)$, where $\sigma$ denotes ReLU and the biases are omitted for simplifying notations.
   If the dimensions of $x$ and $\mathcal{F}$ are NOT equal, we can use a square matrix $W_s$ to match the dimensions:
$$
   y = \mathcal{F}(x, \{W_i\}) + W_sx
$$

11. Plain network: 

    -   The convolutional layers mostly have $3 \times 3$ filters.
    -   For the same output feature map size, the layers have the same number of filters.
    -   If the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer.
    -   Downsample directly by convolutional layers that have a stride of $2$.
    -   The network ends with a global average pooling layer and a $1000$-way
        fully-connected layer with softmax. The total number of weighted layers is $34$.
    -   Fewer filters and lower complexity than VGG nets.

12. Residual network: 

    -   The identity shortcuts can be directly used when the input and output are of the same dimensions (solid line shortcuts).
    -   When the dimensions increase (dotted line shortcuts), there are two options:
        -   The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions.
        -   The projection shortcut is used to match dimensions.`
        -   For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of $2$. The former got lower complexity.
            ![1.2](https://github.com/Oscarshu0719/paper-notes/blob/master/img/1.2.png)
13. Architecture:
    ![1.3](https://github.com/Oscarshu0719/paper-notes/blob/master/img/1.3.png)

### Vocabulary

1.  hamper 阻礙
2.  degradation 降解
3.  akin 類似的
4.  coarse 粗糙的
5.  condition (v.) 訓練；使適應
6.  perturbation 擔心

## Generative Adversarial Nets

>   GAN

1.  Adversarial nets:
    $$
    \min_G\max_D V(D, G) = \mathbb{E}_{x \sim p_{data(x)}}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z))]
    $$
where $p_g$ represents $G$'s distribution over data $x$, $p_z(z)$ represents noise variables, and $D(x)$: represents the probability that $x$ came from the data rather than $p_g$.
    
2.  Algorithm pesudo code:
    ![2.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/2.1.png)
    
3.  For $G$ fixed, the optimal $D$ is 
    $$
    D^*_G(X) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
    $$

4.  The global minimum of the virtual training criterion $C(G)$ is achieved iff $p_g = p_{data}$. At that point, $C(G)$ achieves the value $-\log 4$.

5.  If $G$ and $D$ have enough capacity, and at each step of *Algorithm 1*, $D$ is allowed to reach its optimum given $G$, and $p_g$ is updated so as to improve the criterion
    $$
    \mathbb{E}_{x \sim p_{data}}[\log D^*_G(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D^*_G(x))]
    $$
    then $p_g$ converges to $p_{data}$.

6.  Pros and cons:

    -   Pros:
        -   Markov chains are NEVER needed, only backprop is used to obtain gradients.
        -   NO inference is needed during learning, and a wide variety of functions can be incorporated into the model.
    -   Cons:
        -   There is NO explicit representation of $p_g(x)$, and that $D$ must be synchronized well with $G$ during training (in particular, $G$ must NOT be trained too much without updating $D$, in order to avoid “the Helvetica scenario” in which $G$ collapses too many values of $z$ to the same value of $x$ to have enough diversity to model $p_data$), much as the negative chains of a Boltzmann machine must be kept up to date between learning steps.

### Vocabulary

1.  leverage 平衡
2.  piecewise 分段的
3.  sidestep 迴避
4.  counterfeit 偽造
5.  analogous 相似的
6.  corpora 語料庫
7.  intriguing 有趣的
8.  perceptible 可察覺的
9.  pedagogical 教學法
10.  aforementioned 前面提到的

## Conditional Generative Adversarial Nets

>   cGAN

1.  Conditional Adversarial Nets:
    $$
    \min_G\max_D V(D, G) = \mathbb{E}_{x \sim p_{data(x)}}[\log D(x | y)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z | y))]
    $$
    where $y$ is auxiliary information, such as class labels or data from other modalities.
    ![3.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/3.1.png)

### Vocabulary

1.  intractable 棘手的
2.  modality 情態
3.  synonymous 同義的

## Image-to-Image Translation with Conditional Adversarial Networks

>   Pix2pix

1.  If we take a naive approach and ask the CNN to minimize the Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results. This is because Euclidean distance is **minimized by averaging all plausible outputs**, which causes blurring.

2.  Generator:
    $$
    G: \{x, z\} \rightarrow y
    $$
    where $x$ is image, $z$ is noise vector and $y$ is output image.

3.  Objective:

    -   Conditional GAN: 
        $$
        \mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}[\log (1 - D(x, G(x, z)))]
        $$
        where $G$ tries to minimize this objective against $D$ that tries to maximize it, i.e.
        $$
        G^* = \text{argmin}_G\max_D \mathcal{L}_{cGAN}(G, D)
        $$

    -   GAN:
        $$
        \mathcal{L}_{GAN}(G, D) = \mathbb{E}_y[\log D(y)] + \mathbb{E}_{x, z}[\log (1 - D(G(x, z)))]
        $$

    -   L1 distance:
        $$
        \mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[||y - G(x, z)||_1]
        $$

    -   Final objective:
        $$
        G^* = \text{argmin}_G\max_D\mathcal{L}_{cGAN}(G, D) + \lambda\mathcal{L}_{L1}(G)
        $$
        Without $z$, the net could still learn a mapping from $x$ to $y$, but would produce deterministic outputs.

4.  Architecture:

    -   Generator with skips: We add skip connections between each layer $i$ and layer $n - i$, where $n$ is the total number of layers. Each skip connection simply concatenates all channels at layer $i$ with those at layer $n - i$.
        ![4.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/4.1.png)
    -   Markovian discriminator (PatchGAN): 
        -   Only penalizes structure at the scale of patches. This discriminator tries to classify if each $N \times N$ patch in an image is real or fake. We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate output of $D$.
        -   $N$ can be much smaller than the full size of the image and still produce high quality results.

5.  Optimization and inference:

    -   We alternate between one gradient descent step on $D$, then one step on $G$.

    -   $$
        \begin{align*}
        & \min \quad \log (1 - D(x, G(x, z))) \\
        \Rightarrow \ & \max \quad \log D(x, G(x, z))
        \end{align*}
        $$

    -   We divide the objective by $2$ while optimizing $D$, which slows down the rate at
        which $D$ learns relative to $G$. 

    -   We use **minibatch SGD** and apply the **Adam** solver, with a learning rate of $0.0002$,
        and momentum parameters $\beta_1 = 0.5, \beta_2 = 0.999$.

    -   We run the generator net in exactly the same manner as during the **training** phase. This differs from the usual protocol in that we apply dropout at test time.

    -   We apply batch normalization using the statistics of the **test batch**, rather than aggregated statistics of the training batch.

### Vocabulary

1.  plausible 貌似真實（或可信）的
2.  multitude (n.) 許多的
3.  vigorous 蓬勃的
4.  circumvent 規避

## Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

>   DCGAN

1.  Architecture guidelines:
    -   Replace any pooling layers with strided convolutions ($D$) and fractional-strided convolutions ($G$).
    -   Use batchnorm in both $G$ and $D$.
    -   Remove fully connected hidden layers for deeper architectures.
    -   Use ReLU in $G$ for all layers except for the output, which uses $\tanh$.
    -   Use LeakyReLU in the $D$ for all layers.
2.  Details of adversarial training:
    -   NO pre-processing was applied to training images besides scaling to the range of the $\tanh$ activation function $[-1, 1]$.
    -   All models were trained with mini-batch SGD with a mini-batch size of $128$.
    -   All weights were initialized from a zero-centered Normal distribution with standard deviation $0.02$.
    -   In the LeakyReLU, the slope of the leak was set to $0.2$ in all models.
    -   Use the Adam optimizer with tuned hyperparameters, but set learning rate to $0.0002$.
    -   Set the momentum term $\beta_1$ to $0.5$.
        ![5.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/5.1.png)

### Vocabulary

1.  nonsensical 荒謬的
2.  wobbly 顫抖的
3.  oscillation 震盪

## Wasserstein GAN

>   WGAN

1.  The *Earth-Mover* (EM) distance or Wasserstein-1:
    $$
    W(\mathbb{P}_r, \mathbb{P}_g) = \inf_{\gamma \in \Pi(\mathbb{P}_r, \mathbb{P}_g)} \mathbb{E}_{(x, y) \sim \gamma} [||x - y||]
    $$
    where $\Pi(\mathbb{P}_r, \mathbb{P}_g)$ denotes the set of all joint distributions $\gamma(x, y)$ whose marginals
    are respectively $\mathbb{P}_r$ and $\mathbb{P}_g$. Intuitively,  $\gamma(x, y)$ indicates how much "mass" must be transported from $x$ to $y$ in order to transform the distributions $\mathbb{P}_r$ into the distribution $\mathbb{P}_g$. The EM distance then is the "cost" of the optimal transport plan.

2.  WGAN:

    -   Kantorovich-Rubinstein duality:

    $$
    W(\mathbb{P}_r, \mathbb{P}_\theta) = \sup_{||f||_L \le 1} \mathbb{E}_{x \sim \mathbb{P}_r}[f(x)] - \mathbb{E}_{x \sim \mathbb{P}_\theta}[f(x)]
    $$
    - $1$-Lipschitz functions:

    $$
    ||f||_L \le 1
    $$

    -   Replace with $K$-Lipschitz for some constant $K$:
    $$
    ||f||_L \le K
    $$

    -   $$
        K \times W(\mathbb{P}_r, \mathbb{P}_g) = \max_{w \in W} \mathbb{E}_{x \sim \mathbb{P}_r}[f_w(x)] - \mathbb{E}_{z \sim p(z)}[f_w(g_\theta(z))]
        $$

5.  Loss functions:
    
	-   $G$: 
        $$
        -\mathbb{E}_{x \sim \mathbb{P}_g}[f_w(x)]
        $$
    
    -   $D$:
        $$
        \mathbb{E}_{x \sim \mathbb{P}_g}[f_w(x)]-\mathbb{E}_{x \sim \mathbb{P}_r}[f_w(x)]
        $$
    
4.  Algorithm pseudo code:
    ![6.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/6.1.png)

5.  Differences with GAN:

    -   Remove sigmoid from last layer of $D$.
    -   Remove $\log$ from loss function of $G$ and $D$.
    -   Set the boundary of updated value.
    -   Replace Momentum-based optimizers (e.g. Momentum and Adam) with RMSProp or SGD.

6.  Improvement:

    -   Improve the stability of training extent.
    -   Basically solve the mode collapse problem.

### Vocabulary

1.  manifolds 流形
2.  negligible 微不足道的
3.  remedy 補救
4.  deviation 偏差
5.  delicate 細膩的

## Improved Training of Wasserstein GANs

>   WGAN-GP

1.  Objective function:
    $$
    L = \mathbb{E}_{\tilde{x} \sim \mathbb{P}_g}[D(\tilde{x})] - \mathbb{E}_{x \sim \mathbb{P}_r} [D(x)] + \lambda\mathbb{E}_{\hat{x} \sim \mathbb{P}_\hat{x}}[(||\nabla_{\hat{x}}D(\hat{x})||_2 - 1)^2]
    $$
    where $\tilde{x} = G(z)$, $z \sim p(z)$, and $\mathbb{P}_{\hat{x}}$ sampling uniformly along straight lines between
    pairs of points sampled from the data distribution $\mathbb{P}_r$ and the generator distribution $\mathbb{P}_g$. 
    All experiments in this paper use $\lambda= 10$.

2.  Algorithm pseudo code:
    ![7.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/7.1.png)

3.  Differences with WGAN:

    -   No batch normalization for $D$, but other normalization approaches are fine.

4.  Improvement:

    -   Solve the exploding and vanishing gradients problems.
    -   Converge faster than WGAN.
    -   Better stability of training approach.

## Least Squares Generative Adversarial Networks

>   LSGAN

1.  Objective functions:
    
    -   $D$:
    
    $$
    \min_D V(D) = \frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[(D(x) - b)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_z(z)}[(D(G(z)) - a)^2]
    $$
    
    -   $G$: 
        $$
        \min_G V(G) = \frac{1}{2}\mathbb{E}_{z \sim p_z(z)}[(D(G(z)) - c)^2]
        $$
    
    where $c$ denotes the value that $G$ wants $D$ to believe for fake data.
    
2.  Relation to $f$-divergence: 

    -   Objective function of $G$ can be:

    $$
    \min_G V(G) = \frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[(D(x) - c)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_z(z)}[(D(G(z)) - c)^2]
    $$

    $\frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[(D(x) - c)^2]$ does NOT change the optimal value since this term does NOT contain parameters of $G$.

    -   The optimal discriminator $D$ for a fixed $G$:
        $$
        D^*(x) = \frac{bp_{data}(x) + ap_g(x)}{p_{data}(x) + p_g(x)}
        $$

    -   In the original GAN paper, we have virtual training criterion:
        $$
        C(G) = KL(p_{data}||\frac{p_{data} + p_g}{2}) + KL(p_g||\frac{p_{data} + p_g}{2})
        $$

    -   
        $$
        2C(G) = \chi^2_{Pearson}(p_d + p_g || 2p_g) \\
        (\text{where} \ b - c = 1 \ \text{and} \ b - a = 2)
        $$
        where $\chi^2_{Pearson}$ is the Pearson $\chi^2$ divergence. Thus, we can minimize Pearson $\chi^2$ divergence between $p_d + p_g$ and $2p_g$.

3.  Architecture:
    ![8.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/8.1.png)

4.  Conditional LSGAN:

    -   Objective functions:

        -   $D$:
            $$
            \min_D V(D) = \frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[(D(x | \Phi(y)) - 1)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_z(z)}[(D(G(z) | \Phi(y)))^2]
            $$

        -   $G$:
            $$
            \min_G V(G) = \frac{1}{2}\mathbb{E}_{z \sim p_z(z)}[(D(G(z) | \Phi(y)) - 1)^2]
            $$
            where $\Phi(\cdot)$ denotes the linear mapping function and $y$ denotes the label vectors.
        
    -   Architecture:
        ![8.2](https://github.com/Oscarshu0719/paper-notes/blob/master/img/8.2.png)

5.  Differences:

    -   Similar to GAN and DCGAN, just change the objective function.

## BEGAN: Boundary Equilibrium Generative Adversarial Networks
>   BEGAN

1.  Loss for training a pixel-wise autoencoder:
    $$
    \mathcal{L}(v) = |v - D(v)|^\eta \ \text{where}
    \begin{cases}
    D: \mathbb{R}^{N_x} \mapsto \mathbb{R}^{N_x} & \text{is the autoencoder function.} \\
    \eta \in \{1, 2\} & \text{is the taget norm.} \\
    v \in \mathbb{R}^{N_x} & \text{is a sample of dimension} \ N_x.
    \end{cases}
    $$

2.  Wasserstein distance:
    $$
    W_1(\mu_1, \mu_2) = \inf_{\gamma \in \Gamma(\mu_1, \mu_2)} \ \mathbb{E}_{(x_1, x_2) \sim \gamma}[|x_1 - x_2|]
    $$
    where $\mu_1$ and $\mu_2$ are two distributions of auto-encoder losses, $\Gamma(\mu_1, \mu_2)$ is the set all of couplings of $\mu_1$ and $\mu_2$, and $m_1, m_2 \in \mathbb{R}$ are their respective means.

3.  Using Jensen’s inequality, we can derive a lower bound to $W_1(\mu_1, \mu_2)$:
    $$
    \inf \ \mathbb{E}[|x_1 - x_2|] \ge \inf \ |\mathbb{E}[x_1 - x_2]| = |m_1 - m_2|
    $$

4.  Objective function:

    -   $D$:
        $$
        \mathcal{L}_D = \mathcal{L}(x;\theta_D) - \mathcal{L}(G(z_D;\theta_G);\theta_D)
        $$

    -   $G$:
        $$
        \mathcal{L}_G = -\mathcal{L}_D
        $$
        where $\theta_D$ and $\theta_G$ are parameters for $D$ and $G$, respectively, each updated by minimizing the losses $\mathcal{L}_D$ and $\mathcal{L}_G$, and $z_D$ and $z_G$ are sample from $z$.
        $\mu_1$ is the distribution of the loss $\mathcal{L}(x)$, where $x$ are real samples, and $\mu_2$ is the distribution of the loss $\mathcal{L}(G(z))$, and $z \in [-1, 1]^{N_z}$ are uniform random samples of dimension $N_z$.

5.  Equilibrium:

    -   Equilibrium when:
        $$
        \mathbb{E}[\mathcal{L}(x)] = \mathbb{E}[\mathcal{L}(G(z))]
        $$

    -   If we generate samples that can NOT be distinguished by the discriminator from real ones, the distribution of their errors should be the same, including their expected error.

    -   Hyper-parameter $\gamma$:
        $$
        \gamma = \frac{\mathbb{E}[\mathcal{L}(G(z))]}{
        \mathbb{E}[\mathcal{L}(x)]
        } \in [0, 1]
        $$

6.  BEGAN:

    -   Objective function:
        $$
        \begin{cases}
        \mathcal{L}_D = \mathcal{L}(x) - k_t\mathcal{L}(G(z_D)) & \text{for} \ \theta_D \\
        \mathcal{L}_G = \mathcal{L}(G(z_G)) & \text{for} \ \theta_G \\
        k_{t + 1} = k_t + \lambda_k(\gamma\mathcal{L}(x) - \mathcal{L}(G(z_G))) & \text{for each trainging step} \ t
        \end{cases}
        $$
        where $k_0 = 0$, $\lambda_k$ is the proportional gain for $k$, just like the learning rate for $k$, and $\lambda_k = 0.001$ in these experiments.

    -   Architecture:
        ![9.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/9.1.png)$\theta_D$ and $\theta_G$ are updated independently based on their respective losses
        with separate Adam optimizers. We typically used a batch size of $n = 16$.

    -   Convergence measure:
        $$
        \mathcal{M}_{global} = \mathcal{L}(x) + |\gamma\mathcal{L}(x) - \mathcal{L}(G(z_G))|
        $$

### Vocabulary

1.  alleviate 減輕
2.  repelling 排斥
3.  instantaneous 瞬間

## Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

>   SRGAN

1.  For an image with $C$ color channels, $I^{LR} \in W \times H \times C$ and $I^{HR}, I^{SR} \in rW \times rH \times C$, where $r$ is the downsampling factor.

2.  $$
    \hat{\theta}_G = \text{arg}\min_{\theta_G}\frac{1}{N}\sum^N_{n = 1}l^{SR}(G_{\theta_G}(I_n^{LR}), I_n^{HR})
    $$

    where $\theta_G = \{W_{1:L}; b_{1:L}\}$ denotes the weights and biases of a $L$-layer deep network and is obtained by optimizing a SR-specific loss function $l^{SR}$.
    
3.  Objective function:
    $$
    \min_{\theta_G}\max_{\theta_D}\mathbb{E}_{I^{HR} \sim p_{train}(I^{HR})}[\log D_{\theta_D}(I^{HR})] + \mathbb{E}_{I^{LR} \sim p_G(I^{LR})}[\log (1 - D_{\theta_D}(G_{\theta_G}(I^{LR})))]
    $$

4.  Perceptual loss function:
    $$
    l^{SR} = l_{X}^{SR} + 10^{-3}l_{Gen}^{SR}
    $$

    -   Pixel-wise MSE loss (original):
        $$
        l_{MSE}^{SR} = \frac{1}{r^2WH}\sum^{rW}_{x = 1}\sum^{rH}_{y = 1}(I_{x, y}^{HR} - G_{\theta_G}(I^{LR})_{x, y})^2
        $$

    -   VGG loss:
        $$
        l_{VGG/i,j}^{SR} =\frac{1}{W_{i, j}H_{i, j}}\sum^{W_{i, j}}_{x = 1}\sum^{H_{i , j}}_{y = 1}(\phi_{i, j}(I^{HR})_{x, y} - \phi_{i, j}(G_{\theta_G}(I^{LR}))_{x, y})^2 
        $$
        where ${W_{i, j}, H_{i, j}}$  denote the dimensions of the respective feature  maps within the VGG network.

    -   Adversarial loss:
        $$
        l_{Gen}^{SR} = \sum^N_{n = 1} -\log D_{\theta_D}(G_{\theta_G}(I^{LR}))
        $$
        where $D_{\theta_D}(G_{\theta_G}(I^{LR}))$ is the probability that the reconstructed image $G_{\theta_G}(I^{LR})$ is a natural HR image.

5.  Architecture:
    ![10.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/10.1.png)

## Vocabulary

1.  perceptually 感性地
2.  substantial 充實的

## Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks

>   LAPGAN

1.  Laplacian Pyramid:

    -   Coefficients $h_k$ at each level $k$ of the pyramid $\mathcal{L}(I)$:
        $$
        h_k = \mathcal{L}_k(I) = \mathcal{G}_k(I) - u(\mathcal{G}_{k + 1}(I)) = I_k - u(I_{k + 1})
        $$
        where $d(\cdot)$ is a downsampling operation, and $u(\cdot)$ is an upsampling operation. $d(I) \in j/2 \times j/2, \ u(I) \in 2j \times 2j, \ \text{where} \ I \in j \times j$. $\mathcal{G}_I = [I_0, I_1, \cdots, I_K]$, where $I_0 = I$ and $I_k$ is $k$ repeated applications of $d(\cdot)$ to $I$.

    -   $$
        I_k = u(I_{k + 1}) + h_k
        $$

        where $I_K = h_K$ and the reconstructed image being $I = I_o$.

    -   In other words, starting at the coarsest level, we repeatedly upsample and add the difference image $h$ at the next finer level until we get back to the full resolution image.

2.  LAPGAN:

    -   $$
        \tilde{I}_k = u(\tilde{I}_{k + 1}) + \tilde{h}_k = u(\tilde{I}_{k + 1}) + G_k(z_k, u(\tilde{I}_{k + 1}))
        $$

        where $\tilde{I}_{K + 1} = 0$. The generative models $\{G_0, \cdots, G_k\}$. Noise vector $z_K: \tilde{I}_K = G_K(z_K)$.

    -   $$
        \tilde{h}_k = G_k(z_k, u(I_{k + 1}))
        $$

    -   Architecture:
        ![11.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/11.1.png)

### Vocabulary

1.  decimate 抽取

## Energy-Based Generative Adversarial Networks

>   EBGAN

1.  Objective functions:

    -   $G$:
        $$
        \mathcal{L}_G(z) = D(G(z))
        $$

    -   $D$:
        $$
        \mathcal{L}_D(x, z) = D(x) + [m - D(G(z))]^+
        $$
        where $[\cdot]^+ = \max(0, \cdot)$. Minimizing $\mathcal{L}_G$ with respect to the parameters of $G$ is similar to maximizing the second term of $\mathcal{L}_D$. It has the same minimum but non-zero gradients when $D(G(z)) \ge m$.

2.  Optimality:

    -   $$
        \begin{cases}
        V(G, D) = \int_{x, z}\mathcal{L}_D(x,z)p_{data}(x)p_z(z)dxdz \\
        U(G, D) = \int_{z}\mathcal{L}_G(z)p_z(z)dz
        \end{cases}
        $$

        We train $D$ to minimize $V$ and train $G$ to minimize $U$.

    -   Nash equilibrium:
        $$
        \begin{cases}
        V(G^*, D^*) \le V(G^*, D) & \forall D \\
        U(G^*, D^*) \le U(G, D^*) & \forall G
        \end{cases}
        $$

    -   If $(D^*, G^*)$ is a Nash equilibrium of the system, then $p_G^* = p_{data}$ almost everywhere, and $V^*(D^*, G^*) = m$.

    -   If $(D^*, G^*)$ is a Nash equilibrium of the system, then there exists a constant $\gamma \in [0, m]$ such that $D^*(x) = \gamma$ almost everywhere.

3.  Architecture:

    -   Using auto-encoder:
        $$
        D(x) = ||Dec(Enc(x)) - x||
        $$

        ![12.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/12.1.png)

4.  Repelling regularizer (Pulling-away Term, PT):
    $$
    f_{PT}(S) = \frac{1}{N(N - 1)}\sum_i\sum_{j \ne i}(\frac{S_i^TS_j}{||S_i||||S_j||})^2
    $$
    where $S \in \mathbb{R}^{s \times N}$ denotes a batch of sample representations taken from the encoder output layer. PT is used in the generator loss but NOT in the discriminator loss.

### Vocabulary

1.  postulate 假定
2.  rationale 基本原理

## Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

>   CycleGAN

1.  Architecture:
    ![13.1](https://github.com/Oscarshu0719/paper-notes/blob/master/img/13.1.png)

2.  Adversarial losses:

    -   $D_Y$:

    $$
    \mathcal{L}_{GAN} (G, D_Y, X, Y) = \mathbb{E}_{y \sim p_{data}(y)}[\log D_Y(y)] + \mathbb{E}_{x \sim p_{data}(x)}[\log (1 - D_Y(G(x)))]
    $$

    -   $D_X$:
        $$
        \mathcal{L}_{GAN}(F, D_X, Y, X) = \mathbb{E}_{x \sim p_{data}(x)}[\log D_X(x)] + \mathbb{E}_{y \sim p_{data}(y)}[\log (1 - D_X(F(y)))]
        $$
        where $D_X$ aims to distinguish between images $\{x\}$ and translated images $\{F(y)\}$, and $D_Y$ aims to discriminate between $\{y\}$ and $\{G(x)\}$.

3.  Cycle consistency loss:
    $$
    \mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x \sim p_{data}(x)}[||F(G(x)) - x||_1] + \mathbb{E}_{y \sim p_{data}(y)}[||G(F(y)) - y||_1]
    $$

4.  Full objective function:
    $$
    \mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{GAN}(G, D_Y, X, Y) + \mathcal{L}_{GAN}(F, D_X, Y, X) + \lambda\mathcal{L}_{cyc}(G, F) \\
    $$
    We aim to solve:
    $$
    G^*, F^* = \text{arg}\min_{G, F}\max_{D_X, D_Y}\mathcal{L}(G, F, D_X, D_y)
    $$

5.  Implementation:

    -   $G$:
        $$
        \mathbb{E}_{x \sim p_{data}(x)}[(D(G(x)) - 1)^2]
        $$

    -   $D$:
        $$
        \mathbb{E}_{y \sim p_{data}(y)}[(D(y) - 1)^2] + \mathbb{E}_{x \sim p_{data}(x)}[D(G(x))^2]
        $$
        for $\mathcal{L}_{GAN}(G, D, X, Y)$. Using least-square loss got better stability.

### Vocabulary

1.  incentivize 激勵

## Progressive Growing of GANs for Improved Quality, Stability, and Variation

>   PGGAN

1.  Minibatch Standard Deviation (MSD) increasing variation:

    -   We first compute the standard deviation for each feature in each spatial location over the minibatch. 
    -   We then average these estimates over all features and spatial locations to arrive at a single value. 
    -   We replicate the value and concatenate it to all spatial locations and over the minibatch, yielding one additional (constant) feature map. This layer could be inserted anywhere in the discriminator, but we have found it best to insert it towards the end.

2.  Normalization:

    -   Equalized learning rate:
        $$
        \hat{w}_i = w_i / c
        $$
        where $w_i$ are the weights and $c$ is the per-layer normalization constant from He's initializer.

        -   He's initializer:
            $$
            \frac{1}{2}n_l \text{Var}[w_l] = 1
            $$
            where $w_l$ are weights of layer $l$ and $n_l$ is the number is weights at layer $l$.

    -   Pixelwise feature vector normalization in $G$:
        $$
        b_{x, y} = a_{x, y} / \sqrt{\frac{1}{N}\sum^{N - 1}_{j = 0}(a_{x, y}^j)^2 + \epsilon}
        $$
        where $\epsilon = 10^{-8}$, $N$ is the number of feature maps, and $a_{x, y}$ and $b_{x, y}$ are the original and normalized feature vertor in pixel $(x, y)$, respectively.

### Vocabulary

1.  unprecedented 前所未有的
2.  prone 易於
3.  deviate 偏離

## Conditional Image Synthesis with Auxiliary Classifier GANs

>   ACGAN

1.  

### Vocabulary

1.  

## Improving the Improved Training of Wasserstein GANs: A Consistency Term and Its Dual Effect

>   CTGAN

1.  

## Spectral Normalization for Generative Adversarial Networks

>   SNGAN

1.  

## Wasserstein Divergence for GANs

>   WGAN-div

1.  

## InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets

>   InfoGAN

1.  

## Autoencoding beyond pixels using a learned similarity metric

>   VAE-GAN

1.  

## Adversarial Feature Learning

>   BiGAN

1.  

## Triple Generative Adversarial Nets

>   Triple-GAN

1.  

## Adversarial Ranking for Language Generation

>   RankGAN

1.  

## XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings

>   XGAN

1.  