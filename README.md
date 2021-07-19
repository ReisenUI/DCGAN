<center><font size="5">浙江大学暑期夏令营学习笔记</font></center>

<br/>
<center>作者：黄毓儒</center>
<center>本科院校：浙江工商大学</center>
<center>专业：计算机科学与技术</center>

<br/>
<center><font size="3">前言</font><br/></center><font size="2">
&emsp;&emsp;本文档用以记录浙江大学暑期夏令营期间项目扩展的学习经历，由于本科期间项目是在百度飞桨AI Studio上实现的GANs网络生成图片的应用，代码并非完全参透且实验效果不好，<strong>故从零开始学习PyTorch以及阅读GAN和DCGAN论文并进行实验，期间对比了两者之间的联系与区别，并从DCGAN开始进一步完成选题——DCGAN网络生成图片实验。</strong></font>
<br/><br/>

## 一、GAN和DCGAN
### 1、GANs(Generative Adversarial Nets)
#### 1.1 GANs背景介绍

&emsp;&emsp;Generative Adversarial Nets，翻译为生成式对抗网络，由`Ian J. Goodfellow`于2014年在 [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) 中提出。其背后的原理比较朴素，是两个模型进行一个非零和博弈来达到想要的效果。在GANs网络中，有两个模型——生成模型（Generative Model，G)和判别模型（Discriminative Model，D）组成。下面以生成图片的模型举例：
- 生成模型 **G**，是一个生成图片的网络，通过输入一个随机的噪声z，然后通过噪声来生成图片，生成的图片记作 **G(z)**。
- 判别模型 **D**，是判断所输入的图片是否“真实”的网络，即是否由生成模型生成。输入一个图片 **x**，输出 **D(x)** 则代表 **x** 为真实图片的概率，概率接近于1则说明判别模型认为这是一个真实的图片，反之若为0，则不是真实的。

&emsp;&emsp;在训练的过程中，**G** 的工作即生成一张图片，它足以欺骗 **D** 以至于该图片被判定为真实的概率尽可能高；反之 **D** 的工作即尽可能提高自身识别虚假数据的能力。因此双方形成了一个 **零和博弈** ，与此同时，双方的能力也在不断的博弈过程中不断提高，最理想的情况下，应有 **D(G(z)) = 0.5** 即 **D** 无法区分真实图片和 **G** 产生的图片，若要判别，也只能从一半的概率中随机抽取。

&emsp;&emsp;正如原文中的例子，**G** 比作伪钞生产者，他每次的生成假币不经检测就使用；而 **D** 比作警察，目的是找出 **G** 生成的假币。二者在不断的竞争中能力都得到了提升。（虽然例子不符合常理但是却很形象）

[![GAN模型](https://z3.ax1x.com/2021/07/19/WYnKNq.md.jpg)](https://imgtu.com/i/WYnKNq)

&emsp;&emsp;因此，在GAN网络中，我们需要同时对两个网络进行训练。使用 **反向传播和dropout方法**,无需近似推理和马尔科夫链。

#### 1.2 相关工作
&emsp;&emsp;原文中提到了若干随机生成领域内相关的模型：
1. Boltzmann machines (玻尔兹曼机)
   - 玻尔兹曼机的似然函数难以处理，因此需要大量近似；GANs需要精确的反向传播。
   - GANs消除了生成随机网络的马尔科夫链
2. variational autoencoders (VAEs)
   - VAE不能有离散的潜在变量；GANs不能建模离散数据
   - VAE预先知道图像的密度函数（超参数）；GANs不知道
   - VAE适合学习结构良好的潜在空间；GANs图像可能更逼真，但潜在空间可能没那么多结构和连续性
3. Noise-contrastive estimation (NCE)
   - NCE通过学习使模型有用于从固定的噪声分布中区分数据
4. predictability minimization
   - PM也是两个神经网络
   - 训练标准不同，PM是一个正则化器，鼓励神经网络潜在单元在完成其他任务时在统计上独立，不是一个主要的学习标准；而GANs网络之间的竞争是唯一训练标准
   - 竞争性质不同，PM的一个网络试图让输出近似，另一个网络试图让输出原理，针对的是这一组标量；GANs中一个网络生成高维丰富的变量提交给另一个，并试图让他不知道该怎么处理这一数据
   - 学习过程的规范不同，PM被描述为一个目标函数的优化问题，学习接近目标函数的最小值；GANs基于极小化极大博弈，博弈结束于一个鞍点，即一个玩家策略最小值和另一玩家策略最大值

#### 1.3 对抗网络
>符号定义：<br/>
&emsp;***data：*** 真实数据<br/>
&emsp;***p<sub>data</sub>：*** 真实数据的分布<br/>
&emsp;***z：*** 噪音<br/>
&emsp;***x：*** 特征（包括真实样本和生成的样本）<br/>
&emsp;***p<sub>z</sub>：*** 原始噪音的分布<br/>
&emsp;***G(z;θ<sub>g</sub>)：*** 生成映射函数，将噪声映射到新的数据空间<br/>
&emsp;***D(x;θ<sub>d</sub>)：*** 判别映射函数，输出是图片为**真实图片概率**

GAN的目标函数：<br/>

![minimax-GAN](https://z3.ax1x.com/2021/07/19/WYugQU.md.png)

&emsp;&emsp;一方面 **D** 希望自己能够明确区分真实样本和生成样本，即让D(x)尽可能大，同时也希望 D(G(z)) 尽可能小，总体使 V(D,G) 尽可能大；<br/>
&emsp;&emsp;另一方面 **G** 希望 D(G(z)) 尽可能大，导致 V(D,G) 尽可能小。<br/>
&emsp;&emsp;两个模型对抗的理想结果是达到**全局最优** <br/>
&emsp;&emsp;**注意：** 对于判别器而言，使 **1-D(G(z))** 尽可能小可能导致初始时刻不利于两个模型的判别，因为此时两个模型能力都十分弱，因此可以改为使 **D(G(z))** 尽可能大。

![G和D的对抗](https://z3.ax1x.com/2021/07/19/WYKnf0.md.png)

&emsp;&emsp;如图所示，黑色曲线是真实样本的概率分布函数，绿色曲线是虚假样本的概率分布函数，蓝色是判别器 **D** 的输出，值越高说明越可能是真实数据。最下方的亮条平行线的含义如上述定义，代表了从噪音空间映射到生成出的图片。<br/>
&emsp;&emsp;在模型训练初期，判别器可以十分轻松的判断出哪个是真实的 x ，随着训练进度的不断推进，**D** 和 **G** 的能力都在不断增强，但 **D** 也越来越难判断生成的样本了。<br/>
&emsp;&emsp;**最优状态：** 当上述绿线和黑线重合时，模型达到最优状态，即使是 **D** 也无法区分哪个是真实数据，所以 **D** 对于每个样本的输入都是0.5。

#### 1.4 证明（从简）
##### 1.4.1 KL散度
&emsp;&emsp;KL散度的是统计中的概念，衡量两种概率分布的相似程度，越小则表示两种概率分布越接近。（此处以连续型随机变量举例，离散型类似）

<div align=center>
<img src="https://z3.ax1x.com/2021/07/19/WYQnMT.png"/>
</div>

##### 1.4.2 极大似然估计
&emsp;&emsp;从真实数据分布里取m个观察值，即m个概率 P<sub>G</sub>(x<sup>i</sup>;θ) ，则其极大似然估计为：

<div align=center>
<img src="https://z3.ax1x.com/2021/07/19/WYlpf1.png"/>
</div>

&emsp;&emsp;求得使L获得最大值的θ即可。

##### 1.4.3 综合上述（下述证明看得懂但不会说...）

<div align=center>
<img src="https://z3.ax1x.com/2021/07/19/WYlrBF.png">
<img src="https://z3.ax1x.com/2021/07/19/WYlDnU.png"><br/><br/><br/>
<img src="https://z3.ax1x.com/2021/07/19/WYl0XT.png"><br/>
<img src="https://z3.ax1x.com/2021/07/19/WYlwcV.png">
</div>

##### 1.4.4 全局最优
&emsp;&emsp;证明从略，大致思路为：假设全局最优情况即判别器无法区分真实样本和生成的样本，即每个概率都是0.5，基于这一前提，给定生成器 **G** 最大化两者的 V(D, G) ，这一数值评估了 P<sub>G</sub> 和 P<sub>data</sub> 之间的差异。同时将 V(D, G) 展开为积分形式。带入前提预设的0.5概率，*C(G)* = *D*-max V(D, G) 可以算出最小值为 **-log4** 

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WY8d1I.png"><br/>
</div>

-----------------

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WY8a9A.png">
</div>

&emsp;&emsp;其中由于KL散度非负，可以很容易得出最小可能取值为 **-log4** （其余证明跳过，此处只介绍了判别器相关）

#### 1.5 模型学习
##### 1.5.1 判别器学习
&emsp;&emsp;判别器分别从生成器和真实样本中获取数据，生成器会根据随机噪声向量产生图片，从生成器中获得的数据标注为 **0** ，真实样本中获得的数据标注为 **1**，两者放入判别器中可得到对应判别结果的得分，以判别结果反过来对其进行训练。

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYG0PJ.png">
</div>

##### 1.5.2 生成器学习
&emsp;&emsp;我们将生成器中的数据标记为 **1**，即生成器认为自己的图片是真的，但反过来传输到判别器当中，模型的打分可能会很低，因而产生了误差。因此在训练生成器的过程中，重要的一点就是保持判别器的网格不发生改变。

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYGd54.png">
</div>

##### 1.5.3 伪代码及注意事项
&emsp;&emsp;伪代码直接参考原文

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYJCLV.png">
</div>

&emsp;&emsp;值得注意的是，原文中训练判别器 **k** 次后再训练生成器，但是一般而言，此处的 **k** 取1。即一次循环只训练一次判别器和一次生成器（至少我做实验的时候是这样的，虽然不知道k取高会如何也没实验过）

#### 1.6 代码实现及结果呈现
>环境配置：<br/>
&emsp;***cuda：*** 10.1<br/>
&emsp;***pytorch:*** 1.7.1<br/>
&emsp;***torchvision:*** 0.8.2<br/>
&emsp;***cudatoolkit：*** 10.1<br/>
数据集：<br/>
&emsp;MNIST<br/>
参数配置：<br/>
&emsp;***epoches：*** 200<br/>
&emsp;***batch_size：*** 128<br/>
&emsp;***optimizer：*** Adam with learning rate = 0.0002<br/>
&emsp;***beta：*** 0.5—Decay of first order momentum of gradient<br/>

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYJ4TU.png">
<small>判别器和生成器</small>
</div>

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYJTfJ.png">
<small>训练过程</small>
</div>

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYJzkD.png">
<img src="https://z3.ax1x.com/2021/07/20/WYJvTO.png">
<img src="https://z3.ax1x.com/2021/07/20/WYYSte.png">
<br/>
<small>结果展示——出现了模式坍塌</small>
</div>

#### 1.7 GAN的优点和缺陷
优点：
   1. 看上去比其他模型产生了更好的样本，图像更锐利清晰等。
   2. 任何生成器网络和任何鉴别器都有用。
   3. 无需马尔科夫链采样，无需学习过程中推断，回避了近似计算棘手的问题。
   
缺点：
   1. 理论认为GAN在纳什均衡上有出色的表现，但很难保证模型最后会到达纳什均衡。
   2. 难以训练，会出现模式坍塌问题。
   3. 无需预先建模，整个生成模型是个黑盒，模型自由不可控。

<br/><br/><br/>

### 2. DCGAN（Deep Convolutional Generative Adversarial Networks）
#### 2.1 介绍
&emsp;&emsp;DCGAN的判别器和生成器同时使用了卷积神经网络（CNN）来代替原先GAN中的多层感知机，并增加了结构上的约束使之在大多数情况下训练更加稳定。如下图所示：

![WYYa9J.png](https://z3.ax1x.com/2021/07/20/WYYa9J.png)

#### 2.2 主要改进
1. 使用跨步卷积（判别器）和fractionally-strided（生成器）来代替池化层
2. 使用BatchNorm——同时在生成器和判别器中
3. 去除隐藏层的全连接层
4. 在判别器中的每一层使用LeakyReLU

#### 2.3 一些特性
1. 训练过的判别器具有较强竞争性
2. 生成器具有有趣的向量算术特性，允许操作样本的多语义性质
3. CNNs部件的可视化
4. 若对一个模型训练时间较长，偶尔会出现模型坍塌的现象
5. 个人感觉，和简单的模式识别器不同，我们不是看判别器或生成器的loss，而是最后把判别器作为一个分离器的特征提取器（冻结权重不更新）评估此时分离器的表现，以此来衡量最终GAN训练的好坏。

#### 2.4 代码实现及结果呈现
##### 2.4.1 MNIST数据集
>环境配置：<br/>
&emsp;***cuda：*** 10.1<br/>
&emsp;***pytorch:*** 1.7.1<br/>
&emsp;***torchvision:*** 0.8.2<br/>
&emsp;***cudatoolkit：*** 10.1<br/>
数据集：<br/>
&emsp;MNIST<br/>
参数配置：<br/>
&emsp;***epoches：*** 200<br/>
&emsp;***batch_size：*** 128<br/>
&emsp;***optimizer：*** Adam with learning rate = 0.0002<br/>
&emsp;***beta：*** 0.5—Decay of first order momentum of gradient<br/>
&emsp;***LeakyReLU：*** 倾斜度为0.2

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYYhjI.png">
<small>判别器和生成器</small>
</div>

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYYfgA.png">
<small>判别器和生成器</small>
</div>

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYY7E8.png">
<img src="https://z3.ax1x.com/2021/07/20/WYYoHf.png">
<img src="https://z3.ax1x.com/2021/07/20/WYYHUS.png">
<br/>
<small>结果展示——个人感觉效果略好于原始GAN</small>
</div>

##### 2.4.2 CelebA数据集及自制动漫人物头像数据集
>环境配置：<br/>
&emsp;同上<br/>
数据集：<br/>
&emsp;CelebA名人头像，动漫人物头像（爬虫和脸部识别）<br/>
参数配置：<br/>
&emsp;具体情况参见仓库内代码（为了减少数据量，此处没有上传数据集）

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYYXgs.jpg">
<small>CelebA名人脸部数据集（右侧为生成器产生的图片）</small>
</div>

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYYO3j.jpg">
<small>自制动漫人物脸部数据集（右侧为生成器产生的图片）</small>
</div>

<div align=center>
<img src="https://z3.ax1x.com/2021/07/20/WYYjvn.jpg">
<small>G_D_Loss值（蓝色为G，橙色为D）</small>
</div>