---
layout: post
title: "Convolutional Neural Network Architecture"
author: "HzCeee"
categories: Machine Learning
tags: [Neural Network]
image:
  feature: 
  teaser:
  credit:
  creditlink:
---

# AlexNet

- Parameters

```
INPUT [227x227x3] 

CONV1 [55x55x96]: 96 11x11x3 filters at stride 4, pad 0
--- OUTPUT VOLUME SIZE = (227+2x0-11)/4+1 = 55 ---
--- TOTAL NUMBER OF PARAMETERS = 96x(11x11x3) = 34,848 ---
MAX POOL1 [27x27x96]: 3x3x1 filters at stride 2
--- OUTPUT VOLUME SIZE = (55-3)/2+1 = 27 ---
NORM1 [27x27x96]: Normalization layer

CONV2 [27x27x256]: 256 5x5x96 filters at stride 1, pad 2
--- OUTPUT VOLUME SIZE = (27+2x2-5)/1+1 = 27 ---
--- TOTAL NUMBER OF PARAMETERS = 256x(5x5x96) = 614,400 ---
MAX POOL2 [13x13x256]: 3x3x1 filters at stride 2
--- OUTPUT VOLUME SIZE = (27-3)/2+1 = 13 ---
NORM2 [13x13x256]: Normalization layer

CONV3 [13x13x384]: 384 3x3x256 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (13+2x1-3)/1+1 = 13 ---
--- TOTAL NUMBER OF PARAMETERS = 384x(3x3x256) = 884,736 ---

CONV4 [13x13x384]: 384 3x3x384 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (13+2x1-3)/1+1 = 13 ---
--- TOTAL NUMBER OF PARAMETERS = 384x(3x3x384) = 1,327,104 ---

CONV5 [13x13x256]: 256 3x3x384 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (13+2x1-3)/1+1 = 13 ---
--- TOTAL NUMBER OF PARAMETERS = 256x(3x3x384) = 884,736 ---
MAX POOL3 [6x6x256]: 3x3x1 filters at stride 2
--- OUTPUT VOLUME SIZE = (13-3)/2+1 = 6 ---

FC6 [1x1x4096]: 4096 neurons
--- TOTAL NUMBER OF PARAMETERS = 4096*(6*6*256) = 37,748,736 ---

FC7 [1x1x4096]: 4096 neurons
--- TOTAL NUMBER OF PARAMETERS = 4096*4096 = 16,777,216 ---

FC8 [1x1x1000]: 1000 neurons (class scores)
--- TOTAL NUMBER OF PARAMETERS = 1000*4096 = 4,096,000 ---
```

- AlexNet Architecture

![AlexNet Architecture](https://ws1.sinaimg.cn/large/006tKfTcgy1fjisn4myktj30h30k0whk.jpg)

# VGGNet

- Parameters

```
INPUT [224x224x3]

CONV3-64 [224x224x64]: 64 3x3x3 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (224+2x1-3)/1+1 = 224 ---
--- TOTAL NUMBER OF PARAMETERS = 64x(3x3x3) = 1,728 ---
CONV3-64 [224x224x64]: 64 3x3x64 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (224+2x1-3)/1+1 = 224 ---
--- TOTAL NUMBER OF PARAMETERS = 64x(3x3x64) = 36,864 ---
POOL2 [112x112x64]: 2x2x1 filters at stride 2
--- OUTPUT VOLUME SIZE = (224-2)/2+1 = 112 ---

CONV3-128 [112x112x128]: 128 3x3x64 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (112+2x1-3)/1+1 = 112 ---
--- TOTAL NUMBER OF PARAMETERS = 128x(3x3x64) = 73,728 ---
CONV3-128 [112x112x128]: 128 3x3x128 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (112+2x1-3)/1+1 = 112 ---
--- TOTAL NUMBER OF PARAMETERS = 128x(3x3x128) = 147,456 ---
POOL2 [56x56x128]: 2x2x1 filters at stride 2
--- OUTPUT VOLUME SIZE = (112-2)/2+1 = 56 ---

CONV3-256 [56x56x256]: 256 3x3x128 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (56+2x1-3)/1+1 = 56 ---
--- TOTAL NUMBER OF PARAMETERS = 256x(3x3x128) = 294,912 ---
CONV3-256 [56x56x256]: 256 3x3x256 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (56+2x1-3)/1+1 = 56 ---
--- TOTAL NUMBER OF PARAMETERS = 256x(3x3x256) = 589,824 ---
CONV3-256 [56x56x256]: 256 3x3x256 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (56+2x1-3)/1+1 = 56 ---
--- TOTAL NUMBER OF PARAMETERS = 256x(3x3x256) = 589,824 ---
POOL2 [28x28x256]: 2x2x1 filters at stride 2
--- OUTPUT VOLUME SIZE = (56-2)/2+1 = 28 ---

CONV3-512 [28x28x512]: 512 3x3x256 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (28+2x1-3)/1+1 = 28 ---
--- TOTAL NUMBER OF PARAMETERS = 512x(3x3x256) = 1,179,648 ---
CONV3-512 [28x28x512]: 512 3x3x512 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (28+2x1-3)/1+1 = 28 ---
--- TOTAL NUMBER OF PARAMETERS = 512x(3x3x512) = 2,359,296 ---
CONV3-512 [28x28x512]: 512 3x3x512 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (28+2x1-3)/1+1 = 28 ---
--- TOTAL NUMBER OF PARAMETERS = 512x(3x3x512) = 2,359,296 ---
POOL2 [14x14x512]: 2x2x1 filters at stride 2
--- OUTPUT VOLUME SIZE = (28-2)/2+1 = 14 ---

CONV3-512 [14x14x512]: 512 3x3x512 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (14+2x1-3)/1+1 = 14 ---
--- TOTAL NUMBER OF PARAMETERS = 512x(3x3x512) = 2,359,296 ---
CONV3-512 [14x14x512]: 512 3x3x512 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (14+2x1-3)/1+1 = 14 ---
--- TOTAL NUMBER OF PARAMETERS = 512x(3x3x512) = 2,359,296 ---
CONV3-512 [14x14x512]: 512 3x3x512 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (14+2x1-3)/1+1 = 14 ---
--- TOTAL NUMBER OF PARAMETERS = 512x(3x3x512) = 2,359,296 ---
POOL2 [7x7x512]: 2x2x1 filters at stride 2
--- OUTPUT VOLUME SIZE = (14-2)/2+1 = 7 ---

FC [1x1x4096]: 4096 neurons
--- TOTAL NUMBER OF PARAMETERS = 4096*(7*7*512) = 102,760,448 ---

FC [1x1x4096]: 4096 neurons
--- TOTAL NUMBER OF PARAMETERS = 4096*4096 = 16,777,216 ---

FC [1x1x1000]: 1000 neurons
--- TOTAL NUMBER OF PARAMETERS = 1000*4096 = 4,096,000 ---
```

- VGGNet Architecture

![VGGNet Architecture](https://ws3.sinaimg.cn/large/006tKfTcgy1fjisf1r28wj30qq0fs3zx.jpg)

# GoogLeNet

- Inception

![Inception V1](https://ws4.sinaimg.cn/large/006tKfTcgy1fjicppkh89j310e0iugne.jpg)

```
INPUT [28x28x192]

CONV1-64 [28x28x64]: 64 1x1x192 filters at stride 1, pad 0
--- OUTPUT VOLUME SIZE = (28+2x0-1)/1+1 = 28 ---
--- TOTAL NUMBER OF PARAMETERS = 64x(1x1x192) = 12,288 ---

CONV1-96 [28x28x96]: 96 1x1x192 filters at stride 1, pad 0
--- OUTPUT VOLUME SIZE = (28+2x0-1)/1+1 = 28 ---
--- TOTAL NUMBER OF PARAMETERS = 96x(1x1x192) = 18,432 ---
CONV3-128 [28x28x128]: 128 3x3x96 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (28+2x1-3)/1+1 = 28 ---
--- TOTAL NUMBER OF PARAMETERS = 128x(3x3x96) = 110,592 ---

CONV1-16 [28x28x16]: 16 1x1x192 filters at stride 1, pad 0
--- OUTPUT VOLUME SIZE = (28+2x0-1)/1+1 = 28 ---
--- TOTAL NUMBER OF PARAMETERS = 16x(1x1x192) = 3,072 ---
CONV5-32 [28x28x32]: 32 5x5x16 filters at stride 1, pad 2
--- OUTPUT VOLUME SIZE = (28+2x2-5)/1+1 = 28 ---
--- TOTAL NUMBER OF PARAMETERS = 32x(5x5x16) = 12,800 ---

POOL3 [28x28x192]: 3x3x1 filters at stride 1, pad 1
--- OUTPUT VOLUME SIZE = (28+2x1-3)/1+1 = 28 ---
CONV1-32 [28x28x32]: 32 1x1x192 filters at stride 1, pad 0
--- OUTPUT VOLUME SIZE = (28+2x0-1)/1+1 = 28 ---
--- TOTAL NUMBER OF PARAMETERS = 32x(1x1x192) = 6,144 ---

DepthConcat [28x28x256]
--- OUTPUT DEPTH SIZE = 64+128+32+32 = 256 ---

=== TOTAL NUMBER OF PARAMETERS = 12,288+18,432+110,592+3,072+12,800+6,144 = 163,328 ===
```

- GoogLeNet Architecture

![GoogLeNet Architecture](https://ws4.sinaimg.cn/large/006tKfTcgy1fjik3lpld7j31kw10j134.jpg)

# ResNet

- ResNet Architecture

![ResNet Architecture](https://ws4.sinaimg.cn/large/006tKfTcgy1fjivxjj7hlj31kw0lv44i.jpg)