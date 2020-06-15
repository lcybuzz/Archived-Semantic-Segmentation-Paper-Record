# Personal-Semantic-Segmentation-Paper-Record
# Under construction!
# Table of Contents
- [Deep Learning Methods](#deep-learning-methods)
  - [Semantic Segmentation](#semantic-segmentation)
  - [Panoptic Segmentation](#panoptic-segmentation)
  - [Foreground-background Segmenation](#foreground-background-segmenation)
  - [Transfer related](#transfer-related)
  - [Knowledge distillation](#knowledge-distillation)
  - [Other Interesting Papers](#other-interesting-papers)
- [Traditional Classical Methods](#traditional-classical-methods)
- [Datasets](#datasets)
- [Leaderboards](#leaderboards)
- [Sources-Lists](#sources-lists)
	
# Deep Learning Methods

## Semantic Segmentation

### **FCN ★★★**
**[Paper]** Learning a Deep Convolutional Network for Image Super-Resolution <Br>
**[Year]** CVPR 2015<Br>
**[Authors]** Evan Shelhamer, Jonathan Long, Trevor Darrell<Br>
**[Pages]**<Br>
	 https://github.com/shelhamer/fcn.berkeleyvision.org (official)<Br>
	 https://github.com/MarvinTeichmann/tensorflow-fcn (tensorflow)<Br>
	 https://github.com/wkentaro/pytorch-fcn (pytorch)<Br>
**[Description]**<Br>
1) 首篇（？）使用end-to-end CNN实现Semantic Segmentation，文中提到FCN与提取patch逐像素分类是等价的，但FCN中相邻patch间可以共享计算，因此大大提高了效率
2) 把全连接视为一种卷积
3) 特征图通过deconvolution（初始为bilinear interpolation)上采样，恢复为原来的分辨率
4) 使用skip connection改善coarse segmentation maps
	
	
### **U-Net ★**
**[Paper]** U-Net: Convolutional Networks for Biomedical Image Segmentation<Br>
**[Year]** MICCAI 2015<Br>
**[Authors]** 	Olaf Ronneberge, Philipp Fischer, 	Thomas Brox<Br>
**[Pages]**<Br>
	https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ <Br>
	https://github.com/orobix/retina-unet <Br>
**[Description]**<Br>
1) encoder-decoder结构，encode设计参考的是FCN，decode阶段将encode阶段对应的特征图与up-conv的结果concat起来
2) 用于医学图像分割，数据集小，因此做了很多data augmentation，网络结构也较为简单


### **zoom-out ★** 
**[Paper]** Feedforward semantic segmentation with zoom-out features<Br>
**[Year]** CVPR 2015<Br>
**[Authors]** Mohammadreza Mostajabi, Payman Yadollahpour, Gregory Shakhnarovich<Br>
**[Pages]** https://bitbucket.org/m_mostajabi/zoom-out-release <Br>
**[Description]**<Br>
1) 以超像素为最小单位，逐步zoom out提取更大尺度的信息，zoom out特征是从CNN不同层提取的特征得到的
2) 特征在超像素的范围内进行average pooling，并concat不同level的特征得到该超像素最后的特征向量。用样本集中每一类出现频率的倒数加权loss。

### **Dilated Convolution★** 
**[Paper]**  Multi-Scale Context Aggregation By Dilated Convolutions<Br>
**[Year]** ICLR 2016<Br>
**[Authors]** 	[Fisher Yu](http://www.yf.io/)  ,	[Vladlen Koltun](http://vladlen.info/)<Br>
**[Pages]** https://github.com/fyu/dilation <Br>
**[Description]**<Br>
1) 系统使用了dilated convulution，其实现已被Caffe收录
	 
### **Understanding Convolution ★☆** 
**[Paper]** Understanding Convolution for Semantic Segmentation<Br>
**[Year]** WACV 2018<Br>
**[Authors]** [Panqu Wang](http://acsweb.ucsd.edu/~pawang/homepage_PhD/index.html), Pengfei Chen, Ye Yuan, Ding Liu, Zehua Huang, [Xiaodi Hou](http://www.houxiaodi.com/), [Garrison Cottrell](https://cseweb.ucsd.edu/~gary/)<Br>
**[Pages]** https://github.com/TuSimple/TuSimple-DUC <Br>
**[Description]**<Br>
1) 针对语义分割任务, 为encoding和decoding分别设计了DUC和HDC两个结构, 其设计有被deeplab v3借鉴. <Br>
2) decoding阶段: DUC(dense upsampling convolution), 类似于超分辨和instance分割的一些做法, 令最后阶段特征图的每个channel代表上采样后相应位置的预测结果. <Br>
3) encoding阶段: HDC(hybrid dilated convolution), 交替地进行不同dilation rate的卷积, 避免棋盘效应. <Br>
	
### **DeepLab ★★** 
**[Paper]** Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs<Br>
**[Year]** ICLR 2015<Br>
**[Authors]** [Liang-Chieh Chen](http://liangchiehchen.com/), George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille<Br>
**[Pages]** https://bitbucket.org/deeplab/deeplab-public<Br>
**[Description]**    <Br>
1) 在保证感受野大小的同时，输出dense feature。做法是把VGG16后两个pool stride设置为1，用Hole算法(也就是Dilation卷积)控制感受野范围
2) 输出用全局CRF后处理，一元项为pixel的概率，二元项为当前pixel与图像中除自己外的每个pixel的相似度，考虑颜色和位置，使用高斯核。全连接CRF参考[Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/)
3) 与FCN相似，也使用了多尺度预测

**[Paper]** Weakly- and Semi-Supervised Learning of a Deep Convolutional Network for Semantic Image Segmentation<Br>
**[Year]** ICCV 2015<Br>
**[Authors]** George Papandreou,  [Liang-Chieh Chen](http://liangchiehchen.com/), Kevin Murphy, Alan L. Yuille<Br>

### **DeepLab-V2 ★** 
**[Paper]** DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs<Br>
**[Year]** arXiv 2016<Br>
**[Authors]** [Liang-Chieh Chen](http://liangchiehchen.com/), George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille<Br>
**[Pages]**<Br>
	 http://liangchiehchen.com/projects/DeepLab.html<Br>
	 https://github.com/DrSleep/tensorflow-deeplab-resnet (tensorflow)<Br>
	 https://github.com/isht7/pytorch-deeplab-resnet (pytorch)<Br>
**[Description]** <Br>
 1) 与V1相比的不同是：不同的学习策略，多孔空间金字塔池化(ASPP)，更深的网络和多尺度。ASPP就是使用不同stride的dilated conv对同一特征图进行处理

### **DeepLab-V3 ☆** 
**[Paper]** Rethinking Atrous Convolution for Semantic Image Segmentation<Br>
**[Year]** arXiv 1706<Br>
**[Authors]** [Liang-Chieh Chen](http://liangchiehchen.com/), George Papandreou, Florian Schroff, Hartwig Adam<Br>
**[Pages]** https://github.com/tensorflow/models/tree/master/research/deeplab<Br>
**[Description]** <Br>
1) 使用串联和并行的atrous cov，使用bn，结构优化，达到了soa的精度(080116)

### **DeepLab-V3+ ★☆** 
**[Paper]** Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation <Br>
**[Year]** arXiv 1802<Br>
**[Authors]** [Liang-Chieh Chen](http://liangchiehchen.com/), Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam<Br>
**[Pages]** https://github.com/tensorflow/models/tree/master/research/deeplab <Br>
**[Description]** <Br>
1) 在DeepLab-V3作为encoder的基础上, 加入了一个简单的decoder, 而不是直接上采样; 采用Xception作为backbone
2) VOC上分割任务达到soa (0800314), 效果好
	
### ***Attention to Scale ★*** 
**[Paper]** Attention to Scale: Scale-aware Semantic Image Segmentation  <Br>
**[Year]** CVPR 2016<Br>
**[Authors]** [Liang-Chieh Chen](http://liangchiehchen.com/), Yi Yang, Jiang Wang, Wei Xu,  Alan L. Yuille<Br>
**[Pages]** http://liangchiehchen.com/projects/DeepLab.html <Br>
**[Description]** <Br>
1) 多尺度特征融合是语义分割中提高性能的关键之一, 目前特征融合一般使用简单的max或average操作. 本文则使用一个基于FCN的网络训练一weight map, 给多尺度feature map中不同目标区域的各个scale赋予不同的权值, 最后对多尺度的map进行加权求和, 得出融合特征. <Br>
2) 在训练deeplab中使用了extra supervision. 实验结果表明extra supervision对性能提升有明显作用, 比attention效果明显得多.. <Br>
	
### DPC ★★★
**[Paper]** (NIPS 2018) Searching for Efficient Multi-Scale Architectures for Dense Image Prediction  <Br>
**[Authors]** [Liang-Chieh Chen](http://liangchiehchen.com/), Maxwell D. Collins, Yukun Zhu, George Papandreou, Barret Zoph, Florian Schroff, Hartwig Adam, Jonathon Shlens<Br>
**[[TF-Code](https://github.com/tensorflow/models/tree/master/research/deeplab)]**  <Br>
搜索部分没有细看, NAS用于语义分割的代表性工作, 以被集成到Tensorflow DeepLab工程中.

### **CRFasRNN ★♥** 
**[Paper]** Conditional Random Fields as Recurrent Neural Networks<Br>
**[Year]** ICCV 2015<Br>
**[Authors]** Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, [Philip H. S. Torr](http://www.robots.ox.ac.uk/~phst/)<Br>
**[Pages]** http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html <Br>
**[Description]**<Br>
1) 将CRF推断步骤用卷积, softmax等可微模块替代, 并使用RNN的递归迭代, 将CRF用类似RNN的结构近似. 整个模型都可以end-to-end的优化.
2) 全连接CRF及其推断是在[Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/)的基础上设计的. 待深入研究CRF后应再仔细阅读这篇paper.

### **DeconvNet ★**
**[Paper]** Learning Deconvolution Network for Semantic Segmentation<Br>
**[Year]** ICCV 2015<Br>
**[Authors]** [Hyeonwoo Noh](http://cvlab.postech.ac.kr/~hyeonwoonoh/), [Seunghoon Hong](http://cvlab.postech.ac.kr/~maga33/), [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/index.html)<Br>
**[Pages]** <Br>
	 http://cvlab.postech.ac.kr/research/deconvnet/ <Br>
	 https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation (tensorflow) <Br>
**[Description]** <Br>
1) encoder-decoder的代表模型之一, conv-pool特征提取, unpool-deconv恢复分辨率.

### **SegNet ★★**
**[Paper]** SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling <Br>
**[Year]** arXiv 2015 <Br>
**[Authors]** [Alex Kendall](https://alexgkendall.com/), [Vijay Badrinarayanan](https://sites.google.com/site/vijaybacademichomepage/home), [Roberto Cipolla](http://mi.eng.cam.ac.uk/~cipolla/index.htm) <Br>
**[Pages]** http://mi.eng.cam.ac.uk/projects/segnet/ <Br>
	
**[Paper]** SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation <Br>
**[Year]** PAMI 2017 <Br>
**[Authors]** [Vijay Badrinarayanan](https://sites.google.com/site/vijaybacademichomepage/home), [Alex Kendall](https://alexgkendall.com/), [Roberto Cipolla](http://mi.eng.cam.ac.uk/~cipolla/index.htm) <Br>
**[Description]** <Br>
1) encoder-decoder的代表模型之一，特点是将encoder中的pooling indices保存下来，decoder上采样时用这些indices得到sparse feature map，再用trainable conv得到dense feature map


### ***Piecewise CRF* ★**
**[Paper]** Efficient piecewise training of deep structured models for semantic segmentation<Br>
**[Year]** CVPR 2016 <Br>
**[Authors]** [Guosheng Lin](https://sites.google.com/site/guoshenglin/), [Chunhua Shen](https://cs.adelaide.edu.au/~chhshen/2016.html), [Anton van dan Hengel](https://cs.adelaide.edu.au/~hengel/), [Ian Reid](http://www.robots.ox.ac.uk/~ian/)  <Br>
**[Pages]**<Br>
**[Description]** <Br>
1) 粗读. CRF部分没怎么看懂.
2) FeatMap-Net接受multi-scale的输入, 生成feature map; 基于feature map设计了CRF的unary和pairwise potential, pairwise中考虑了surrounding和above/below两种context.
3) CRF training提出了基于piecewise learning的方法.

### **ENet ★**
**[Paper]** ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation<Br>
**[Year]** arXiv 1606<Br>
**[Authors]** Adam Paszke, Abhishek Chaurasia, Sangpil Kim, Eugenio Culurciello <Br>
**[Pages]** https://github.com/e-lab/ENet-training  <Br>
**[Description]** <Br>
1) 一种快速的encoder-decoder分割网络
2) 大encoder，小decoder; PReLU代替ReLU; 1xn和nx1卷积代替nxn卷积

### **ParseNet ★**
**[Paper]** ParseNet: Looking Wider to See Better <Br>
**[Year]** ICLR 2016 <Br>
**[Authors]** [Wei Liu](http://www.cs.unc.edu/~wliu/), Andrew Rabinovich, [Alexander C. Berg](http://acberg.com/) <Br>
**[Pages]** https://github.com/weiliu89/caffe/tree/fcn <Br>
**[Description]** <Br>
1) 一种简单的加入global context的方法. 将feature map进行global pooling和L2 norm, 将得到的向量unpool成与原feature map相同尺寸, 再concatenate到也进行了L2 norm的feature map上.
2) 通过简单实验, 提出实际感受野往往远小于理论感受野. 很多paper都引用了这一类观点, 但是感觉缺乏理论论证-_-||

### **FoveaNet ★★**
**[Paper]** FoveaNet: Perspective-aware Urban Scene Parsing <Br>
**[Year]** ICCV 2017 Oral <Br>
**[Authors]** 	Xin Li, Zequn Jie, [Wei Wang](https://weiwangtrento.github.io/), Changsong Liu,	[Jimei Yang](https://eng.ucmerced.edu/people/jyang44),	[Xiaohui Shen](http://users.eecs.northwestern.edu/~xsh835/), [Zhe Lin](https://research.adobe.com/person/zhe-lin/), Qiang Chen, [Shuicheng Yan](https://www.ece.nus.edu.sg/stfpage/eleyans/), [Jiashi Feng](https://sites.google.com/site/jshfeng/home)  <Br>
**[Pages]** <Br>
**[Description]** <Br>
1) 提出了一种perspective-aware parsing network, 以解决 heterogeneous object scales问题, 提高远处小物体的分割精度, 减少近处大物体的”broken-down”现象.
2) 为更好解析接近vanishing point(即远离成像平面处)的物体, 提出了perspective estimation network(PEN). 通过PEN得到距离的heatmap, 根据heatmap得到包含大多数小目标的fovea region. 将fovea region放大, 与原图并行地送入网络解析. 解析出来的结果再放回原图.
3) 为解决近处目标的”broken-down”问题, 提出了perspective-aware CRF. 结合PEN得到的heatmap和目标检测, 使属于近处目标的像素有更大的pairwise potential, 属于远处目标的像素有更小的parwise potential, 有效缓解了”broken-down”和过度平滑的问题.


### **PSPNet ★☆**
**[Paper]** Pyramid Scene Parsing Network <Br>
**[Year]** CVPR 2017 <Br>
**[Authors]** Hengshuang Zhao, Jianping Shi,  Xiaojuan Qi, [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Jiaya Jia](http://www.cse.cuhk.edu.hk/leojia/) <Br>
**[Pages]** https://hszhao.github.io/projects/pspnet/ <Br>
**[Description]** <Br>
1) 提出了pyramid pooling module结合不同尺度的context information。PSPNet把特征图进行不同尺度的pooling(类似spatial pyramid pooling)，再将所有尺度的输出scale到相同尺寸，并concat起来
2) 再res4b22后接了一个auxiliary loss，使用resnet网络结构


### **RefineNet ★☆**
**[Paper]** RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation <Br>
**[Year]** CVPR 2017 <Br>
**[Authors]**  	[Xiaohang Zhan](https://xiaohangzhan.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Ping Luo](http://personal.ie.cuhk.edu.hk/~pluo/) , [Xiaoou Tang](http://www.ie.cuhk.edu.hk/people/xotang.shtml),	[Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)  <Br>
**[Pages]**  https://github.com/guosheng/refinenet <Br>
**[Description]** <Br>
1) encoder为4组残差块, 逐渐降低分辨率; decoder部分为论文提出的RefineNet. 作者认为提出的模型对高分辨率图像的细节信息有更好的分辨能力;
2) RefineNet前半部分为multi-resolution fusion, 类似于UNet, decoder的每一级模块都利用了对应的encoder模块的信息;
3) RefineNet后半部分为Chained residual pooling, 目的是"capture background context from a large image region".

### **GCN ★**
**[Paper]** Large Kernel Matters—— Improve Semantic Segmentation by Global Convolution <Br>
**[Year]** CVPR 2017 <Br>
**[Authors]**  	[Peng Chao](http://www.pengchao.org/), [Xiangyu Zhang](https://scholar.google.ca/citations?user=yuB-cfoAAAAJ&hl=zh-CN&oi=sra) [Gang Yu](http://www.skicyyu.org/), Guiming Luo, [Jian Sun](http://www.jiansun.org/)  <Br>
**[Pages]**  https://github.com/ycszen/pytorch-segmentation (Unofficial)<Br>
**[Description]** <Br>
1) 文章认为, segmentation包括localization和classification两部分, 分类需要全局信息, localization需要保证feature map的分辨率以保证空间准确度, 因此二者存在矛盾. 本文提出的解决办法就是用large kernel, 既可以保持分辨率, 又能近似densely connections between feature maps and per-pixel classifiers; <Br>
2) 文中使用k*1+1*k和1*k+k*1代替k*k的大kernel. 引入boundary refinement模块, 使用残差结构, 捕捉边界信息; <Br>
3) 只根据实验说明提出的模型优于k*k kernel和多个小kernel堆叠的策略, 但是并没什么理论支持; <Br>
4) 一点不明白: 为什么提出的基于残差结构的BR可以model the boundary alignment? <Br>

### **FastMask ★**
**[Paper]** FastMask: Segment Multi-scale Object Candidates in One Shot <Br>
**[Year]** CVPR 2017 Spotlight  <Br>
**[Authors]** [Hexiang Hu](http://hexianghu.com/), [Shiyi Lan](https://voidrank.github.io/), Yuning Jiang, Zhimin Cao, [Fei Sha](http://www-bcf.usc.edu/~feisha/) <Br>
**[Pages]**  https://github.com/voidrank/FastMask <Br>
**[Description]**<Br>
1) 粗读. 提出了一个body, neck, head的one-shot模型. 
2) body net部分进行特征提取. 提取到的特征组成多尺度的特征金字塔, 分别送入共享参数的neck module提取multi-scale特征, neck module为residual neck. 得到的特征图进行降维后提取dense sliding window, sliding windows经batch normalization后送入head module, head module为attention head 
3) neck module部分以2为步长对feature map进行下采样, 可能导致尺度过于稀疏. 因此提出two-stream FastMask architecture, 使scale更密集.

### **Layer Cascade ★**
**[Paper]** Not All Pixels Are Equal: Difficulty-Aware Semantic Segmentation via Deep Layer Cascade <Br>
**[Year]** CVPR 2017 Spotlight  <Br>
**[Authors]** Xiaoxiao Li, Ziwei Liu, [Ping Luo](http://personal.ie.cuhk.edu.hk/~pluo/), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/), [Xiaoou Tang](http://www.ie.cuhk.edu.hk/people/xotang.shtml) <Br>
**[Pages]**  https://liuziwei7.github.io/projects/LayerCascade.html <Br>
**[Description]**<Br>
1) 多个小模型级联, 不同阶段处理不同难度的样本, 是一类典型的节省计算的方法. 本文提出一种层级联的语义分割方法, 把网络中不同层视为不同stage, 达到近似于模型级联的效果, 提升了性能, 降低了计算量. <Br>
2) 将backbone的不同阶段(3阶段)拉出来做预测, 把其中置信度低于一定阈值的区域作为目标, 下一阶段只对其卷积, 其余位置直接置0. 最后把不同阶段的结果合成成最后的输出. <Br>
3) 感觉思路很直观清晰, 或许可以在后续的工作中参考. 有个疑问是: 上下文信息在语义分割中应该是很重要的, 这种只对region进行处理的方案会不会导致全局信息不足? 
	
### **PixelNet ★**
**[Paper]** Representation of the pixels, by the pixels, and for the pixels <Br>
**[Year]** TPAMI 2017 <Br>
**[Authors]**  	[Aayush Bansal](http://www.cs.cmu.edu/~aayushb/),	[Xinlei Chen](http://www.cs.cmu.edu/~xinleic/), [Bryan Russell](http://www.bryanrussell.org/), [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg/), [Deva Ramanan](http://www.cs.cmu.edu/~deva/) <Br>
**[Pages]**  http://www.cs.cmu.edu/~aayushb/pixelNet/ <Br>
**[Description]** <Br>
1) 粗读. 使用hypercolumn思想, 速度快. 适用于segmentation, 边缘检测, normal estimation等low-level到high-level的多种问题.
2) hypercolumn即: 对于一个pixel, 将每一层feature map中其对应位置的feature连接起来组成一个vector, 用MLP对该vector分类.
3) 文中提出, 训练时 just sampling a small number of pixels per image is sufficient for learning. 这样一个mini-batch里就可以从多张图片中采样, 增加了diversity.

### **LinkNet ☆**
**[Paper]** LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation <Br>
**[Year]** arXiv 1707 <Br>
**[Authors]**  	[Abhishek Chaurasia](https://codeac29.github.io/),	[Eugenio Culurciello](http://e-lab.github.io/)<Br>
**[Pages]**  https://codeac29.github.io/projects/linknet/ <Br>
**[Description]** <Br>
1) 还没读, 大致是一个类似U-Net的结构, 速度快 <Br>
	
### SDN ★
**[Paper]** Stacked Deconvolutional Network for Semantic Segmentation  <Br>
**[Year]** CVPRW 2017<Br>
**[Author]** Simon J´egou, Michal Drozdza, [David Vazquez](http://www.david-vazquez.com/), [Adriana Romero](https://sites.google.com/site/adriromsor/), [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html)<Br>
**[Pages]**  https://github.com/SimJeg/FC-DenseNet<Br>
**[Description]**  <Br>
1) DenseNet + 类似U-Net的结构. 大致浏览 <Br>
	
### FC-DenseNet ★
**[Paper]** The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation  <Br>
**[Year]** arXiv 1708 <Br>
**[Author]** 	Jun Fu, Jing Liu, Yuhang Wang, Hanqing Lu  <Br>
**[Pages]**  <Br>
**[Description]**  <Br>

### ISCTF ☆
**[Paper]** Real-time Semantic Image Segmentation via Spatial Sparsity  <Br>
**[Year]** arXiv 1712 <Br>
**[Author]** Zifeng Wu, [Chunhua Shen](https://cs.adelaide.edu.au/~chhshen/), [Anton van den Hengel](https://cs.adelaide.edu.au/~hengel/)  <Br>
**[Pages]**  <Br>
**[Description]**  <Br>
1) 粗读, 实时语义分割方法. 提出了一个由低分辨率图像产生稀疏weight map, 引导高分辨率图只处理少数区域, 以达到减小计算量的同时保持边缘精度的目的.<Br>
2) 得到spatial sparsity部分以及从原图以什么尺度crop部分没细看. 算法看上去实现可能有些繁琐<Br>
3) 从实验大致来看, 提出的基于sparisity的方案似乎带来的提升有限. 另外在速度和性能上与18年以来的real time方案相比, 似乎不占优势. 但是论文的思路很有意 思, 可以日后持续关注. <Br>

### *Dense Decoder Shortcut Connections* ★
**[Paper]** Dense Decoder Shortcut Connections for Single-Pass Semantic Segmentation <Br>
**[Year]** CVPR 2018 <Br>
**[Author]** [Piotr Bilinski](http://www.piotr-bilinski.com/), [Victor Prisacariu](https://scholar.google.com/citations?user=szL1daIAAAAJ&hl=en) <Br>
**[Pages]**  <Br>
**[Description]**  <Br>
1. 粗读. 一种encoder-decoder的语义分割模型, 大致就是skip-connection和dense-connection, 并用ResNeXt做backbone, 思路没什么新奇的. 在设计网络时, 加入了很多对multi-scale的支持, 因此文中生成他们的网络只需要single-scale inference. <Br> 

### DFN ★☆
**[Paper]** Learning a Discriminative Feature Network for Semantic Segmentation <Br>
**[Year]** CVPR 2018 <Br>
**[Author]** [Changqian Yu](http://changqianyu.me/), [Jingbo Wang](https://github.com/wangjingbo1219), [Chao Peng](http://www.pengchao.org/), [Changxin Gao](https://sites.google.com/site/changxingao), [Gang Yu](http://www.skicyyu.org/), Nong Sang <Br>
**[Pages]**  https://github.com/whitesockcat/Discriminative-Feature-Network (Unofficial)<Br>
**[Description]**  <Br>
1) 本文的目的是解决intra-class inconsistency和inter-class indistinction两个, 为此设计了Smooth Network和Border Network, 为实现这两个网络, 又设计了Refinement Residual Block(RRB)和Channel Attention Block (CAB).<Br>
2) Smooth Network是本文的重点. 文中认为类内不一致主要是由于缺乏上下文信息, 因此设计了从global pooling开始自顶向下的逐层refine, 利用来自上层的全局信息得到channel attention vector作为guidance, 使下层选出最有用的channel. <Br>
3) Border Network就是一有前面几层concat的encoder和decode结构. 文中说是给高层特征提供边界信息, 其实对最后结果作用不大. <Br>
4) 本篇paper中的RRB, CAB结构上虽然没有很新颖, 但把它们用到要解决的问题上并且得到很好的效果还是很厉害的. 另外paper中对问题和自己工作的阐述很值得学习. 两个问题: 本文提出的border network来解决inter-class indistinction, 说服力不太强; Smooth Network是用上下文信息去选择channel, 没有考虑feature的空间修正, 但不能保证仅靠选特定的feature就能解决intra-class inconsistency问题. <Br>

### *Adapt Structured Output Space* ★
**[Paper]** Learning to Adapt Structured Output Space for Semantic Segmentation <Br>
**[Year]** CVPR 2018 Spotlight <Br>
**[Author]** [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/home), [Wei-Chih Hung](https://hfslyc.github.io/), [Samuel Schulter](https://samschulter.github.io/), [Kihyuk Sohn](https://sites.google.com/site/kihyuksml/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/index.html), Manmohan Chandraker  <Br>
**[Pages]**  https://github.com/wasidennis/AdaptSegNet<Br>
**[Description]**  <Br>
1) 提出了一种基于对抗学习的用于语义分割的domain adaptation方法, 在GTA5上训练CityScape测试, 效果不错. <Br>
2) 在输出层和中间的一个特征层上做multi-level的对抗训练, 使target domain的预测结果逼近于source domain的预测结果. <Br>

### EncNet ★★
**[Paper]** Context Encoding for Semantic Segmentation <Br>
**[Year]** CVPR 2018 Oral<Br>
**[Author]** [Hang Zhang](http://hangzh.com/), [Kristin Dana](http://eceweb1.rutgers.edu/vision/dana.html), [Jianping Shi](http://shijianping.me/), [Zhongyue Zhang](http://zhongyuezhang.com/), [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Ambrish Tyagi](https://scholar.google.com/citations?user=GaSWCoUAAAAJ&hl=en), [Amit Agrawal](http://www.amitkagrawal.com/)<Br>
**[Pages]**  <Br>
	https://hangzhang.org/PyTorch-Encoding/ <Br>
	https://github.com/zhanghang1989/PyTorch-Encoding<Br>
**[Description]**  <Br>
1) 提出了基于Deep Ten的Context Encoding Module, 嵌入语义分割网络中, 提高对global context information的利用. <Br>
2) 虽然网络的创新性工作不多, 但把VLAD一类的思想用来挖掘语义分割任务中的上下文信息, 思路还是值得借鉴的. <Br>

### CCNet ★☆
**[Paper]** CCNet: Criss-Cross Attention for Semantic Segmentation <Br>
**[Year]** ICCV 2019 <Br>
**[Author]** [Zilong Huang](https://speedinghzl.github.io/), [Xinggang Wang](http://xinggangw.info/), Lichao Huang, Chang Huang, [Yunchao Wei](https://weiyc.github.io/), Wenyu Liu<Br>
**[Pages]** https://github.com/speedinghzl/CCNet <Br>
**[Description]**  <Br>
1) 在何凯明等提出的Non-Local的基础上, 提出了一个recurrent的十字形attention形式, 并采用了残差+attention的结构. 方法相比于Non-local和许多其它attention方法, 计算成本低, 且效果不错. <Br>
2) recurrent的次数设为2, 因为文中指出, 两次十字形的attention计算已经可以对图像中的任意两点建立连接. 实验表明, 这种二次循环的十字形attention的确可以捕捉到有用的信息, 且增大递归次数对性能提升作用不大. <Br>

### **ICNet ★** 
**[Paper]** ICNet for Real-Time Semantic Segmentation on High-Resolution Images <Br>
**[Year]**  ECCV 2018 <Br>
**[Authors]** [Hengshuang Zhao](https://hszhao.github.io/), [Xiaojuan Qi](https://xjqi.github.io/), [Xiaoyong Shen](http://xiaoyongshen.me/), [Jianping Shi](http://shijianping.me/), [Jiaya Jia](http://jiaya.me/)<Br>
**[Pages]**  <Br>
https://github.com/hszhao/ICNet <Br>
https://github.com/hellochick/ICNet-tensorflow <Br>
**[Description]** <Br>
1) 粗读, 多分辨率输入网络级联的实时语义分割算法, 是多分辨率特征融合做轻量级语义分割模型的代表方法之一. <Br>
2) 输入图像以原尺寸, 1/2, 1/4三个分辨率输入三个分支, 小分辨率分支层数较多, 负责提取全局信息; 大分辨率网络层数少, 节省计算成本. 设计了融合模块融合各分辨率的特征. 对每个分支的输出均计算loss. <Br>

### **BiSeNet ★☆** 
**[Paper]** BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation <Br>
**[Year]**  ECCV 2018 <Br>
**[Authors]** Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, Nong Sang<Br>
**[Pages]** <Br>
https://github.com/ycszen/TorchSeg (3rd party)<Br>
https://github.com/ooooverflow/BiSeNet (3rd party)<Br>
https://github.com/GeorgeSeif/Semantic-Segmentation-Suite (3rd party)<Br>
**[Description]** <Br>	
1) 实时语义分割, 空间细节特征和上下文语义特征融合的典型方法, 在时间和性能上取得了很好的平衡. <Br>
2) 分为spatial path和context path两支, spatial部分用3个conv下采样8倍, 保持空间细节信息; context部分使用xception39或resnet18为backbone, 设计了类似SENet的attention进行refine; 最后特征融合时设计了类似于residual attention的融合模块. 使用了auxiliary loss. 

### **ShelfNet ★** 
**[Paper]* ShelfNet for Fast Semantic Segmentation <Br>
**[Year]**  arXiv 1811 <Br>
**[Authors]** Juntang Zhuang, Junlin Yang, Lin Gu, Nicha Dvornek<Br>
**[Pages]** https://github.com/juntang-zhuang/ShelfNet <Br>
**[Description]** <Br>	
1) 一个实时分割网络, 使用多个encoder-decoder结构, 达到ensemble的作用. 性能不错, 同样速度下优于BiSeNet. <Br>
	
### **Fast-SCNN ★** 
**[Paper]** Fast-SCNN: Fast Semantic Segmentation Network <Br>
**[Year]**  arXiv 1902 <Br>
**[Authors]** 	Rudra PK Poudel, Stephan Liwicki, [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/)<Br>
**[Pages]**  <Br>
**[Description]** <Br>
1) 采用two branch和encoder-decoder的思路做real time的语义分割, 整体思路与BiSeNet非常相似. 在大图像上(1027*2018)速度很快(123fps), 性能与sota相比有差距. 
2) 结构: 先用3个卷积降采样8倍, 从此引出一skip connection负责保留空间细节, 另一分支作为feature extractor由若干bottleneck和金字塔池化组成, 最后通过sum将特征融合. 大量使用depthwise separable convolution提速
	
### **DFANet ★** 
**[Paper]** DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation <Br>
**[Year]**  CVPR 2019 <Br>
**[Authors]** 	Hanchao Li, Pengfei Xiong, Haoqiang Fan, [Jian Sun](http://www.jiansun.org/)<Br>
**[Pages]**  <Br>
**[Description]** <Br>	
1) 旷世提出的一个real time语义分割方法, 在性能和速度上达到不错的平衡, 可保持关注. <Br>
2) 采用多分辨率网络聚合的策略, 挖掘不同尺度的上下文信息, 聚合的创新之处在于同时使用了sub-network层级的聚合和sub-stage(即网络内部的feature)层级的聚合. decoder阶段利用各分辨率特征的信息. Backbone是简化版的Xception. <Br>

### **ShuffleNetV2+DPC ★** 
**[Paper]** An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions <Br>
**[Year]**  arXiv 1902 <Br>
**[Authors]** Sercan Turkmen, Janne Heikkila<Br>
**[Pages]**  https://github.com/sercant/mobile-segmentation<Br>
**[Description]** <Br>	
1) 基于Deeplab+DPC, 用ShuffleNet v2做backbone. <Br>

### **JPU ★** 
**[Paper]** FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation <Br>
**[Year]**  arXiv 1903 <Br>
**[Authors]** [Huikai Wu](http://wuhuikai.me/), Junge Zhang, Kaiqi Huang, Kongming Liang, Yizhou Yu<Br>
**[Pages]** <Br>
	http://wuhuikai.me/FastFCNProject/ <Br>
	https://github.com/wuhuikai/FastFCN <Br>
**[Description]** <Br>
1) 粗读. dilation卷积计算量很大, 本位利用所谓joint upsampling的思想, 找到一种更高效的提取高精度特征的方案以代替dilation. <Br>
2) 前面论述很很多, 最后设计的JPU模块实际上就是一个类似于ASPP的结构, 只不过是利用了前三层的特征而不是一层. 感觉后面的方法和前面的叙述有些脱节, 可能是自己没完全理解. <Br>
3) 实验结果上看, JPU和FPN结构相比性能和速度上非常接近.<Br>
	
### **Gated-SCNN ★☆** 
**[Paper]** Gated-SCNN: Gated Shape CNNs for Semantic Segmentation <Br>
**[Year]**  ICCV, 2019 <Br>
**[Authors]** [Towaki Takikawa](https://tovacinni.github.io/), [David Acuna](http://www.cs.toronto.edu/~davidj/), [Varun Jampani](https://varunjampani.github.io/), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)<Br>
**[Pages]** <Br>
	https://nv-tlabs.github.io/GSCNN/ <Br>
**[Description]** <Br>
1) 提出一个双分支网络, 一个分支负责分割, 一个分支负责边缘检测, 最后将两分支融合生成最后的分割结果. <Br>
2) 边缘检测部分, 使用Gated-Convolution, 让分割一支的high-level特征引导lower-level的边缘检测特征, 以去除noise. <Br>
3) 使用dual task regularization, 目的是exploit the duality between semantic segmentation and semantic boundary prediction. <Br>
4) 在语义分割中引入gated convolution作为一种gating mechanism的思路值得借鉴. 本方法在cityscapes上性能与DPC持平 <Br>

### **CFNet ★☆** 
**[Paper]** Co-occurrent Features in Semantic Segmentation <Br>
**[Year]**  CVPR 2019 <Br>
**[Authors]** [Hang Zhang](https://hangzhang.org), [Han Zhang](http://paul.rutgers.edu/~hz138/), [Chenguang Wang](https://cgraywang.github.io/), [Junyuan Xie](https://scholar.google.com/citations?user=qJsC_XsAAAAJ&hl=en)<Br>
**[Pages]**  <Br>
**[Description]** <Br>
1) 通过计算target feature和其它feature的co-occurrence概率,去挖掘co-occurenct context information. 从报告的数据来看, 效果不错. <Br>
2) CNN特征提取后, 分为三部分：co-occurrent概率计算, co-occurent context先验提取, global pooling. <Br>
3) 个人水平有限, 感觉paper的论述有点不太清晰, 而且写作也是。。。  <Br>

### **DANet ★☆** 
**[Paper]** Dual Attention Network for Scene Segmentation <Br>
**[Year]**  CVPR 2019 <Br>
**[Authors]** Jun Fu, [Jing Liu](http://www.nlpr.ia.ac.cn/iva/liujing/index.html), Haijie Tian, [Yong Li](http://www.foreverlee.net/), Yongjun Bao, Zhiwei Fang,and Hanqing Lu<Br>
**[Pages]**  https://github.com/junfu1115/DANet<Br>
**[Description]** <Br>
1) 大致浏览, 提出了一个包括position attention和channel attention的dual attention模块, 并在此基础上设计了语义分割网络DANet, 取得了还不错的效果. <Br>
2) position attention就是计算两像素在channel维度上的相似度, channel attention就是计算两通道在空间维度上的相似度. 此处相似度都是通过计算內积得到的. <Br>
	
### ***Integrated Classification* ★** 
**[Paper]** Scene Parsing via Integrated Classification Model and Variance-Based Regularization <Br>
**[Year]**  CVPR 2019 <Br>
**[Authors]** Hengcan Shi, Hongliang Li, Qingbo Wu, Zichen Song<Br>
**[Pages]**  https://github.com/shihengcan/ICM-matcaffe<Br>
**[Description]** <Br>	
1) 提出了一个二阶段pixel像素分类的场景检测方法. 第一阶段用多个二分类器初步分类, 第二阶段对一阶段结果进行refine, 修正之前混淆的类别. <Br>
2) 使用了一个variance-based regularization, 促使最后的类别间概率相差尽可能大. <Br>

### **ADVENT** 
**[Paper]** ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation<Br>
**[Year]**  CVPR 2019 Oral<Br>
**[Authors]** [Tuan-Hung Vu](https://tuanhungvu.github.io/), [Himalaya Jain](https://himalayajain.github.io/), [Maxime Bucher](https://maximebucher.github.io/), [Matthieu Cord](http://webia.lip6.fr/~cord/), [Patrick Pérez](https://ptrckprz.github.io/)<Br>
**[Pages]**  https://github.com/valeoai/ADVENT<Br>
**[Description]** <Br>	
	
### **ShelfNet ** 
**[Paper]**  ShelfNet for fast semantic segmentation<Br>
**[Year]** arXiv 1811<Br>
**[Authors]** Juntang Zhuang, Junlin Yang, Lin Gu, Nicha Dvornek<Br>
**[Pages]**  https://github.com/juntang-zhuang/ShelfNet <Br>
**[Description]** <Br>	
	
### **SwiftNet ** 
**[Paper]**  In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images<Br>
**[Year]** CVPR 2019 <Br>
**[Authors]** Marin Oršić, [Ivan Kreš](https://ivankreso.github.io/), [Siniša Šegvić](http://www.zemris.fer.hr/~ssegvic/index_en.html), Petra Bevandić<Br>
**[Pages]** https://github.com/orsic/swiftnet <Br>
**[Description]** <Br>			
	
## Panoptic Segmentation
### **DeeperLab ★** 
**[Paper]** DeeperLab: Single-Shot Image Parser<Br>
**[Year]**  CVPR 2019 <Br>
**[Authors]** 	Tien-Ju Yang, Maxwell D. Collins, [Yukun Zhu](http://www.cs.toronto.edu/~yukun/), Jyh-Jing Hwang, [Ting Liu](http://www.tliu.org/), Xiao Zhang, Vivienne Sze, [George Papandreou](https://ttic.uchicago.edu/~gpapan/), [Liang-Chieh](http://liangchiehchen.com/) Chen<Br>
**[Pages]**  <Br>
**[Description]** <Br>
1) 大致浏览, XX Lab又一弹, 借鉴了Xception和MobileNet等网络的backbone设计思路, 使用了两个分支分别做semantic和instance的分割. <Br> 


## Foreground-background Segmenation

### **Pixel Objectness ★** <Br>
**[Paper]** Pixel Objectness: Learning to Segment Generic Objects Automatically in Images and Videos <Br>
**[Year]** TPAMI 2018 <Br>
**[Authors]** [Bo Xiong](https://www.cs.utexas.edu/~bxiong/publication.html), [Suyog Jain](http://www.cs.utexas.edu/~suyog/), [Kristen Grauman](http://www.cs.utexas.edu/~grauman/) <Br>
**[Pages]** http://vision.cs.utexas.edu/projects/pixelobjectness/ <Br>
**[Description]** <Br>
1) 大致浏览, 提出pixel objectness这一术语表示得到图像和视频中generic object二类分割图的过程. 提出基于CNN的网络, 对图像和视频中的目标进行分割, 该模型对没见过的物体也很鲁棒.
2) 大段论述了他们的方法为什么合理, 比如用ImageNet预训练模型里面已经包含类别信息了之类的囧, 没仔细看. 
	

## Transfer Related

### ***Image-level to Pixel-level Labeling* ★** <Br>
**[Paper]** From Image-level to Pixel-level Labeling with Convolutional Networks <Br>
**[Year]** CVPR 2015 <Br>
**[Authors]** [Pedro O. Pinheiro](http://pedro.opinheiro.com/publications/), [Ronan Collobert](http://ronan.collobert.com/) <Br>
**[Pages]** <Br>
**[Description]** <Br>
1) 一种weakly supervised方法，用图像类别标签训练分割模型，分割中每个类别的特征图用log-sum-exp变换为分类任务中每个类别的概率，通过最小化分类的loss优化分割模型
2) 推断时为抑制False Positive现象，使用了两种分割先验：Image-Level Prior(分类概率对分割加权)和Smooth Prior(超像素，bounding box candidates，无监督分割MCG)。
	
### **BoxSup ★**
**[Paper]** BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation <Br>
**[Year]** ICCV 2015 <Br>
**[Authors]** [Jifeng Dai](http://www.jifengdai.org/), [Kaiming He](http://kaiminghe.com/), [Jian Sun](http://www.jiansun.org/) <Br>
**[Pages]** <Br>
**[Description]** <Br>
1) 弱监督语义分割，用bounding box结合region proposal(MCG)生成初始groundtruth mask，再交替更新分割结果和mask.

### **Mix-and-Match ★** 
**[Paper]** Mix-and-Match Tuning for Self-Supervised Semantic Segmentation <Br>
**[Year]**  AAAI 2018 <Br>
**[Authors]** 	[Xiaohang Zhan](https://xiaohangzhan.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Ping Luo](http://personal.ie.cuhk.edu.hk/~pluo/), [Xiaoou Tang](http://www.ie.cuhk.edu.hk/people/xotang.shtml), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)  <Br>
**[Pages]**  http://mmlab.ie.cuhk.edu.hk/projects/M&M/   <Br>
**[Description]** <Br>
1) self-supervision可分为proxy stage和fine-tuning stage两个阶段. 先用无需标签数据的proxy task(如图像上色)进行预训练, 学到某种语义特征, 再用少量的标记数据进行微调. 但由于proxy task和target task之间存在semantic gap, 自监督方法性能明显较监督方法差.
2) 论文提出了"mix-and-match"策略, 利用少数标记数据提升自监督预训练网络的性能. mix step: 从不同图像中随机提取patch. match step: 在训练时通过on-the-fly的方式构建graph, 并生成triplet, triplet包括anchor , positive, negative patch三个元素. 据此可定义一triplet loss, 鼓励相同类别的patch更相似, 不同类别的patch差别更大.
3) 对自监督了解不够深入, 看代码有助理解. segmentation部分采用的hypercolumn方法论文中貌似没仔细说, 以后可以再研究研究.<Br>

### ***Bidirectional Learning* ★** 
**[Paper]** Bidirectional Learning for Domain Adaptation of Semantic Segmentation <Br>
**[Year]** CVPR 1904 <Br>
**[Authors]** 	Yunsheng Li, Lu Yuan, [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno/)  <Br>
**[Pages]** https://github.com/liyunsheng13/BDL   <Br>
**[Description]** <Br>
1) 提出了一个双向的语义分割domain adaptation方法, 可以用于没有任何target domain真值的情况, 从实验结果看效果不错. <Br>
2) 目前常用的的步骤是学习两个分离的网络, 首先是使用GAN学习source到target的变换, 以减小两个domain的gap, 然后用变换了的source图像做分割. 本文提出的所谓双向是指两个阶段的网络会互相作用以提升对方的性能, 另外提出了一个所谓的自监督方法, 用高置信度的target分割结果作为真值. 具体loss论文中描述的很清楚. <Br>
	
### ***Hierarchical Region Selection* ☆** 
**[Paper]** Not All Areas Are Equal: Transfer Learning for Semantic Segmentation via Hierarchical Region Selectionn <Br>
**[Year]** CVPR 2019 Oral <Br>
**[Authors]** 	Ruoqi Sun, [Xinge Zhu](https://xingezhu.me/aboutme.html), Chongruo Wu, Chen Huang, [Jianping Shi](http://shijianping.me/pub.html), Lizhuang Ma <Br>
**[Pages]**  <Br>
**[Description]** <Br>	
1) 提出一种针对语义分割的transfer learning方法, 从pixel, region, image三个尺度挖掘source样本中与target domain相似的部分用来训练, 以弥补source和target domain的gap. <Br>
2) 学习了三个weight map分别代表pixel, region, image层面上source与target的相似程度, 三个map取平均候得到最后的weight map, 在计算source的loss时用该map对每个像素加权. 另外在encoder出来的特征后加一生成对抗网络, 帮助domain adaptation. <Br>
3) 个人感觉作为一篇Oral来说有趣的地方并不多. 实验用VGG和FCN做backbone, 对比方法只用了transfer learning相关的几种方法, 性能与目前没有用transfer learning的SOTA方法比差距很大, 另外GAN似乎并没有提升性能. <Br>  
	
### **SPNet ★** 
**[Paper]** Semantic Projection Network for Zero- and Few-Label Semantic Segmentation <Br>
**[Year]** CVPR 2019 <Br>
**[Authors]** 	Yongqin Xian, Subhabrata Choudhury, Yang He, Bernt Schiele, Zeynep Akata <Br>
**[Pages]** https://github.com/subhc/SPNet <Br>
**[Description]** <Br>		
1) 基于word embeddings提出的针对无标签或少量标签样本的语义分割算法. <Br>
2) 由CNN生成每个像素的embedding, 然后计算其与预先得到的class prototype矩阵的內积, 取最相似的类别作为该像素的类别. 该算法的核心是得到word embedding, 本文中是用已有算法(如word2vec)计算的. Inference时只要用感兴趣类别组成的embedding矩阵(见过或没见过的类别均可)去做projection即可. <Br>
	
## Knowledge Distillation
### ***Knowledge Adaptation* ★☆** 
**[Paper]** Knowledge Adaptation for Efficient Semantic Segmentation <Br>
**[Year]** CVPR 2019 <Br>
**[Authors]** 	Tong He, Chunhua Shen, Zhi Tian, Dong Gong, Changming Sun, Youliang Yan <Br>
**[Pages]** <Br>
**[Description]** <Br>	
1) 粗读, 提出了一个用于语义分割的知识蒸馏方法. 在MobileNetV2的基础上mIoU提高了2个点.<Br>
2) 本文不是直接让student的feature逼近teacher的feature, 而是用一个自编码器把teacher的feature变换为更compact的表示, 并令student去逼近这个表示. <Br>
3) 本文认为小网络捕获long-term dependency的能力比较弱, 所以设计了一个affinity distillation模块, 采用Non Local的思路, 计算两两像素间的內积, 并使teacher和student网络的affinity matrix相近. <Br>

## Other Interesting Papers
### COB ★
**[Paper]** Convolutional Oriented Boundaries <Br>
**[Year]** ECCV 2016 <Br>
**[Author]** K.K. Maninis, J. Pont-Tuset, P. Arbeláez, L.Van Gool <Br>
**[Pages]** http://www.vision.ee.ethz.ch/~cvlsegmentation/cob/index.html <Br>
**[Description]** <Br>
1) 由边缘概率得到分割结果, 整体流程来自伯克利的gPb-owt-ucm, 将前面得到概率图的部分用CNN代替 
2) CNN部分使用多尺度模型预测coarse和fine的8方向的概率 
3) UCM部分提出了sparse boundaries representation, 加快了速度



# Traditional Classical Methods
### **gPb-owt-ucm** ★★★
**[Paper]** Contour Detection and Hierarchical Image Segmentation <Br>
**[Year]** TPAMI 2011 <Br>
**[Authors]**  [Pablo Arbelaez](https://biomedicalcomputervision.uniandes.edu.co/),	[Michael Maire](http://ttic.uchicago.edu/~mmaire/) ,	[Charless Fowlkes](http://www.ics.uci.edu/~fowlkes/) , 	[Jitendra Malik](http://people.eecs.berkeley.edu/~malik/) <Br>
**[Pages]**  https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html  <Br>
**[Reference]** http://blog.csdn.net/nature_XD/article/details/53375344?locationNum=9&fps=1 <Br>
**[Description]** <Br>
1) gPb(Global Probability of Boundary):由mPb和sPb组成
2) OWT:对分水岭变换得到的arc上的像素依据其方向重新计算gPb
3) UCM:貌似和MST聚类差不多？
4) sPb还没看懂<Br>


# Datasets
[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) <Br>
[MSCOCO](http://cocodataset.org/#home) <Br>
[ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) <Br>
[MITScenceParsing](http://sceneparsing.csail.mit.edu/) <Br>
[Cityscapes](https://www.cityscapes-dataset.com/) <Br>
## 3D
[PartNet](https://cs.stanford.edu/~kaichun/partnet/)


# Leaderboards
[PASCAL VOC](http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php) <Br>
[ILSVRC2016](http://image-net.org/challenges/LSVRC/2016/results)  <Br>
[Cityscapes](https://www.cityscapes-dataset.com/benchmarks/#instance-level-scene-labeling-task) <Br>


# Sources-Lists<Br>
https://handong1587.github.io/deep_learning/2015/10/09/segmentation.html <Br>
https://github.com/mrgloom/awesome-semantic-segmentation <Br>
https://blog.csdn.net/zziahgf/article/details/72639791 <Br>

