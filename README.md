# Personal-Semantic-Segmentation-Paper-Record
# Under construction!
# Table of Contents
- [Deep Learning Methods](#deep-learning-methods)
  - [Semantic Segmentation](#semantic-segmentation)
  - [Weakly/Self - Supervision](#weakly-self-supervision)
  - [Self-supervision](#self-supervision)
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
**[Paper]** Rethinking Atrous Convolution for Semantic Image Segmentation<Br>
**[Year]** arXiv 2017<Br>
**[Authors]** [Liang-Chieh Chen](http://liangchiehchen.com/), Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam<Br>
**[Pages]** https://github.com/tensorflow/models/tree/master/research/deeplab <Br>
**[Description]** <Br>
1) 在DeepLab-V3作为encoder的基础上, 加入了一个简单的decoder, 而不是直接上采样; 采用Xception作为backbone
2) VOC上分割任务达到soa (0800314), 效果好

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
**[Pages]**  <Br>
**[Description]** <Br>
1) 文章认为, segmentation包括localization和classification两部分, 分类需要全局信息, localization需要保证feature map的分辨率以保证空间准确度, 因此二者存在矛盾. 本文提出的解决办法就是用large kernel, 既可以保持分辨率, 又能近似densely connections between feature maps and per-pixel classifiers; <Br>
2) 文中使用k*1+1*k和1*k+k*1代替k*k的大kernel. 引入boundary refinement模块, 使用残差结构, 捕捉边界信息; <Br>
3) 只根据实验说明提出的模型由于k*k kernel和多个小kernel堆叠的策略, 但是并没什么理论支持; <Br>
4) 一点不明白: 为什么提出的基于残差结构的BR可以model the boundary alignment? <Br>

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
**[Year]** arXiv 1708 <Br>
**[Author]** 	Jun Fu, Jing Liu, Yuhang Wang, Hanqing Lu  <Br>
**[Pages]**  <Br>
**[Description]**  <Br>
1) 粗读. 效果好, 未开源.
2) 以DenseNet为基础, 构建了stacked的encoder-decoder模型, 论文中认为这能更好的捕捉multi-scale context. 网络充满了inter和intra的unit connections, 并加入了hierarchical supervisions, 使非常深的SDN能够成功训练.


## Weakly/Self - Supervision

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


## Self-supervision
### **Mix-and-Match ★** 
**[Paper]** Mix-and-Match Tuning for Self-Supervised Semantic Segmentation <Br>
**[Year]**  AAAI 2018 <Br>
**[Authors]** 	[Xiaohang Zhan](https://xiaohangzhan.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Ping Luo](http://personal.ie.cuhk.edu.hk/~pluo/), [Xiaoou Tang](http://www.ie.cuhk.edu.hk/people/xotang.shtml), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)  <Br>
**[Pages]**  http://mmlab.ie.cuhk.edu.hk/projects/M&M/   <Br>
**[Description]** <Br>
1) self-supervision可分为proxy stage和fine-tuning stage两个阶段. 先用无需标签数据的proxy task(如图像上色)进行预训练, 学到某种语义特征, 再用少量的标记数据进行微调. 但由于proxy task和target task之间存在semantic gap, 自监督方法性能明显较监督方法差.
2) 论文提出了"mix-and-match"策略, 利用少数标记数据提升自监督预训练网络的性能. mix step: 从不同图像中随机提取patch. match step: 在训练时通过on-the-fly的方式构建graph, 并生成triplet, triplet包括anchor , positive, negative patch三个元素. 据此可定义一triplet loss, 鼓励相同类别的patch更相似, 不同类别的patch差别更大.
3) 对自监督了解不够深入, 看代码有助理解. segmentation部分采用的hypercolumn方法论文中貌似没仔细说, 以后可以再研究研究.<Br>


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


# Leaderboards
[PASCAL VOC](http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php) <Br>
[ILSVRC2016](http://image-net.org/challenges/LSVRC/2016/results)  <Br>


# Sources-Lists<Br>
https://handong1587.github.io/deep_learning/2015/10/09/segmentation.html <Br>
https://github.com/mrgloom/awesome-semantic-segmentation <Br>
https://blog.csdn.net/zziahgf/article/details/72639791 <Br>

