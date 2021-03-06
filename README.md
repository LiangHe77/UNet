UNet是医学图像处理方面著名的图像分割网络，过程是这样的：输入是一幅图，输出是目标的分割结果。继续简化就是，一幅图，编码，或者说降采样，然后解码，也就是升采样，然后输出一个分割结果。根据结果和真实分割的差异，反向传播来训练这个分割网络。其网络结构如下：

![Alt text](https://github.com/LiangHe77/UNet/blob/master/UNet_structure.jpg)

可以看出，该网络结构主要分为三部分：下采样，上采样以及跳跃连接。首先将该网络分为左右部分来分析，左边是压缩的过程，即Encoder。通过卷积和下采样来降低图像尺寸，提取一些浅显的特征。右边部分是解码的过程，即Decoder。通过卷积和上采样来获取一些深层次的特征。其中卷积采用的valid的填充方式来保证结果都是基于没有缺失上下文特征得到的，因此每次经过卷积后，图像的大小会减小。中间通过concat的方式，将编码阶段获得的feature map同解码阶段获得的feature map结合在一起，结合深层次和浅层次的特征，细化图像，根据得到的feature map进行预测分割。要注意的是这里两层的feature map大小不同，因此需要经过切割。最后一层通过1x1的卷积做分类。# UNet
