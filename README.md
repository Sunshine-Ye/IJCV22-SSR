# IJCV22-SSR
Official code of **"Efficient Joint-Dimensional Search with Solution Space Regularization for Real-Time Semantic Segmentation"**

**文章摘要：** 语义分割是计算机视觉的热门研究课题，研究者们对此进行了大量的研究并取得了令人印象深刻的成果。在本文中，我们打算通过神经架构搜索（NAS）自动搜索一个最优的网络结构，可以实现实时且准确的语义分割。为了达到这一目标，我们联合搜索网络的深度、通道数、膨胀率和特征空间分辨率，这是一个由大约2.78×10^324个可能选择组成的搜索空间。为了处理如此大的搜索空间，我们利用可微分架构搜索方法。然而，现有可微分方法搜索所用的架构参数需要离散化，这将导致可微分方法学到的架构参数与架构搜索最终解的离散化版本之间存在离散化差距问题。因此，我们提出从解空间正则化的创新角度来理解并解决离散化差距问题。
具体地，我们首先提出了一种新颖的解空间正则化（SSR）损失函数，有效地鼓励超网收敛到其离散网络。然后，我们提出了一个分层渐进解空间收缩策略，进一步提高搜索的效率并减少计算消耗。此外，我们从理论上证明了SSR损失在优化上等效于L0范数正则化，这解释了缩小的离散化差距。综合实验表明，所提搜索方案可以高效地找到最优的分割网络结构，在保持精度相当的同时，以极小的模型尺寸（1 M）获得极快的分割速度（175 FPS）。

**研究动机：** 联合多维搜索可以保留更多上下文语义和空间细节信息，实现更好的分割
![image](https://github.com/Sunshine-Ye/IJCV22-SSR/blob/main/assets/%E6%A1%86%E6%9E%B6%E5%9B%BE1_13.png)

**高效联合多维搜索：** 多维搜索空间（深度、宽度、膨胀率和特征空间分辨率）+SSR搜索算法
![image](https://github.com/Sunshine-Ye/IJCV22-SSR/blob/main/assets/%E4%B8%BB%E6%A1%86%E5%9B%BE2_11.png)

**定量结果：** 在不同分割数据集上，同其他SOTA方法的比较

**定性结果：** 可视化分割结果
![image](https://github.com/Sunshine-Ye/IJCV22-SSR/blob/main/assets/%E4%B8%BB%E6%A1%86%E5%9B%BE5_2.png)
![image](https://github.com/Sunshine-Ye/IJCV22-SSR/blob/main/assets/%E4%B8%BB%E6%A1%86%E5%9B%BE6_3.png)
