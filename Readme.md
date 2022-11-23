## Note

目前大致的路线是借用TinyBert的训练方法，首先，阶段一：对文本 encoder 和图像 encoder 单独蒸馏。然后，阶段二：二者在进行原本对比学习任务，使用标签loss + 蒸馏loss。

- CLIP训练是对比学习，缺少标签。因此在最后一层的pred_loss决定修改为最后输出representation的KL散度。
- [原版训练过程小细节](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/general_distill.py#L425)：此处目前没有采用，把attention小于-100的值修改成0
- TinyBert 在计算中间层attention和hidden state的loss时使用了一个矩阵进行线性的映射。
- ~~Tiny模型embedding维度比原始的低，那么在最后两个Tiny模型合并的时候，由于缺少标签的监督。效果会不会比较差？具体差多少？这个在阶段二会不会有所改善？~~
  - 在两个encoder最后一层加入个Linear，投射到相同空间
- Vit的模型架构中的embedding层或许有所差异。是否蒸馏Embeeding？
- 蒸馏Embedding是否加position embedding
  - 目前采用加的方法
- 蒸馏的attention map是均值？还是每一个头都进行loss计算？
  - 均值的话，attention 头数可以减少，但似乎头数是一个比较重要的参数。
  - 目前使用计算每一个头的loss
- 或许两个一起蒸馏才能拥有好的效果？
- 图像数据较少，蒸馏后对于一些没有出现在train dataset中的物体结果比较差
  - 由于图像和文本是对齐的，能否直接用文本的dataset去蒸馏图像的Encoder？ (input 无法确定)
- 只使用mscoco数据蒸馏很难。如果加上hard label loss的话模型的bias会比较高。mean_score可以达到0.6，teacher只有0.3
  - 为什么mean_score可以到0.6但是准确度不高呢？

### bug fix
- [x] teacher model 没有使用 `.eval()`
- [x] 测试model load
- [x] 因为copy的小错误，导致ImageEncoder构建的时候，`attention head num` 出现错误调试bug一个晚上。警钟长鸣
- [x] 测试teacher的mode
- [ ] config 的更新能否细化
- [x] 目前训练花费时间比较长。ImageEncoder一个epoch ~ 35min，TextEncoder ~ 25min。
  - [x] ~~将图像存为lmdb格式~~
    - 似乎没什么用
  - [x] 将data存放到home下
  - [x] pin_memory 参数设置
  - [x] 提前做好数据裁切
- [ ] 多GPU val metric计算有问题. 目前来看和batch和GPU数量有关.
  - 在不能整除batch的时候,剩下的部分会由GPU平分,导致最后一个batch的准确率很高,聚合会提高整体的准确率
  - 5000 - 4 * 1000 = 1000 剩下的1000会由GPU评分,导致最后一个的实际batch是250
  - 在val_epoch_end处收集数据进行计算(但是目前还是存在问题: 无论单GPU还是多GPU,在不同的val batch size的时候会出现不同的结果,在0.3474 ~ 0.3478波动)
    
    - 使用16 precision + 多卡


### Config
#### dataset
依据之前的经验，数据量需要达到一定的程度蒸馏效果才会比较好。但目前缺少大量的图像文本对。这也是采用分开蒸馏的原因之一
- ImageDateset: MSCOCO + Caltech 256 + Imagenet

![image-20220731142222754](https://jadepicgo.oss-cn-shenzhen.aliyuncs.com/img/image-20220731142222754.png)

- TextDataset: MSCOCO + Conceptual Captions

![image-20220731142303147](https://jadepicgo.oss-cn-shenzhen.aliyuncs.com/img/image-20220731142303147.png)



## Ex

1. 32精度效果比较好

2. loss 搭配：

   - 只使用 `l1 + cos` 
   - 使用`l1 + cos + kl`
   - 使用 `value + attn_probs`
   - 使用 `rep + attn_probs + emb`
   - 使用 `l1 + cos + vit_kd`
3. 数据
   - 使用 data_256 + imagenet + MSCOCO
   - 不用 data_256
   - 只使用 MSCOCO

4. 模型架构
   暂定这四个模型

   - 减少层数，不减少宽度 ，增加头数(768 width + 4 layers + 24 heads) **效果最好**
   - 减少宽度，减少层数，增加头数 (768 // 2 width + 4 layers + 24 heads)
   - 减少层数，不减少宽度 ，不增加头数(768 width + 4 layers + 12 heads) **做为基础模型**
   - 减少宽度，减少层数，不增加头数 (768 // 2 width + 4 layers + 12 heads)
   - Wight Share (768 width + 4 layers + 24 heads + 2 repeat)


5. Deit 提到dropout会导致性能损失（确实如此）
6. 将最后的输出norm去做蒸馏



### On Going
- 编写通用蒸馏模型
  - [x] 获取attention分数
  - [x] 编写teacher模型加载权重
    - [x] 代码书写
      - [x] CLIP使用的module不支持获取attention map，需要自己写Multi attention结构，二者代码结构上不同，所以权重加载得自己写。
    - [x] 测试
      - [x] 原版CLIP与加载权重后的CLIP输出结果误差： 1e-5
  - [ ] CLIP 的Text Transformer带mask，而TinyBert不使用mask。导致CLIP的attention map存在 -inf。会影响loss计算
    - [x] 将 inf 设置为0
    - [ ] 使用不加mask的attention进行蒸馏
  - [ ] 各个loss的权重
    - [x] 手动调节
  - [x] 指标的书写，测试结果
  - [x] 模型蒸馏的速度评估
    - [参考指标与计算方法](https://zhuanlan.zhihu.com/p/376925457)
  - [ ] 计划在下载好 Conceptual Captions的Image的时候，可以考虑一起蒸馏。或者单纯增加图像的数量。大概达到6M的样子。
  - [x] 使用RandAugmentation进行增强。
  - [ ] **Fine-Grain loss 添加**
    - FICLP model采用的方法需要的是匹配的文本图像对，需要重新选择数据集。同时对比学习的效果不一定好。（batch 太小，使用梯度积累？）
      - [多GPU DDP 计算对比学习的损失](https://github.com/Lightning-AI/lightning/discussions/14390)
  - [ ] Weight share model 暂时不支持选用特定的layers蒸馏，所选的是全部层
  - [ ] 将logits图show出来
  
  

