## Note

目前大致的路线是借用TinyBert的训练方法，首先，阶段一：对文本 encoder 和图像 encoder 单独蒸馏。然后，阶段二：二者在进行原本对比学习任务，使用标签loss + 蒸馏loss。

- CLIP训练是对比学习，缺少标签。因此在最后一层的pred_loss决定修改为最后输出representation的KL散度。
- [原版训练过程小细节](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/general_distill.py#L425)：此处目前没有采用，把attention小于-100的值修改成0
- TinyBert 在计算中间层attention和hidden state的loss时使用了一个矩阵进行线性的映射。
- Tiny模型的 embedding 维度比原始的低，在单独蒸馏一个编码器时，无法使用CLIP中另一种编码器进行指标的计算。（使用投影矩阵投射？去计算一个指标）
- Tiny模型embedding维度比原始的低，那么在最后两个Tiny模型合并的时候，由于缺少标签的监督。效果会不会比较差？具体差多少？这个在阶段二会不会有所改善？
  - 在最后一层加入个Linear，投射到相同空间
- Vit的模型架构中的embedding层或许有所差异。是否蒸馏Embeeding？
- 蒸馏的attention map是均值？还是每一个头都进行求和？
  - 均值的话，attention 头数可以减少，但似乎头数是一个比较重要的参数。

- 或许两个一起蒸馏才能拥有好的效果？
- 使用`nn.Parameter()`的时候一定要初始化，不然会直接`nan`


### On Going
- 编写通用蒸馏模型
  - [x] 获取attention分数
  - [ ] 编写teacher模型加载权重
    - [x] 代码书写
      - [x] CLIP使用的module不支持获取attention map，需要自己写Multi attention结构，二者代码结构上不同，所以权重加载得自己写。
    - [ ] 测试
      - [x] 原版CLIP与加载权重后的CLIP输出结果误差： 1e-5
      - [ ] 在imagenet上的预测效果
  - [ ] CLIP 的Transformer带mask，而TinyBert不使用mask。导致CLIP的attention map存在 -inf。会影响loss计算
    - [x] 将 inf 设置为0
    - [ ] 使用不加mask 的attention进行蒸馏
  - [ ] 各个loss的权重
    - [x] 手动调节
  
  - [ ] 指标的书写，测试结果
  
  

### 模型书写过程

- [ ] forwad 函数  
- [ ] 输出测试
- [ ] init 权重
