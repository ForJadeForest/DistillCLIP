## Note

目前大致的路线是借用TinyBert的训练方法，首先对文本 encoder 和图像 encoder 单独蒸馏，然后二者在进行原本对比学习任务，使用标签loss + 蒸馏loss。

- CLIP训练是对比学习，缺少标签。因此在最后一层的pred_loss决定修改为最后输出representation的KL散度。
- Vit的模型架构中的embedding层或许有所差异。是否蒸馏Embeeding？
- [原版训练过程小细节](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/general_distill.py#L425)：此处目前没有采用，把attention小于-100的值修改成0

- 蒸馏的attention是没有经过softmax的



### Trouble

- [x] 获取attention分数
- [ ] 

