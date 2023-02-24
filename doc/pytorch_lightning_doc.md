# Pytorch Lightning Template

本模板旨在希望利用`pytroch-lightning` 内置的功能，帮助我们更好的去做实验。减轻每次实验大部分重复代码的书写（比如log记录等），以及对于平常一些复杂的功能，比如多卡训练，混合精度训练，模型执行时间的分析等采用**声明式**的控制方式。

## 文件结构

```
├─config
├─data
├─model
├─res
└─sh
```

- `model`：存放模型文件
  - `_common.py`：写模型需要的一些组件，比如`Attention`等
  - `_metrics.py`：如果需要自定义指标，写在此处
  - `_utils.py`：需要用到的其他函数，比如记录log函数，或者模型forward相同的部分。
  - `xxxx.py`：写你的模型结构和forward过程。定义`pl.LightningModule`，需要实现几个步骤：
    - `training_step`，模型的`forward`函数，写前向过程和log信息
    - `validation_step`，模型的验证过程。记录指标
-  `data`：存放`dataset`类
  - `yyy.py`：定义`pl.LightningDataModule`，需要实现几个步骤：
    - `prepare_data(self)`，该函数一般用于准备数据，在多卡时只会在一个进程上运行
    - `setup(self, stage)`，该函数依据不同的stage准备相应的数据集
    - `xxx_dataloader(self), xxx = train, test, val, predict`： 返回对应的`Dataloarder`。
- `config`：存放训练用的参数，采用`.yaml`文件进行参数存储
- `res`：默认的结果存放位置
- `sh`：一些可能会用到的脚本函数



## 使用方法

主要采用声明式的思想，当你需要修改组件（优化器，调度器，模型结构，模型参数.....）的时候，修改对应的`.yaml`即可。

### 准备数据

1. 首先你需要构建自己的`dataset` ，写好自己的`dataset.py`。如果是有公开的数据集，直接导入即可。
2. 构建`pl.LightningModule`类
   1. `__init__()`：保存一些需要的变量
   2. 在`prepare_data()`函数中可以写数据集下载，或者获取数据前的一些准备工作。
   3. 在`setup()`函数中构建好训练集，这个是一定需要的。验证集，测试集和预测集都是可选的。
   4. 写`xxxx_dataloader()`函数。返回对应的`Dataloader`格式。

### 准备模型

1. 构建好自己的模型`nn.Module`类
2. 导入需要的模型类，构建好`pl.LightningModule`类
   1. `__init__()`函数，定义好模型，指标
   2. `training_step(self, batch, batch_idx)`，完成一个batch的forward的训练过程。使用`self.log()`记录指标
   3. `validation_step(self, batch, batch_idx)`，完成一个batch的validation过程。记录需要的指标。
   4. 可选：`configure_optimizers(self)`，返回需要的优化器。
   
      > 如果在下面的config文件中配置了优化器，此处的优化器会被覆盖。

### 构建配置文件 

**这里采用最简单的配置文件方法，把所有需要的参数写在一个`.yaml`文件中。**

一般来说，一个配置文件需要包含以下几个参数

- `model`：需要写以下几个参数
  - `class_path`：你使用的`pl.LightningModule`的类名
  - `init_args`：该`pl.LightningModule`初始化所需要的参数
- `data`：需要写以下几个参数
  - class_path：使用的 `pl.LightningDataModule`的类名
  - `init_args`：该`pl.LightningDataModule`初始化需要的参数
- `optimizer`：需要几个参数
  - `class_path`：`optimizer`的torch类名
  - `init_args`：对应的`optimizer`内部包含的参数，比如学习率等。
- `lr_scheduler`：`scheduler`
  - `class_path`：`scheduler`的torch类名
  - `init_args`：对应的`scheduler`内部包含的参数，不同的scheduler有所不同
- `trainer`：`pytorch-lightning`的`trainer`对应的参数。包含所有训练是需要考虑的参数。选择几个较为复杂的参数设置
  - `callbacks`：包含许多功能性函数，可以使用`pytorch-lightning`自带的，也可以自己实现
    - `LearningRateMonitor`：自动log学习率变化
    - `ModelCheckpoint`：自动保存模型

  - `logger`：logger记录器，本项目默认使用`tensorboard`。主要的参数包括:
    - `save_dir`：保存数据的根目录
    - `name`：该次实验的名称
    - `version`：该实验的版本说明。默认为`version_x x = 0, 1, ....`

  - 其余的参数可以参考[这里](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api)


## Quick Start

本模板提供了训练mnist数据集的Resnet模型。

```sh
python main.py fit -c ./config/example.yaml
```

其中 `-c` 指定配置文件路径。



## Work Flow

本章主要讲解调参的Work Flow。其中的核心受启发于`tensorboard`的参数。首先介绍几个概念：

**实验**：实验是使用**一个模型或者一个任务的**训练的过程。

> 选择任务或者模型主要取决于你的需求。
>
> - 比如对一个模型进行调参，或者训练一个模型不同的版本（如Bert base，Bert large）。对于这些，你需要在config中主要调整模型内部的参数或者训练所需参数（学习率...）
> - 比如测试不同的模型架构在一个任务上的表现，此时你需要在config中主要调整的是模型种类。

**版本**：是在实验中使用不同的超参数对模型进行调优，或者使用不同的模型对于同一个任务进行比较。

****

在你设置好数据集和模型架构之后，只需要进行一个config模板编写即可（此处暂命名为`template.yaml`）。我提供了一个模板，包含了trainer大部分可能用的到的参数。



### Config

首先应当注意到，在一个实验中，会包含你不需要调整的参数，比如在测试模型结构的时候，你不会希望去修改`logger`的参数。因此在每一个实验中，需要有一个`share.yaml`的文件控制这一部分的参数。同时对于每一个版本有自己的**独特的参数**。对于这部分的参数，由`version.yaml`控制。

> 本模板采用update的思想，也就是说在`version.yaml`中写的参数会替换掉在`share.yaml`中写的参数。因此在`share.yaml`导入的是完整的参数列表。而我们只需要将**所有需要的参数**都在`share.yaml`指定。之后在`version.yaml`指定所需要的参数以对其进行覆盖即可。
>
> 在本项目中，会提供一个`template.yaml`文件，里面配置了一些简单的内容。

****

本项目提供了几个脚本来方便的实现config配置的书写

#### structure.py  生成文件结构

用于一个实验的config配置的文件结构的生成。其中的`share.yaml`会复制一份已经写好的`template.yaml`。

**parameter**

- `-e, --ex_name`：需要生成实验的名字。
- `-v, --v_num`：需要生成的版本数量。
- `-c, --config`：生成的config文件夹路径。
- `-t, --template`：模板yaml文件路径。



**example**

```sh
python .\sh\structure.py -e resnet -v 3 -c ./config -t ./config/template.yaml
```

**result**

```
│  desc.txt
│  share.yaml
│  
├─version_0
│      detail_desc.txt
│      version.yaml
│      
├─version_1
│      detail_desc.txt
│      version.yaml
│      
└─version_2
        detail_desc.txt
        version.yaml
```

- 对每一个实验会包含一个`desc.txt`阐述实验思路
- 对每一个版本会包含一个`detail_desc.txt`阐述不同版本之间的差异

#### run.py  运行所有脚本

用于运行`config`文件中的实验以及版本。

**parameters**

- `-e, --ex_name`：进行的实验名称，如果需要运行全部的实验，则可以不设置，需要设置`--all_ex`参数
- `-v, --v_num`：需要进行的版本号，如果需要运行全部的版本，则可以不设置，需要设置`--all_ver`参数
- `-c, --config`：config文件夹的根目录，默认：`./config`
- `--all_ver`：运行所有实验，当这个参数设置时，其他的参数都失效。
- `--all_ex`：运行所有的版本，只当指定了`-e, --ex_name`的时候才会生效
- `-b, --begin_ver`：需要开始运行的版本号，需要搭配`-t, --end_ver`，默认为0
- `-t, --end_ver`：需要运行的最后一个版本号，需要搭配`-b, --end_ver`，默认为最后一个版本
- `-n`：需要运行的版本号
- `-o, --other_para`：需要补充的其他参数

目前支持以下几种模式：

- 运行所有的实验以及内部的版本：

```sh
python ./sh/run.py --all_ex
```

- 运行指定实验的所有版本：

 ```sh
 python ./sh/run.py -e ex_name --all_ex
 ```

- 运行单个实验和指定版本：运行`ex_name`的`version_2`内部的config
```python
python ./sh/run.py -e ex_name -v 2
```

- 运行单个实验多个指定版本：运行`ex_name`中0，1，3，18版本

```
python ./sh/run.py -e ex_name -n 0 1 3 18 
```

- 运行单个实验多个连续的版本：运行ex_name中[0-10)的版本
  - Tips：当-b或者-t只有一个存在时，另一个会采用默认值。同时注意是左闭右开区间！

```
python ./sh/run.py -e ex_name -b 0 -t 10
```





#### 超参生成脚本 （TODO）

- 指定需要调整的超参数，以及给定的超参列表，生成对应的config文件
- 可以支持网格搜索方法。











## 使用样例

这里介绍一些`pytorch-lightning`常用的一些操作

### 基本使用

**一次完整的实验流程应该包括以下几个步骤**：

1. 数据集，模型代码编写，具体参照上文中的**使用方法**。数据集最好分成训练集，验证集，测试集
2. 依据需求编写config文件。
3. 模型训练，验证。参数调整。
4. 结果测试。

****







### 多卡训练

> 本篇文章只讨论GPU！其他的卡作者本人没有用过。

多卡训练在`pytorch-lightning`十分简单，只需要在config文件中的`trainer`指定`devices`以及多卡训练的`strategy`。具体的细节可以参考这里[gpus选择](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html#train-on-multiple-gpus)和[strategy](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_intermediate.html#distributed-training-strategies)

#### 自动挑选gpu设置

```yaml
# 自动选择`k`个合适的gpu，使用`ddp`策略。
trainer:
  accelerator: 'gpu'
  auto_select_gpus: True
  devices: k
  strategy: 'ddp'
```

****

#### 选择指定的gpu

```yaml
# 选择1, 3, 5号gpu进行训练，采用`ddp`策略。
trainer:
  accelerator: 'gpu'
  auto_select_gpus: True
  devices: 
    - 1
    - 3
    - 5
  strategy: 'ddp'
  
# or 
trainer:
  accelerator: 'gpu'
  auto_select_gpus: True
  devices: "1, 3, 5"
  strategy: 'ddp'
```

****

#### 选择所有GPU

```yaml
trainer:
  accelerator: 'gpu'
  auto_select_gpus: True
  devices: -1
  strategy: 'ddp'
# or
trainer:
  accelerator: 'gpu'
  auto_select_gpus: True
  devices: "-1"
  strategy: 'ddp'
```

**注意，当前最新版本真正做到了自动选择空闲GPU，但目前我没找到yaml文件如何设置，因此最新版本代码是在main.py中设置的**

```python
cli = MyLightningCLI(seed_everything_default=2022, save_config_overwrite=True,
                     trainer_defaults={'devices': find_usable_cuda_devices(4)})
# 寻找四张空闲卡
```







### Debug

#### 快速运行模型所有训练代码

**`fast_dev_run`**：参数通过`Trainer`运行5个batch的训练、验证、测试和预测数据，以查看是否存在任何错误:

```yaml
# run default 5 batch
trainer:
  fast_dev_run: True
# or run 7 batch
trainer:
  fast_dev_run: 7
```

**这个参数会使得tuner, checkpoint callbacks, earlystopping callbacks, logger, logger callbacks(LearningRateMonitor, DeviceStateMonitor)**失效

#### 缩小epoch长度

```yaml
# use only 10% of training data and 1% of val data
trainer:
  limit_train_batches: 0.1
  limit_val_batches: 0.01
    
# use 10 batches of train and 5 batches of val
trainer:
  limit_train_batches: 10
  limit_val_batches: 5
```

#### Run a Sanity Check

- Lightning 在开始训练前会自动运行两次验证函数。这样可以避免在有时深入到冗长的训练循环中的验证循环中崩溃。

```yaml
# default is 2, you can change it.
trainer:
  num_sanity_val_steps: 10
```

#### 打印模型的weights summary

- 这会生成一个这样的表格

```
  | Name  | Type        | Params
----------------------------------
0 | net   | Sequential  | 132 K
1 | net.0 | Linear      | 131 K
2 | net.1 | BatchNorm1d | 1.0 K
```

- 为了监控子模块需要添加 [`ModelSummary`](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.ModelSummary.html#pytorch_lightning.callbacks.ModelSummary):

```python
from pytorch_lightning.callbacks import ModelSummary
trainer = Trainer(callbacks=[ModelSummary(max_depth=-1)])
```

- 关闭该信息的输出

```python
Trainer(enable_model_summary=False)
```

#### 打印每一层输出维度

**需要在`LightningModule`中实现`forward()`的方法**

- 需要在模型中加入一个 `self.example_input_array`

```python
class LitModel(LightningModule):
    def __init__(self, *args, **kwargs):
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
    def forward(self, x):
        x = ....
```

- 在使用 `.fit`函数后会自动输出

```
  | Name  | Type        | Params | In sizes  | Out sizes
--------------------------------------------------------------
0 | net   | Sequential  | 132 K  | [10, 256] | [10, 512]
1 | net.0 | Linear      | 131 K  | [10, 256] | [10, 512]
2 | net.1 | BatchNorm1d | 1.0 K  | [10, 512] | [10, 512]
```



### 找到训练循环中的bottleneck

#### 简单信息记录

```python
trainer = Trainer(profiler="simple")
```

- `simple profiler`自动测量训练循环中使用的所有标准方法
  - on_train_epoch_start
  - on_train_epoch_end
  - on_train_batch_start
  - model_backward
  - on_after_backward
  - optimizer_step
  - on_train_batch_end
  - training_step_end
  - on_training_end
  - etc…

#### 记录每一个函数的运行时间

```python
trainer = Trainer(profiler="advanced")
```

- 如果信息过长，可以输出到文件中查看

```python
from pytorch_lightning.profiler import AdvancedProfiler

profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
trainer = Trainer(profiler=profiler)

```



在`.yaml`文件中这么设置即可：

```yaml
trainer:
# 这是详细信息输出
#  profiler:
#    class_path: pytorch_lightning.profiler.AdvancedProfiler
#    init_args:
#      dirpath: './'
#      filename: 'profile.log'
# 这是简易信息输出
  profiler:
    class_path: pytorch_lightning.profiler.SimpleProfiler
    init_args:
      dirpath: './'
      filename: 'profile.log'
```



#### 检测硬件

- 另一个有用的瓶颈检测技术是确保您使用了加速器的全部容量(GPU/TPU/IPU/HPU)。这可以通过 `DeviceStatsMonitor` 来测量:

```python
from pytorch_lightning.callbacks import DeviceStatsMonitor

trainer = Trainer(callbacks=[DeviceStatsMonitor()])
```

- CPU metrics will be tracked by default on the CPU accelerator. To enable it for other accelerators set `DeviceStatsMonitor(cpu_stats=True)`. To disable logging CPU metrics, you can specify `DeviceStatsMonitor(cpu_stats=False)`.



在`./yaml`这么设置即可：

```yaml
trainer:
  callbacks:
    - class_path: DeviceStatsMonitor
      init_args:
        cpu_stats: True
    - class_path: LearningRateMonitor
```



## Trouble shooting

- 对于每个不同的数据集相同的模型可能需要不同的输入

[两个optimizer的设置：](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html#optimizers-and-learning-rate-schedulers)

[如何手动优化](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#automatic-optimization)

 [训练过程](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#hooks)

