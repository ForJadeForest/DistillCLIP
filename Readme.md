# L-CLIPScore
This repo is for L-CLIPScore: a lightweight Embedding-based Captioning Metric for Evaluating and Training



## Data
1. We use the mscoco2017 and imagenet1M to train our image encoder in the first stage. 
   - You should use a script to move the data in the same folder.

2. We use the Conceptual Caption and mscoco2017 captions as the text data. 

3. We use the mscoco2017 for the L-CLIP model to learn the multimodal information.

If you want to run all model: you should have these data:
```yaml
data:
  mscoco:
    annotations
    val2017
  cc:
    train_cc3m.tsv
  # the folder contain imagenet1M and mscoco2017 train dataset.
  combine_data:
    000.png
    001.png
    ...

```

## Run
### For reproduction

1. image encoder 
First, you should set the data path in the `image.yaml` file.
  - the `data.prepare_para.raw_data_dir` is the mscoco2017 data path, this is to load the val image and caption.
  - the `data.dataset_para.combine_dataset_path` is the path to a folder that contain all mscoco2017 and imagenet1M data.
```shell
python main.py fit --conf config/final_config/image.yaml
```

2. text encoder
First, you should set the `raw_data_dir` contain the cc folder and mscoco folder. 
```python
# the path you should set:
coco2017_file = raw_data_dir / 'mscoco' / 'annotations' / 'captions_train2017.json'
cc_file = raw_data_dir / 'cc' / 'train_cc3m.tsv'
```

```shell
python main.py fit --conf config/final_config/text.yaml
```

3. L-CLIP model
First, you should set `data.dataset_para.root_path` and `data.dataset_para.annotation_path`
```yaml
# For example
root_path: '~/mscoco'
annotation_path: '~/mscoco/annotations'
```
```shell
python main.py fit --conf config/final_config/l_clip.yaml
```


### For Research



1. Use a `config.yaml` file
```shell
python main.py fit --conf path/to/config.yaml
```
2. use the `sh/run.py`

Tips: Please use the `sh/ex.py` to create the experiment folder.

```shell
python sh/run.py -e Ex_name -v 0
```

### run.py  run all version

Used to run experiments and versions in the `./config` file.

Parameters:

- -e, --ex_name: The name of the experiment to be run. If you want to run all experiments, you can leave it blank and use the --all_ex parameter.
- -v, --v_num: The version number to be run. If you want to run all versions, you can leave it blank and use the --all_ver parameter.
- -c, --config: The root directory of the config folder. Default is ./config.
- --all_ver: Run all experiments. When this parameter is set, other parameters will be ignored.
- --all_ex: Run all versions. This only works when the -e, --ex_name parameter is specified.
- -b, --begin_ver: The starting version number to run. It needs to be used with -t, --end_ver and the default value is 0.
- -t, --end_ver: The last version number to be run. It needs to be used with -b, --begin_ver and the default value is the last version.
- -n: The version number to be run.
- -o, --other_para: Other parameters to be added.

Currently, the following modes are supported:

- Run all experiments and internal versions:
```shell
python ./sh/run.py --all_ex
```


- Run all versions of a specific experiment:
```shell
python ./sh/run.py -e ex_name --all_ex
```


- Run a single experiment and a specific version: Run the config of version_2 in ex_name
```shell
python ./sh/run.py -e ex_name -v 2
```


- Run multiple specified versions of a single experiment: Run versions 0, 1, 3, and 18 in ex_name
```shell
python ./sh/run.py -e ex_name -n 0 1 3 18 
```


- Run multiple consecutive versions of a single experiment: Run the versions [0-10) in ex_name
  - Tips: When only -b or -t exists, the other one will use the default value. Also, note that this is a left-closed and right-open interval.
```shell
python ./sh/run.py -e ex_name -b 0 -t 10
```
