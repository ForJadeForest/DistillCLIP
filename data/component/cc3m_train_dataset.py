"""
This file comes from
https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/data.py
"""
import ast
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    shards_list = shards
    dir_path = os.path.dirname(shards[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum(
            [
                int(sizes[os.path.basename(shard)])
                if os.path.basename(shard) in sizes
                else 0
                for shard in shards_list
            ]
        )
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
        print(n_batches, n_elements)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    return ("txt" in sample) and (
            "png" in sample or "jpg" in sample or "jpeg" in sample
    )


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if "No images in sample" in str(exn) or "Only one image in sample" in str(
            exn
    ):  # Avoid spamming logs with these
        return True
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
        data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
                current_sample is None
                or prefix != current_sample["__key__"]
                or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
            self,
            urls,
            nshards=sys.maxsize,
            worker_seed=None,
            deterministic=False,
            epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                # 确保dataloader每一个worker的seed不同
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            # 产生一个url
            yield dict(url=self.rng.choice(self.urls))


def get_cc3m_dataset(cc3m_shards, batch_size, num_workers,
                     cc3m_dataset_length=2905954, epoch=0, floor=False, need_text_processor=True):
    from .utils import make_transformers
    from clip import tokenize
    from torch import distributed as dist

    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    image_processor = make_transformers(True)
    text_processor = tokenize
    input_shards = cc3m_shards
    num_samples = cc3m_dataset_length
    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)

    pipeline = [
        ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
    ]

    # at this point we have an iterator over all the shards
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.decode("pil"),
            wds.to_tuple("jpg", "txt", handler=log_and_continue),
        ]
    )
    if need_text_processor:
        pipeline.extend(
            [
                wds.map(lambda x: (image_processor(x[0]), text_processor(x[1], truncate=True).squeeze())),
                wds.batched(batch_size, partial=False),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.map(lambda x: (image_processor(x[0]), x[1])),
                wds.batched(batch_size, partial=False),
            ]
        )

    dataset = wds.DataPipeline(*pipeline)
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = batch_size * world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, num_workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)
    dataset = dataset.with_length(num_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader


if __name__ == '__main__':
    shards_urls = '/data/pyz/data/cc/train_cc3m/{00000..00331}.tar'
    image_transformers = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    from clip import tokenize

    cc3m_dataloader = get_cc3m_dataset(shards_urls, image_transformers, tokenize, batch_size=128, num_workers=12,
                                       world_size=1).dataloader
    print(count_samples(cc3m_dataloader))
    # output: (2906112, 22704)
