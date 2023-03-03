'''preprocesses the composite corpus.
each csv need to delete the last two rows
'''
# from google_drive_downloader import GoogleDriveDownloader as gdd
import os
# import tqdm
import collections
import json
# import numpy as np
import pandas as pd
import csv


def process_composite(root_dir, flickr8k_dir, coco_dir, flickr30k_dir):
    # download dataset
    # if not os.path.exists('compositeDataset'):
    #     os.makedirs('compositeDataset')
    #
    # if not os.path.exists('compositeDataset'):
    #     gdd.download_file_from_google_drive(file_id='1WNY8pV-u8xtBYBVal03qwjQs4VKurUZn',
    #                                         dest_path='./flickr8k/Flickr8k_Dataset.zip',
    #                                         unzip=True,
    #                                         showsize=True)
    # if not os.path.exists('flickr8k/Flickr8k_text.zip'):
    #     gdd.download_file_from_google_drive(file_id='1ljB7DR-YM-q9WKnHDW5dHjauK029B2s6',
    #                                         dest_path='./flickr8k/Flickr8k_text.zip',
    #                                         unzip=True,
    #                                         showsize=True)

    # composite
    ann_list = ['8k_correctness.csv', '30k_correctness.csv', 'coco_correctness.csv']

    # flickr8k #
    flickr8k_image2ann = collections.defaultdict(list)
    captionid2caption = {}
    flickr8k_file_path = os.path.join(root_dir, 'Flickr8k.token.txt')
    with open(flickr8k_file_path) as f:
        for line in f:
            image, ann = line.strip().split('\t')
            flickr8k_image2ann[image.split('#')[0]].append(ann)
            captionid2caption[image] = ann

    flickr8k_image2ann = {k.split('.')[0]: v for k, v in flickr8k_image2ann.items()}
    print("8k is read")

    # flickr30k #
    flickr30k_image2ann = collections.defaultdict(list)

    data30k_file_path = os.path.join(root_dir, 'results_20130124.token')
    annotations = pd.read_table(data30k_file_path, sep='\t', header=None, names=['image', 'caption'])

    for idx, line in enumerate(annotations['image']):
        flickr30k_image2ann[line.split('#')[0]].append(annotations['caption'][idx])
    flickr30k_image2ann = {k.split('.')[0]: v for k, v in flickr30k_image2ann.items()}
    print("30k is read")

    mscoco_image2ann = collections.defaultdict(list)
    coco_file_path = os.path.join(root_dir, 'captions_val2014.json')
    with open(coco_file_path) as f:
        data = {}
        data.update(json.load(f))
        for line in data['annotations']:
            mscoco_image2ann['COCO_val2014_' + str(line['image_id']).zfill(12) + '.jpg'].append(line['caption'])

    mscoco_image2ann = {k.split('.')[0]: v for k, v in mscoco_image2ann.items()}
    print("coco is read")
    composite_path = os.path.join(root_dir, 'composite.json')
    with open(composite_path, 'w') as f:
        all_index = {}
        for corpus in ann_list:
            print("now process {}".format(corpus))
            skip = 0
            corpus_path = os.path.join(root_dir, corpus)
            with open(corpus_path, 'rt') as csvfile:
                all_corpus = []
                reader = csv.reader(csvfile)
                header = next(reader)
                raw_ann = [row[3] for row in reader]
                for item in raw_ann:
                    real_ann = item.split("http")[1].split(";")
                    if corpus[:2] == '8k':
                        all_corpus.append({
                            'image_id': real_ann[0].split("/")[-1].split(".")[0],
                            'image_path': os.path.join(flickr8k_dir, real_ann[0].split("/")[-1]),
                            # here need to be updated in real image dataset
                            'candidate1': real_ann[1],
                            'candidate2': real_ann[2],
                            'candidate3': real_ann[3],
                            'rating1': real_ann[4],
                            'rating2': real_ann[5],
                            'rating3': real_ann[6]
                        })
                    elif corpus[:2] == 'co':
                        all_corpus.append({
                            'image_id': real_ann[0].split("/")[-1].split(".")[0],
                            'image_path': os.path.join(coco_dir, real_ann[0].split("/")[-1]),
                            # here need to be updated in real image dataset
                            'candidate1': real_ann[1],
                            'candidate2': real_ann[2],
                            'candidate3': real_ann[3],
                            'candidate4': real_ann[4],
                            'rating1': real_ann[5],
                            'rating2': real_ann[6],
                            'rating3': real_ann[7],
                            'rating4': real_ann[8]
                        })
                    else:
                        all_corpus.append({
                            'image_id': real_ann[0].split("/")[-1].split(".")[0],
                            'image_path': os.path.join(flickr30k_dir, real_ann[0].split("/")[-1]),
                            # here need to be updated in real image dataset
                            'candidate1': real_ann[1],
                            'candidate2': real_ann[2],
                            'candidate3': real_ann[3],
                            'candidate4': real_ann[4],
                            'rating1': real_ann[5],
                            'rating2': real_ann[6],
                            'rating3': real_ann[7],
                            'rating4': real_ann[8]
                        })
            for d in all_corpus:
                if d['image_id'] not in all_index:
                    if corpus[:2] == 'co':
                        ground_truth = mscoco_image2ann[d['image_id']]
                    elif corpus[:2] == '8k':
                        ground_truth = flickr8k_image2ann[d['image_id']]
                    else:
                        ground_truth = flickr30k_image2ann[d['image_id']]
                    all_index[d['image_id']] = {
                        'human_judgement': [],
                        'image_id': d['image_id'],
                        'image_path': d['image_path'],
                        'ground_truth': ground_truth}
                if corpus[:2] == '8k':
                    if d['candidate1'] in flickr8k_image2ann[d['image_id'].split('.')[0]]:
                        skip += 1
                    else:
                        all_index[d['image_id']]['human_judgement'].append(
                            {'image_id': d['image_id'],
                             'image_path': d['image_path'],
                             'caption': d['candidate1'],
                             'rating': float(d['rating1'])})
                    if d['candidate2'] in flickr8k_image2ann[d['image_id'].split('.')[0]]:
                        skip += 1
                    else:
                        all_index[d['image_id']]['human_judgement'].append(
                            {'image_id': d['image_id'],
                             'image_path': d['image_path'],
                             'caption': d['candidate2'],
                             'rating': float(d['rating2'])})
                    if d['candidate3'] in flickr8k_image2ann[d['image_id'].split('.')[0]]:
                        skip += 1
                    else:
                        all_index[d['image_id']]['human_judgement'].append(
                            {'image_id': d['image_id'],
                             'image_path': d['image_path'],
                             'caption': d['candidate3'],
                             'rating': float(d['rating3'])})
                else:
                    all_index[d['image_id']]['human_judgement'].append(
                        {'image_id': d['image_id'],
                         'image_path': d['image_path'],
                         'caption': d['candidate1'],
                         'rating': float(d['rating1'])})
                    all_index[d['image_id']]['human_judgement'].append(
                        {'image_id': d['image_id'],
                         'image_path': d['image_path'],
                         'caption': d['candidate2'],
                         'rating': float(d['rating2'])})
                    all_index[d['image_id']]['human_judgement'].append(
                        {'image_id': d['image_id'],
                         'image_path': d['image_path'],
                         'caption': d['candidate3'],
                         'rating': float(d['rating3'])})
                    all_index[d['image_id']]['human_judgement'].append(
                        {'image_id': d['image_id'],
                         'image_path': d['image_path'],
                         'caption': d['candidate4'],
                         'rating': float(d['rating4'])})

            print('For expert, we are dumping {} judgments between {} images'.format(
                len(all_corpus) * 3,
                len(all_index)))
        f.write(json.dumps(all_index))


def main():
    root_dir = '/data/ll/composite'
    coco_dir = '/data/pyz/data/mscoco/val2014/'  # the path to val2014
    flickr30k_dir = '/data/ll/composite/flickr30k-images/'
    flickr8k_dir = '/data/pyz/data/flickr8k/Flickr8k_Dataset/'
    if not os.path.exists(os.path.join(root_dir, 'composite_correctness.json')):
        process_composite(root_dir, flickr8k_dir, coco_dir, flickr30k_dir)


if __name__ == '__main__':
    main()
