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

def process_composite():
    '''
    # download dataset

    if not os.path.exists('compositeDataset'):
        os.makedirs('compositeDataset')

    if not os.path.exists('compositeDataset'):
        gdd.download_file_from_google_drive(file_id='1WNY8pV-u8xtBYBVal03qwjQs4VKurUZn',
                                            dest_path='./flickr8k/Flickr8k_Dataset.zip',
                                            unzip=True,
                                            showsize=True)
    if not os.path.exists('flickr8k/Flickr8k_text.zip'):
        gdd.download_file_from_google_drive(file_id='1ljB7DR-YM-q9WKnHDW5dHjauK029B2s6',
                                            dest_path='./flickr8k/Flickr8k_text.zip',
                                            unzip=True,
                                            showsize=True)
    '''

    # composite #
    ann_list = ['8k_correctness.csv', '8k_throughness.csv', '30k_correctness.csv', '30k_throughness.csv',
               'coco_correctness.csv', 'coco_throughness.csv']

    # flickr8k #
    flickr8k_image2ann = collections.defaultdict(list)
    captionid2caption = {}

    with open('C:\\Users\\86189\\Desktop\\Deep learning\\metrics advancement\\clipscore-main\\flickr8k_example\\Flickr8k_text\\Flickr8k.token.txt') as f:
        for line in f:
            image, ann = line.strip().split('\t')
            flickr8k_image2ann[image.split('#')[0]].append(ann)
            captionid2caption[image] = ann

    flickr8k_image2ann = {k.split('.')[0]: v for k, v in flickr8k_image2ann.items()}
    print("8k is read")

    # flickr30k #
    flickr30k_image2ann = collections.defaultdict(list)

    annotations = pd.read_table('C:\\Users\\86189\\Desktop\\Deep learning\\metrics advancement\\composite\\results_20130124.token', sep='\t', header=None,
                                names=['image', 'caption'])

    for idx, line in enumerate(annotations['image']):
        flickr30k_image2ann[line.split('#')[0]].append(annotations['caption'][idx])
    flickr30k_image2ann = {k.split('.')[0]: v for k, v in flickr30k_image2ann.items()}
    print("30k is read")


    # coco #
    mscoco_image2ann = collections.defaultdict(list)

    with open("C:\\Users\\86189\\Desktop\\Deep learning\\metrics advancement\\composite\\captions_val2014.json") as f:
        data = {}
        data.update(json.load(f))
        for line in data['annotations']:
            mscoco_image2ann['COCO_val2014_' + str(line['image_id']).zfill(12) + '.jpg'].append(line['caption'])

    mscoco_image2ann = {k.split('.')[0]: v for k, v in mscoco_image2ann.items()}
    print("coco is read")

    with open('C:\\Users\\86189\\Desktop\\Deep learning\\ijcai22\\DistillCLIP\\test\\composite.json', 'w') as f:
        for corpus in ann_list:
            print("now process {}".format(corpus))
            all_index = {}
            skip = 0
            with open('C:\\Users\\86189\\Desktop\\Deep learning\\metrics advancement\\composite\\' + corpus, 'rt') as csvfile:
                all_corpus = []
                reader = csv.reader(csvfile)
                header = next(reader)
                raw_ann = [row[3] for row in reader]
                for item in raw_ann:
                    real_ann = item.split("http")[1].split(";")
                    if corpus[:2] == '8k':
                        all_corpus.append({
                            'image_id': real_ann[0].split("/")[-1].split(".")[0],
                            'image_path': 'http'+ real_ann[0],   # here need to be updated in real image dataset
                            'candidate1' : real_ann[1],
                            'candidate2' : real_ann[2],
                            'candidate3' : real_ann[3],
                            'rating1' : real_ann[4],
                            'rating2' : real_ann[5],
                            'rating3' : real_ann[6]
                        })
                    else:
                        all_corpus.append({
                            'image_id': real_ann[0].split("/")[-1].split(".")[0],
                            'image_path': 'http' + real_ann[0],  # here need to be updated in real image dataset
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

            print('For 8k,we skip {} ground truth.(it should be 158)'.format(skip))
            print('For expert, we are dumping {} judgments between {} images'.format(
            len(all_corpus)*3,
            len(all_index)))
            f.write(json.dumps(all_index))


def main():
    if not os.path.exists('composite.json'):
        process_composite()

if __name__ == '__main__':
    main()
