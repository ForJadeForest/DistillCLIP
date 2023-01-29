'''preprocesses the composite corpus.

'''
# from google_drive_downloader import GoogleDriveDownloader as gdd
import os
# import tqdm
import collections
import json
import numpy as np
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
    composite_image2ann = collections.defaultdict(list)
    captionid2caption = {}

    with open('composite.json', 'w') as f:
        for corpus in ann_list:
            all_index = {}
            with open(corpus, 'rb') as csvfile:
                all_corpus = []
                reader = csv.reader(csvfile)
                raw_ann = [row[3] for row in reader]
                for item in raw_ann:
                    real_ann = item.split("http")[1].split(";")
                    all_corpus.append({
                        'image_id': real_ann[0].split("/")[-1],
                        'image_path': 'http'+ real_ann[0],
                        'candidate1' : real_ann[1],
                        'candidate2' : real_ann[2],
                        'candidate3' : real_ann[3],
                        'rating1' : real_ann[4],
                        'rating2' : real_ann[5],
                        'rating3' : real_ann[6]
                    })
                for d in all_corpus:
                    if d['image_id'] not in all_index:
                        all_index[d['image_id']] = {
                            'human_judgement': [],
                            'image_id': d['image_id'],
                            'image_path': d['image_path']}

                    all_index[d['image_id']]['human_judgement'].append(
                        {'image_id': d['image_id'],
                         'image_path': d['image_path'],
                         'caption': d['candidate1'],
                         'rating': d['rating1']})
                    all_index[d['image_id']]['human_judgement'].append(
                        {'image_id': d['image_id'],
                         'image_path': d['image_filepath'],
                         'caption': d['candidate2'],
                         'rating': d['rating2']})
                    all_index[d['image_id']]['human_judgement'].append(
                        {'image_id': d['image_id'],
                         'image_path': d['image_filepath'],
                         'caption': d['candidate3'],
                         'rating': d['rating3']})

            print('For expert, we are dumping {} judgments between {} images'.format(
            len(all_corpus)*3,
            len(all_index)))
            f.write(json.dumps(all_index))


def main():
    if not os.path.exists('composite.json'):
        process_composite()

if __name__ == '__main__':
    main()
