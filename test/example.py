import json
from pathlib import Path

from test.utils.load_model import load_model

with open('/data/pyz/data/mscoco/annotations/captions_val2014.json', 'r') as f:
    ref_data = json.load(f)['annotations']
id2ref = {}
for d in ref_data:
    if d['image_id'] in id2ref:
        id2ref[d['image_id']].append(d['caption'])
    else:
        id2ref[d['image_id']] = [d['caption']]

images_root_dir = Path(r'/data/pyz/data/mscoco/val2014')


def get_image(images_root_dir):
    images_id_list = sorted(list(images_root_dir.iterdir()))
    return images_id_list


def filename2id(filename):
    import re
    pattern = re.compile('COCO_val2014_0*(.*).jpg')
    return int(re.findall(pattern, str(filename))[0])


image_file = get_image(images_root_dir)
refs = [id2ref[filename2id(file)] for file in image_file]
can = [i[0] for i in refs]

can[0] = 'A footwear stand with several shoes and a resting dog atop.'
can[1] = 'A vintage motorbike is parked next to other motorcycles with a brown leather saddle.'
can[2] = 'The white dog lies beside the bike on the pavement.'
can[3] = 'A double-decker bed with a slim shelf located beneath it.'


can[0] = 'A cat walking on a show rack in the shoes.'
can[1] = 'An old bicycle parked beside other motorcycles with a red leather seat.'
can[2] = 'A cat rests on the street next to a bicycle.'
can[3] = 'A bed and table in a big room.'

cider = load_model('cider')
cider_res = cider(None, can, refs)

l_clip_model = load_model('L-CLIP', 'cuda')

l_clip_score = l_clip_model(image_file, can, refs)

print('mean cider score: ', sum(cider_res['ref_score'][4:]) / len(cider_res['ref_score'][4:]))
print(cider_res['ref_score'][:4])

print('mean ref clip score: ', sum(l_clip_score['ref_score'][4:]) / len(l_clip_score['ref_score'][4:]))
print(l_clip_score['ref_score'][:4])

print('mean clip score: ', sum(l_clip_score['score'][4:]) / len(l_clip_score['score'][4:]))
print(l_clip_score['score'][:4])

for i in range(4):
    print('## refs: ')
    print(refs[i])
    print()
    print('## candidate: ')
    print(can[i])
    print()
    print('## res')
    print(f'cider: {cider_res["ref_score"][i]} / {sum(cider_res["ref_score"][4:]) / len(cider_res["ref_score"][4:])}')
    print(f'clip-s: {l_clip_score["score"][i]} / {sum(l_clip_score["score"][4:]) / len(l_clip_score["score"][4:])}')
    print(f'ref-cider: {l_clip_score["ref_score"][i]} / {sum(l_clip_score["ref_score"][4:]) / len(l_clip_score["ref_score"][4:])}')
    print('='*40)