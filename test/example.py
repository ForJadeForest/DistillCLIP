import json
from test.test_model.n_gram_model import NGramModel

with open('/data/pyz/data/mscoco/annotations/captions_val2014.json', 'r') as f:
    ref_data = json.load(f)['annotations']
id2ref = {}
for d in ref_data:
    if d['image_id'] in id2ref:
        id2ref[d['image_id']].append(d['caption'])
    else:
        id2ref[d['image_id']] = [d['caption']]


cider = NGramModel('cider')

ref = list(id2ref.values())
can = [i[0] for i in ref]
can[0] = 'A clock is used instead of the front wheel in a bicycle ornament.'


print(cider(None, can, ref))
