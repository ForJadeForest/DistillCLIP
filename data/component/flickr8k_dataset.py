from torch.utils.data import Dataset
from test.utils.ex_script.flickr8k_ex import load_data


class Flickr8kDataset(Dataset):
    def __init__(self, input_json_path, image_directory, is_train=False):
        assert not is_train, f"the Flickr8k only use to validation! please set is_train as False"
        images, refs, candidates, human_scores = load_data(input_json_path, image_directory)
        self.images = images
        self.refs = refs
        self.candidates = candidates
        self.human_scores = human_scores

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.refs[item], self.candidates[item], self.human_scores[item]
