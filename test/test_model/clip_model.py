from .base_model import BaseModel
from test.clip_score import get_all_clip_score, get_clip_score
from model.component.clip_model import CLIPModel


class TestCLIPModel(BaseModel):
    def __init__(self, model_name, model: CLIPModel, device, w=2.5):
        super(TestCLIPModel, self).__init__(model_name)
        self.model = model
        self.device = device
        self.w = w

    def cal_score(self, images=None, candidates=None, references=None, reduction=False):
        if references:
            average_score, all_scores = get_clip_score(self.model, images, candidates, self.device, self.w)
            if not reduction:
                return {'clip_score': all_scores}
            else:
                return {'mean_clip_score': average_score}

        else:
            average_score, all_scores, average_ref_score, all_ref_score = \
                get_all_clip_score(self.model, images, references, candidates, self.device, self.w)
            if not reduction:
                return {'clip_score': all_scores, 'ref_clip_score': all_ref_score}
            else:
                return {'mean_clip_score': average_score, 'mean_ref_clip_score': average_ref_score}
