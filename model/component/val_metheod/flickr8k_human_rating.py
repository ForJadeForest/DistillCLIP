import torch
from scipy import stats
from torch.distributed.nn import all_gather

from .base_val_method import BascValMetric


class Flickr8kHumanRating(BascValMetric):
    def __init__(self):
        super().__init__()

    def validation_step(self, batch, model):
        images, references, candidates, human_score = batch
        candidates_features = model.encode_text(candidates).last_representation
        images_features = model.encode_image(images).last_representation
        # candidates: batch, ref_num,
        references = references.reshape(-1, references.shape[-1])
        refs_features = model.encode_text(references).last_representation

        candidates_features = candidates_features / candidates_features.norm(dim=1, keepdim=True)
        images_features = images_features / images_features.norm(dim=1, keepdim=True)
        refs_features = refs_features / refs_features.norm(dim=1, keepdim=True)

        clip_score = 2.5 * torch.clip(torch.sum(images_features * candidates_features, dim=1), 0, None)

        ref_candidate_score = torch.empty_like(clip_score, device=clip_score.device)
        for i, candidates_feature in enumerate(candidates_features):
            ref_feature = refs_features[i * 5: (i + 1) * 5, :]
            ref_candidate_score[i] = torch.max(candidates_feature @ ref_feature.T)

        ref_clip_scores = 2 * clip_score * ref_candidate_score / (clip_score + ref_candidate_score)
        self.validation_step_outputs.append({
            'clip_score': torch.stack(all_gather(clip_score)),
            'ref_clip_score': torch.stack(all_gather(ref_clip_scores)),
            'human_rating': torch.stack(all_gather(human_score)),
        })
        return self.res_step_dict

    def validation_end(self):
        clip_score_list = []
        ref_clip_score_list = []
        human_rating_list = []
        for batch in self.validation_step_outputs:
            clip_score = batch['clip_score']
            ref_clip_score = batch['ref_clip_score']
            human_rating = batch['human_rating']
            clip_score_list.append(clip_score.reshape(-1))
            ref_clip_score_list.append(ref_clip_score.reshape(-1))
            human_rating_list.append(human_rating.reshape(-1))

        clip_scores = torch.cat(clip_score_list, dim=0).float().detach().cpu().numpy()
        ref_clip_scores = torch.cat(ref_clip_score_list, dim=0).float().detach().cpu().numpy()
        human_ratings = torch.cat(human_rating_list, dim=0).float().detach().cpu().numpy()

        tau = 100 * stats.kendalltau(clip_scores, human_ratings, variant='c')[0]

        self.res_end_dict[f"{self.model_name}-clip_score-human-rating"] = {
            'section': f'val',
            'prefix': f'{self.model_name}-clip_score-human_rating-tau_c',
            'value': tau
        }
        tau = 100 * stats.kendalltau(ref_clip_scores, human_ratings, variant='c')[0]
        self.res_end_dict[f"{self.model_name}-ref_clip_score-human-rating"] = {
            'section': f'val',
            'prefix': f'{self.model_name}-ref_clip_score-human_rating-tau_c',
            'value': tau
        }
        return self.res_end_dict

    def reset(self):
        self.res_end_dict.clear()
        self.res_step_dict.clear()
        self.validation_step_outputs.clear()
