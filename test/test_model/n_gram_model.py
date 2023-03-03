from .base_model import BaseModel
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


class NGramModel(BaseModel):
    def __init__(self, model_name, use_PTB=False):
        super(NGramModel, self).__init__(model_name)
        self.scorer = name2scorer(model_name)
        self.use_PTB = use_PTB

    def cal_score(self, images=None, candidates=None, references=None, reduction=False):
        overall, per_cap = pycoco_eval(self.scorer, references, candidates, self.use_PTB)
        if reduction:
            return {f'mean_{self.model_name}': overall}

        return {f'{self.model_name}': per_cap}


def name2scorer(model_name):
    scorers_dict = {
        'bleu': Bleu(4),
        'meteor': Meteor(),
        'rouge': Rouge(),
        'cider': Cider(),
        'spice': Spice()
    }
    if model_name not in scorers_dict:
        raise ValueError(f'The model_name should in {scorers_dict.keys()}, but got {model_name}')
    return scorers_dict[model_name]


def tokenize(refs, cands, no_op):
    # no_op is a debug option to see how significantly not using the PTB tokenizer
    # affects things
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}

    else:
        refs = {idx: [{'caption': r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption': c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


def pycoco_eval(scorer, refs, cands, use_PTB=False):
    """
    scorer is assumed to have a compute_score function.
    refs is a list of lists of strings
    cands is a list of predictions
    """
    refs, cands = tokenize(refs, cands, use_PTB)
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores
