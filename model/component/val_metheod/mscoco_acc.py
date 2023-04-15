import torch
from torch.distributed.nn import all_gather
from torchmetrics.functional import accuracy

from .base_val_method import BascValMetric


def norm_and_logits(img_encode, text_encode):
    img_encode = img_encode / img_encode.norm(dim=1, keepdim=True)
    text_encode = text_encode / text_encode.norm(dim=1, keepdim=True)
    logits = img_encode @ text_encode.t()
    return logits, logits.T


class MscocoValAccuracy(BascValMetric):
    def __init__(self):
        self.k_list = [i for i in [1, 3, 5, 10, 20, 50]]
        super().__init__()

    def validation_step(self, batch, model):
        image, text = batch
        outs = model(text, image)
        i2t_logits, t2i_logits = norm_and_logits(outs.visual_output.last_representation,
                                                 outs.text_output.last_representation)
        self.model_step_output = outs
        self.record(logits=i2t_logits, name='i2t', stage='val_step', recorder=self.res_step_dict)
        self.record(logits=i2t_logits, name='t2i', stage='val_step', recorder=self.res_step_dict)
        self.validation_step_outputs.append({
            'image_outs': torch.stack(all_gather(outs.visual_output.last_representation), dim=0),
            'text_outs': torch.stack(all_gather(outs.text_output.last_representation), dim=0),
        })
        return self.res_step_dict

    def validation_end(self):
        image_outs = []
        text_outs = []

        for batch in self.validation_step_outputs:
            image_out, text_out = batch['image_outs'], batch['text_outs']
            embedding = image_out.shape[-1]
            image_outs.append(image_out.reshape(-1, embedding))
            text_outs.append(text_out.reshape(-1, embedding))

        image_outs = torch.cat(image_outs, dim=0).float()
        text_outs = torch.cat(text_outs, dim=0).float()

        i2t_logits, t2i_logits = norm_and_logits(image_outs, text_outs)
        self.record(logits=i2t_logits, name='i2t', stage='val', recorder=self.res_end_dict)
        self.record(logits=t2i_logits, name='t2i', stage='val', recorder=self.res_end_dict)
        return self.res_end_dict

    def reset(self):
        self.res_end_dict.clear()
        self.res_step_dict.clear()
        self.validation_step_outputs.clear()

    def record(self, logits, name, stage, recorder):
        acc = self.cal_acc(logits)

        for k, v in acc.items():
            recorder[f"{self.model_name}-{name}-acc {k}"] = {
                'section': f'{stage}',
                'prefix': f'{self.model_name}-{name}-acc_{k}',
                'value': v
            }

    def cal_acc(self, logits):
        label = torch.arange(logits.shape[0], device=logits.device)
        acc_res = {}
        for k in self.k_list:
            if k >= logits.shape[0]:
                continue
            acc = accuracy(logits, label, top_k=k, task='multiclass', num_classes=logits.shape[0])
            acc_res[f"top-{k}"] = acc
        return acc_res
