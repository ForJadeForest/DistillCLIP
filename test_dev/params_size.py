import torch

from test.utils import get_model, get_args, total_ex


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def cal_parameters_size_count(model, model_name):
    para_count = count_parameters(model)
    torch.save(model, './result/model_cpk/' + model_name + '.pth')
    print(para_count)
    return {'para_count': {'model_para_count': para_count}}


def main(args):
    device = args.device
    image_path = args.image_path
    text_path = args.text_path
    clip_path = args.clip_path
    load_teacher = args.load_teacher
    model = get_model(device, load_teacher, clip_path, image_path, text_path)
    return cal_parameters_size_count(model, args.model_name)


if __name__ == '__main__':
    args = get_args()
    total_ex(args, main)
