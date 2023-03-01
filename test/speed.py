import torch
from model._metrics import cal_speed
from test.utils import get_model, total_ex, get_args


def main(args, *inputs):
    device = args.device
    image_path = args.image_path
    text_path = args.text_path
    clip_path = args.clip_path
    load_teacher = args.load_teacher
    with torch.autocast('cuda' if 'cuda' in args.device else 'cpu'):
        model = get_model(device, load_teacher, clip_path, image_path, text_path, use_fp16=args.fp16)
        cal_speed(model, inputs)


if __name__ == '__main__':
    args = get_args()
    image_fake_input = torch.rand(32, 3, 224, 224).to('cuda')
    text_fake_input = torch.randint(size=(32, 77), low=0, high=4000).to('cuda')
    if args.fp16:
        image_fake_input = image_fake_input.to(torch.float16)
    total_ex(args, main, text_fake_input, image_fake_input)
