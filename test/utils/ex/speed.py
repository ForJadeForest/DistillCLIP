import torch
from model._metrics import cal_speed
from test.utils import get_model, total_ex, get_args
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def main(args, *inputs):
    device = args.device
    print(f'Now is the {device}')
    image_path = args.image_path
    text_path = args.text_path
    clip_path = args.clip_path
    load_teacher = args.load_teacher
    model = get_model(device, load_teacher, clip_path, image_path, text_path, use_fp16=args.fp16)
    metric_name = ['mean_time', 'std_time', 'mean_fps']
    speed_dict = {}
    with torch.autocast('cuda' if 'cuda' in args.device else 'cpu'):

        inputs = [i.to(device) for i in inputs]
        res = cal_speed(model, inputs)
        # flops = FlopCountAnalysis(model, tuple(inputs))
        # print("FLOPs: ", flops.total())
        speed_dict['speed'] = {k: round(v, 2) for k, v in zip(metric_name, res)}

        visual_model = model.image_encoder
        image_inputs = (inputs[1], )
        res = cal_speed(visual_model, image_inputs)
        # flops = FlopCountAnalysis(visual_model, inputs)
        # print("FLOPs: ", flops.total())
        speed_dict['visual_speed'] = {k: round(v, 2) for k, v in zip(metric_name, res)}

        text_model = model.text_encoder
        text_inputs = (inputs[0], )
        res = cal_speed(text_model, text_inputs)
        # flops = FlopCountAnalysis(text_model, inputs)
        speed_dict['text_speed'] = {k: round(v, 2) for k, v in zip(metric_name, res)}
        print(f'res time: {res}')
        # print("FLOPs: ", flops.total())


    # speed_dict['speed']['FLOPs'] = flops.total()

    print(parameter_count_table(model))

    return speed_dict


if __name__ == '__main__':
    args = get_args()
    batch = 512
    image_fake_input = torch.rand(batch, 3, 224, 224)
    text_fake_input = torch.randint(size=(batch, 77), low=0, high=4000)
    if args.fp16:
        image_fake_input = image_fake_input.to(torch.float16)
    total_ex(args, main, text_fake_input, image_fake_input)
