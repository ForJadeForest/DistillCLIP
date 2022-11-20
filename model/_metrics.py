# 如果需要自定义metrics，在此处定义
import numpy as np
import torch
from thop import profile


def cal_flop(model, inputs):
    # input_shape of model,batch_size=1
    flops, params = profile(model, inputs=(inputs, ))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
    return flops, params


def cal_speed(model, inputs):
    dummy_input = inputs
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    return mean_syn, std_syn, mean_fps

