import datetime
import logging
import math
import time
import torch
from os import path as osp
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (get_env_info, get_root_logger, get_time_str)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
from torch.quantization import quantize_dynamic, prepare, convert, default_observer, default_weight_observer
import torch.nn.utils.prune as prune
import torch.nn as nn
import os

def infer_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    torch.cuda.set_device(0)
    sample_input = torch.randn(1, 3, 224, 224)
    # copy the yml file to the experiment root
    # copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    model = build_model(opt)
    model.unet.qconfig = torch.quantization.QConfig(
        activation=default_observer.with_args(dtype=torch.quint8),
        weight=default_weight_observer.with_args(dtype=torch.qint8)
    )
    torch.quantization.prepare(model.unet, inplace=True)
    quantized_model = torch.quantization.convert(model.unet, inplace=True)
    for name, module in quantized_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Upsample)):
            print (module)
            prune.l1_unstructured(module, name='weight', amount=0.05)
                    # prune.remove(module, 'weight')  # Remove the pruning reparameterization


    torch.save(quantized_model, 'quantize_model.pth')

    state_dict = quantized_model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    torch.save(filtered_state_dict, 'filtered_model.pth')
    state_dict = torch.load('filtered_model.pth', map_location=torch.device('cpu'))
    quantized_model.load_state_dict(state_dict, strict=False)
    quantized_model.eval()  # Set the model to evaluation mode

    # torch.onnx.export(quantized_model,               # model being run
    #               sample_input,                         # model input (or a tuple for multiple inputs)
    #               "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
    #               export_params=True,        # store the trained parameter weights inside the model file
    #               opset_version=10,          # the ONNX version to export the model to
    #               do_constant_folding=True,  # whether to execute constant folding for optimization
    #               input_names = ['input'],   # the model's input names
    #               output_names = ['output'], # the model's output names
    #               dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #                             'output' : {0 : 'batch_size'}})
