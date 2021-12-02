import os
import torch

from runpy import run_path
from thop import profile
from thop import clever_format

from ptflops import get_model_complexity_info


# model_names = ['MIMO-UNet', 'MIMO-UNetPlus']

# using https://github.com/Lyken17/pytorch-OpCounter
# for model_name in model_names:
#     model_mod = build_net(model_name)
#     input = torch.randn(1, 3, 256, 256)
#     macs, params = profile(model_mod, inputs=(input,))
#     macs, params = clever_format([macs, params], "%.3f")

#     print(f'{model_name}:')
#     print(macs, params)


# for model_name in model_names:
task = 'Debanding'
model_name = 'UNet'
load_file = run_path("UNet.py")
model = load_file['model']('UNet-96')
macs, params = get_model_complexity_info(
    model, (3, 256, 256), as_strings=True,
    print_per_layer_stat=True, verbose=True)
print(f"{model_name}")
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
