"""
for some reason, sometimes parameters or buffers of one model may be not contiguous, thus it raises an error that model can't broad between different gpus,
the following function force all the parameters to be contiguous
"""


def ensure_contiguous(module):
    for name, param in module.named_parameters():
        if param.data.is_contiguous():
            continue
        param.data = param.data.contiguous()
        print('***** params *****',type(module).__name__)

    for name, buffer in module.named_buffers():
        if buffer.data.is_contiguous():
            continue
        buffer.data = buffer.data.contiguous()
        print('***** buffers *****',type(module).__name__)
        
ensure_contiguous(G)
ensure_contiguous(D)
ensure_contiguous(G_ema)
