import argparse
import torch
from ptflops import get_model_complexity_info

from parse_config import ConfigParser
import model.model as module_arch


def _register_hooks(model, inputs):
    shapes = []

    def hook(module, inp, out):
        try:
            in_shape = tuple(inp[0].size()) if isinstance(inp, (list, tuple)) and hasattr(inp[0], 'size') else getattr(inp, 'size', lambda: None)()
        except Exception:
            in_shape = None
        try:
            out_shape = tuple(out.size()) if hasattr(out, 'size') else None
        except Exception:
            out_shape = None
        shapes.append((module.__class__.__name__, in_shape, out_shape))

    handles = []
    for m in model.modules():
        # skip container modules
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook))

    # run a forward pass
    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    # remove hooks
    for h in handles:
        h.remove()

    return shapes


def main():
    parser = argparse.ArgumentParser(description='Compute model FLOPs, params and layer shapes')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-i', '--input-shape', default='1,3000', type=str,
                        help='input shape as C,L or C,H,W (comma separated)')
    parser.add_argument('--use-cuda', action='store_true', help='use GPU for analysis')
    # Pass the parser object to ConfigParser so it can call parse_args()
    config = ConfigParser.from_args(parser, 1, [])
    # now parse args locally to get input shape / cuda flag
    args = parser.parse_args()

    # load model from config
    model = config.init_obj('arch', module_arch)

    # prepare dummy input
    shape = tuple(int(x) for x in args.input_shape.split(','))
    # ptflops expects (C, H, W) like, and model likely expects (batch, C, L)
    dummy = torch.randn((1,)+shape)
    device = torch.device('cuda:0' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)
    dummy = dummy.to(device)

    # register hooks to capture shapes
    shapes = _register_hooks(model, dummy)

    print('\nLayer input/output shapes (module, in_shape, out_shape):')
    for name, in_s, out_s in shapes:
        print(f"{name}: in={in_s} -> out={out_s}")

    # compute MACs and params
    model.eval()
    if device.type == 'cuda':
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model, shape, as_strings=True, print_per_layer_stat=False, verbose=False)
    else:
        macs, params = get_model_complexity_info(model, shape, as_strings=True, print_per_layer_stat=False, verbose=False)

    print(f"\nComputational complexity (MACs): {macs}")
    print(f"Number of parameters: {params}")


if __name__ == '__main__':
    main()
