import time
import argparse

import torch
import torchvision
import torchvision.models as models
from torchvision.models import list_models


def benchmark_forward_pass(model_name, n_runs, device):
    print('-' * 60)
    print('Model name:', model_name)

    IMAGE_SIZE = 224

    # Warmup
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    x = x.to(device)
    model = models.__dict__[model_name]()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print('Input size:', x.shape)
    print('Output size:', out.shape)

    with torch.no_grad():
        total_time = 0
        for i in range(n_runs):
            x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            x = x.to(device)

            t0 = time.time()
            out = model(x)
            total_time += time.time() - t0

    print('Total time: %.2f sec' % total_time)
    print('Average time: %.1f ms' % (1000.0 * total_time / n_runs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--n_runs', default=1000, type=int)
    parser.add_argument('--models', nargs='*', default=None,
                        help='Specific model families to benchmark (e.g., resnet vgg). If not provided, benchmarks all models.')
    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("torch.__version__: ", torch.__version__)
    print("torchvision.__version__: ", torchvision.__version__)
    print('GPU:', torch.cuda.get_device_properties(device).name)

    print('Number of runs:', args.n_runs)

    all_model_names = list_models(module=torchvision.models)

    if args.models:
        model_names = []
        for family in args.models:
            # Filter models that start with the family name
            family_models = [name for name in all_model_names if name.startswith(family)]
            model_names.extend(family_models)
        print(f'Model families to test: {args.models}')
        print(f'Number of models to test: {len(model_names)}')
        if not model_names:
            print(f'No models found for families: {args.models}')
            print(f'Available models (sample): {all_model_names}')  # Show sample
            exit(1)
    else:
        model_names = all_model_names
        print('Number of models to test:', len(model_names))

    for model_name in model_names:
        benchmark_forward_pass(model_name, args.n_runs, device)
        # After each model benchmark
        if device.type == 'cuda':
            torch.cuda.empty_cache()