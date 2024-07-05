import os
import torch
import warnings


def setup_cuda(use_memory_fraction: float = 0.1, num_threads: int = 8, device: str = 'cuda',
               multiGPU: bool = False, visible_devices: str = "0", use_cuda_with_id: int = 0) -> None:
    if device == 'cpu':
        import multiprocessing
        print(f'Torch version: {torch.__version__}')
        print(f"Available CPU cores: {multiprocessing.cpu_count()}")
        torch.set_num_threads(num_threads)
        print(f"Torch using {num_threads} threads.")

    elif device == 'gpu' or device == "cuda":
        # setup environmental variables
        print(f'Torch version: {torch.__version__}')
        if int(torch.__version__.split("+")[0].split('.')[1]) >= 12:
            warnings.warn(
                'Since 1.12.0+ torch has lazy init and may ignore environmental variables :( - consider variable: use_cuda_with_id')

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        assert torch.cuda.is_available(
        ), f"cuda is not available, cannot setup cuda devices: {visible_devices}"

        # Setup num threads limit - e.g. for data loader
        torch.set_num_threads(num_threads)
        print(f"Torch using {num_threads}/{torch.get_num_threads()} threads.")

        # check available
        devices = [d for d in range(torch.cuda.device_count())]
        device_names = [torch.cuda.get_device_name(d) for d in devices]
        device_to_name = dict(zip(devices, device_names))
        print(f"Available devices: {device_to_name}")

        # setup cuda
        if len(devices) > 1 and multiGPU:
            for dev_id in devices:
                cu_dev = torch.device(device, dev_id)
                print(f"***Device - {cu_dev} setup:***")
                gpu_properties = torch.cuda.get_device_properties(dev_id)
                total_memory = gpu_properties.total_memory
                memory_info = torch.cuda.mem_get_info()
                print(
                    f"\tCuda available: {torch.cuda.is_available()}, device: {cu_dev} num. devices: {torch.cuda.device_count()}, device name: {gpu_properties.name}, free memory: {memory_info[0] / 1024 ** 2:.0f} MB, total memory: {total_memory / 1024 ** 2:.0f} MB.")
                torch.cuda.set_per_process_memory_fraction(use_memory_fraction, torch.cuda.current_device())
                print(
                    f"\tMemory fraction: {use_memory_fraction}, memory limit {int(total_memory * use_memory_fraction) / 1024 ** 3:.2f} GB.")
            print(f"Setup completed - devices in use: {', '.join([f'{device}:{id}' for id in devices])}.")
        else:
            cu_dev = torch.device(device, use_cuda_with_id)
            print(f"***Device - {cu_dev} setup:***")
            gpu_properties = torch.cuda.get_device_properties(cu_dev)
            total_memory = gpu_properties.total_memory
            memory_info = torch.cuda.mem_get_info()
            print(
                f"\tCuda available: {torch.cuda.is_available()}, device: {cu_dev} num. devices: {torch.cuda.device_count()}, device name: {gpu_properties.name}, free memory: {memory_info[0] / 1024 ** 2:.0f} MB, total memory: {total_memory / 1024 ** 2:.0f} MB.")
            torch.cuda.set_per_process_memory_fraction(use_memory_fraction, cu_dev)
            print(
                f"\tMemory fraction: {use_memory_fraction}, memory limit {int(total_memory * use_memory_fraction) / 1024 ** 3:.2f} GB.")
            current_device_index = torch.cuda.current_device()
            print(f"Setup completed - device in use: {torch.device(device, current_device_index)}.")
        print('\n')