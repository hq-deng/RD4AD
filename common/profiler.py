import os
import pprint

import psutil
import torch.cuda

class Profiler(object):
    # Each is a tuple of (value, program point)
    max_allocated_gpu_memory = (0, '')
    max_cpu_utilization = (0, '')
    max_used_ram_percent = (0, '')
    max_used_ram_bytes = (0, '')
    start_used_ram_percent = psutil.virtual_memory().percent

    @classmethod
    def measure(cls, program_point: str, device=None, logger=None):
        max_allocated_gpu_memory = torch.cuda.max_memory_allocated(device=device)
        cpu_utilization = psutil.cpu_percent()
        used_ram_percent = psutil.virtual_memory().percent

        pid = os.getpid()
        python_process = psutil.Process(pid)
        used_ram_bytes = python_process.memory_info().vms

        if max_allocated_gpu_memory > cls.max_allocated_gpu_memory[0]:
            cls.max_allocated_gpu_memory = (max_allocated_gpu_memory, program_point)
        if cpu_utilization > cls.max_cpu_utilization[0]:
            cls.max_cpu_utilization = (cpu_utilization, program_point)
        if used_ram_percent > cls.max_used_ram_percent[0]:
            cls.max_used_ram_percent = (used_ram_percent, program_point)
        if used_ram_bytes > cls.max_used_ram_bytes[0]:
            cls.max_used_ram_bytes = (used_ram_bytes, program_point)
        if logger:
            logger.info("Logging current information on GPU, CPU, and RAM {}".format(program_point))
            profile_info = {
                "Current allocated GPU memory": torch.cuda.memory_allocated(device=device),
                "Maximum allocated GPU memory (likely since start of program)": cls.max_allocated_gpu_memory,
                "CPU utilization since last call - may be 0 on first call": cpu_utilization,
                # TODO(@nraghuraman-matroid): Percent is for all processes, whereas bytes is just
                # for this process. I should make these consistent.
                "Percentage of used RAM (virtual memory)": used_ram_percent,
                "Bytes of used RAM (virtual memory)": used_ram_bytes,
            }
            logger.info(pprint.pformat(profile_info))

    @classmethod
    def restart_profilers(cls, logger):
        """
        Ensures that the GPU maximum memory allocation and the cpu usage are measured
        starting now.

        Does NOT clear existing maximum GPU memory and CPU usage values.
        """
        logger.info("Resetting GPU and CPU profilers")
        torch.cuda.reset_peak_memory_stats()
        # This function measures the cpu usage since last call, so by calling it here
        # and discarding the return value, we ensure that the next call measures cpu
        # usage starting now.
        psutil.cpu_percent() 

    @classmethod
    def log_stats(cls, logger):
        logger.info("Logging maximum information on GPU, CPU, and RAM, as well as program points when possible")
        profile_info = {
            "Maximum allocated GPU memory and program point": cls.max_allocated_gpu_memory,
            "Max CPU usage and program point": cls.max_cpu_utilization,
            "Max percentage of used RAM and program point" : cls.max_used_ram_percent,
            "Max # bytes of used RAM and program point" : cls.max_used_ram_bytes,
            "Difference between max used RAM percent and start used RAM percent": (
                cls.max_used_ram_percent[0] - cls.start_used_ram_percent
            ),
        }
        logger.info(pprint.pformat(profile_info))

