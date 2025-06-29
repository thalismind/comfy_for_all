import time
import threading
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetCount

IDLE_TIMEOUT = 15 * 60 # 15 minutes

class GPUIdleTimer:
  def __init__(self, gpu_index=0, idle_threshold=IDLE_TIMEOUT, start_time=None, last_time=None):
    self.loaded = False
    self.gpu_index = gpu_index
    self.idle_threshold = idle_threshold  # percent
    self.idle_time = 0
    self.start_time = start_time if start_time is not None else time.time()
    self.last_time = last_time if last_time is not None else time.time()
    self._stop_event = threading.Event()
    self._thread = None

  def load_nvml(self):
    try:
      nvmlInit()
      self.loaded = True
      print(f"Number of GPUs: {nvmlDeviceGetCount()}")
    except Exception as e:
      print(f"Failed to initialize NVML: {e}")

  def unload_nvml(self):
    if not self.loaded:
      return

    nvmlShutdown()

  def reset(self, start_time=None, last_time=None):
    self.idle_time = 0
    self.start_time = start_time if start_time is not None else time.time()
    self.last_time = last_time if last_time is not None else time.time()

  def increment_timer(self, now=None):
    now = now if now is not None else time.time()
    elapsed = now - self.last_time
    self.last_time = now
    if self.is_gpu_idle():
      self.idle_time += elapsed
    else:
      self.idle_time = 0

  def has_reached_idle_threshold(self):
    return self.idle_time >= self.idle_threshold

  def is_gpu_idle(self):
    if not self.loaded:
      print("NVML not loaded, cannot check GPU utilization.")
      return True

    handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
    util = nvmlDeviceGetUtilizationRates(handle)
    return util.gpu <= self.idle_threshold

# Example usage:
if __name__ == "__main__":
    gpu_timer = GPUIdleTimer(gpu_index=0)
    gpu_timer.load_nvml()

    try:
        while True:
            gpu_timer.increment_timer()
            if gpu_timer.has_reached_idle_threshold():
                print(f"GPU {gpu_timer.gpu_index} has been idle for {gpu_timer.idle_time:0.2f} seconds.")
                # Perform any action needed when the GPU is idle
                gpu_timer.reset()  # Reset the timer after action
            time.sleep(1)  # Check every second
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        gpu_timer.unload_nvml()