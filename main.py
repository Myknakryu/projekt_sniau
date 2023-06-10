# Tensorflow annoyance with AMDGPU since ROCm support is unstable as HELL
import logging, os
from matplotlib import pyplot as plt
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from SimulationAgent import SimulationAgent

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    simulation = SimulationAgent()
    simulation.run_parallelized(12)
    plt.ioff()
    plt.show()
