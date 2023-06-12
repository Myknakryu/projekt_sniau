# Tensorflow annoyance with AMDGPU since ROCm support is unstable as HELL
import logging, os, atexit
from matplotlib import pyplot as plt


logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from SimulationAgent import SimulationAgent
from utils import show_yes_no_dialog, get_input_string
from PyQt5 import *
from PyQt5.QtWidgets import *

def clean_up():
    app.quit()
    plt.close('all')

if __name__ == "__main__":
    app = QApplication([])
    atexit.register(clean_up)
    if show_yes_no_dialog("Czy chcesz trenować nowy model?"):
        tf.compat.v1.disable_eager_execution()
        simulation = SimulationAgent()
        simulation.run_parallelized(12)
        if show_yes_no_dialog("Czy chcesz zapisać model?"):
            base_name = get_input_string("Podaj nazwę: ")
            if base_name == None:
                base_name = "latest"
            actor_name = base_name + "_actor.h5"
            critic_name = base_name + "_critic.h5"
            simulation.save(actor_file=actor_name, critic_file=critic_name)
        plt.ioff()
        plt.show()
    else:
        base_name = get_input_string("Podaj nazwę wczytywanego modelu: ")
        if base_name == None:
            base_name = "latest"
        actor_name = base_name + "_actor.h5"
        critic_name = base_name + "_critic.h5"
        simulation = SimulationAgent()
        simulation.load(actor_file=actor_name, critic_file=critic_name)
        simulation.play()

