import sys
from PyQt5 import *
from PyQt5.QtWidgets import *

class PlotDialog(QDialog):
    def __init__(self, agent):
        super().__init__()
        self.setWindowTitle("Plot Dialog")

        layout = QVBoxLayout()
        self.label = QLabel("Plot will be shown below:")
        layout.addWidget(self.label)

        # Create a Matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.agent = agent
        # Call your code to generate and update the plot in a separate thread
        plot_thread = threading.Thread(target=self.agent.redraw)
        plot_thread.start()