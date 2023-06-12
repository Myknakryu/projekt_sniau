import sys
from PyQt5 import *
from PyQt5.QtWidgets import *

def show_yes_no_dialog(message):
    msg_box = QMessageBox()
    msg_box.setWindowTitle(message)
    msg_box.setText(message)
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.Yes)
    result = msg_box.exec_()
    return result == QMessageBox.Yes

def get_input_string(prompt):
    input_dialog = QInputDialog()
    input_dialog.setWindowTitle(prompt)
    input_dialog.setLabelText(prompt)
    input_dialog.setInputMode(QInputDialog.TextInput)

    result = input_dialog.exec_()
    
    if result == QInputDialog.Accepted:
        text = input_dialog.textValue()
        return text
    else:
        return None