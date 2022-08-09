import sys
import os

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import qdarkstyle

import numpy as np


from probe_calibration_wizard_ui import Ui_MainWindow

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalsSlots()

    def connectSignalsSlots(self):
        self.setupModeControl()
        self.setupSliderGroup(self.lineEdit_dz,
                              self.lineEdit_dzMin,
                              self.horizontalSlider_dz,
                              self.lineEdit_dzMax,
                              self.comboBox_dzUnits,
                              self.pushButton_dzReset)
        self.setupSliderGroup(self.lineEdit_e,
                              self.lineEdit_eMin,
                              self.horizontalSlider_e,
                              self.lineEdit_eMax,
                              self.comboBox_eUnits,
                              self.pushButton_eReset)
        self.setupFileManagement()

        
        self.disableControls()

    def disableControls(self):
        self.groupBox_probeAdjustment.setDisabled(True)
        self.groupBox_probeViewer.setDisabled(True)
        self.groupBox_calibrationHints.setDisabled(True)
        self.pushButton_saveAs.setDisabled(True)
        self.pushButton_save.setDisabled(True)

    def enableControls(self):
        self.groupBox_probeAdjustment.setDisabled(False)
        self.groupBox_probeViewer.setDisabled(False)
        self.groupBox_calibrationHints.setDisabled(False)
        self.pushButton_saveAs.setDisabled(False)
        self.pushButton_save.setDisabled(False)
    
    def setupModeControl(self):
        slider = self.horizontalSlider_nmodes
        spinbox = self.spinBox_nmodes
        label = self.label_nmodeReadout
        currentMode = self.spinBox_viewMode

        slider.valueChanged.connect(spinbox.setValue)

        def nmodes_spinbox_finished_callback():
            val = spinbox.value()
            slider.setValue(val)
            label.setText('/%d' % val)

            if currentMode.value() > val:
                currentMode.setValue(val)
            
        spinbox.valueChanged.connect(nmodes_spinbox_finished_callback)

        
    def setupSliderGroup(self, textbox, minbox, slider, maxbox, unit, reset):
        textbox.setValidator(QDoubleValidator())
        minbox.setValidator(QIntValidator())
        maxbox.setValidator(QIntValidator())
        
        # This causes problems because the value is changed when the text
        # box is updated too.
        #slider.valueChanged.connect(update_textbox)
        slider.sliderMoved.connect(lambda val : textbox.setText(str(val)))
        
        def update_slider_range():
            value = slider.value()
            minval = int(minbox.text())
            maxval = int(maxbox.text())
            if maxval < minval:
                minval, maxval = maxval, minval
                minbox.setText(str(minval))
                maxbox.setText(str(maxval))
                
                   
            slider.setRange(minval, maxval)
            if minval > value:
                slider.setValue(minval)
            elif maxval < value:
                slider.setValue(maxval)
                
            if slider.value() != value:
                textbox.setText(str(slider.value()))

        def textbox_finished_callback():
            value = float(textbox.text())
            if value < float(minbox.text()):
                minbox.setText(str(int(np.floor(value))))
            if value > float(maxbox.text()):
                slider.setValue(int(textbox.text()))
                maxbox.setText(str(int(np.ceil(value))))
                
            update_slider_range()
            slider.setValue(int(np.round(value)))
            
        textbox.editingFinished.connect(textbox_finished_callback)
        minbox.editingFinished.connect(update_slider_range)
        maxbox.editingFinished.connect(update_slider_range)

        def reset_slider():
            textbox.setText('0')
            textbox_finished_callback()
            
        reset.clicked.connect(reset_slider)

    def setupFileManagement(self):

        def getSaveLocation():
            saveloc = QFileDialog.getSaveFileName(
                caption='Save Location',
                directory=self.lineEdit_saveLocation.text(),
                filter='Probe Calibration (*.mat)')
            if saveloc[1] != '':
                self.lineEdit_saveLocation.setText(saveloc[0])
        

        default_saveloc = os.path.join(os.getcwd(), 'my_probe.mat')
        self.lineEdit_saveLocation.setText(default_saveloc)
        self.pushButton_chooseSaveLocation.clicked.connect(getSaveLocation)

        self.pushButton_load.clicked.connect(self.loadFile)

        def saveAs():
            saveloc = QFileDialog.getSaveFileName(
                caption='Save Location',
                filter='Probe Calibration (*.mat)')
            if saveloc[1] != '':
                self.saveFile(saveloc[0])
                
        self.pushButton_saveAs.clicked.connect(saveAs)

        self.pushButton_save.clicked.connect(lambda x: self.saveFile())
        
    def loadFile(self):
        to_load = QFileDialog.getOpenFileName(
            caption='Load a Probe Calibration',
            filter='Probe Calibration (*.mat);; CXI ptycho result (*.cxi)')

        # If they cancelled the load
        if to_load[1] == '':
            return

        # TODO
        print('Loading files is not yet implemented')
        self.enableControls()

    def saveFile(self, filename=None):
        if filename is None:
            filename = self.lineEdit_saveLocation.text()

        # Not sure if this is the most sane approach
        if filename[-4:].lower() != '.mat':
            filename += '.mat'

        # TODO
        print('Saving files is not yet implemented')

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName('PCW7012')
    app.setStyleSheet(qdarkstyle.load_stylesheet())
        
    win = Window()
    win.show()
    sys.exit(app.exec())
