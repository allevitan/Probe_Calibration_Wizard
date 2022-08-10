import sys
import os
from copy import copy

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
#import qdarkstyle

import zmq

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
import datetime

import numpy as np
import torch as t
from scipy.io import loadmat, savemat

from cdtools.tools import propagators, analysis

from probe_calibration_wizard_ui import Ui_MainWindow

"""
This section deals with unit conversions
"""

SI_CONVERSIONS = {'A': 1e-10,
                  'nm': 1e-9,
                  'um': 1e-6,
                  'mm': 1e-3,
                  'cm': 1e-2,
                  'm': 1,
                  'km': 1e3,
                  'ueV': 1.602177e-25,
                  'meV': 1.602177e-22,
                  'eV': 1.602177e-19,
                  'keV': 1.602177e-16,
                  'MeV': 1.602177e-13} 

def convert_to_SI(value, unit):
    return value * SI_CONVERSIONS[unit.strip()]

def convert_from_SI(SI_value, unit):
    return SI_value / SI_CONVERSIONS[unit.strip()]

def convert_to_best_unit(SI_value, unit_options, target=500):
    converted = [convert_from_SI(SI_value, unit) for unit in unit_options]
    logs = np.log10(np.array(converted))
    target_log = np.log10(target)

    best = np.argmin(np.abs(logs-target_log))
    return converted[best], unit_options[best], best

def autoset_from_SI(SI_value, textbox, combobox, format_string='%0.2f'):
    units = [combobox.itemText(idx) for idx in range(combobox.count())]
    converted, best_unit, idx = convert_to_best_unit(SI_value, units)
    
    textbox.setText(format_string % converted)
    combobox.setCurrentIndex(idx)
    return converted

def get_SI_from_lineEdit(textbox, combobox):
    value = float(textbox.text())
    unit = combobox.currentText()
    return convert_to_SI(value, unit)


"""
Below we define some modifications of the near field propagation code that
are tuned for SPEED
"""

def make_universal_propagator(shape, spacing, wavelength):
    """This returns a map from k-space to the phase per propagation distance.
    This is a really big speedup, because the calculation of the base
    propagator isn't done in a super efficient manner, so having this stored
    is a big help.
    """
    # We need to choose a value of Z that is small enough that all the phases
    # will be less than 2pi, but as large as possible otherwise

    min_pitch = t.min(t.as_tensor(spacing))
    DOF = 2 * min_pitch**2 / wavelength

    z = DOF/2
    prop = propagators.generate_angular_spectrum_propagator(shape, spacing, wavelength, z, remove_z_phase=True)
    universal_prop = t.angle(prop) / z
    return universal_prop
        
"""
Below we define the functionality of the main window
"""

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalsSlots()

    def connectSignalsSlots(self):
        self.setupModeControl()
        self.setupProbeAdjustment()
        self.setupProbeViewer()
        self.setupFileManagement()
        self.setupCalibrationHints()

        self.setupZMQ()
        
        self.disableControls()

    def disableControls(self):
        self.groupBox_probeAdjustment.setDisabled(True)
        self.groupBox_probeViewer.setDisabled(True)
        self.groupBox_calibrationHints.setDisabled(True)
        self.pushButton_saveAs.setDisabled(True)
        self.pushButton_save.setDisabled(True)
        self.checkBox_zmq.setDisabled(True)
        
    def enableControls(self):
        self.groupBox_probeAdjustment.setDisabled(False)
        self.groupBox_probeViewer.setDisabled(False)
        self.groupBox_calibrationHints.setDisabled(False)
        self.pushButton_saveAs.setDisabled(False)
        self.pushButton_save.setDisabled(False)
        self.checkBox_zmq.setDisabled(False)
    
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
            currentMode.setMaximum(val)
            
            if currentMode.value() > val:
                currentMode.setValue(val)
            
        spinbox.valueChanged.connect(nmodes_spinbox_finished_callback)

    def setupProbeAdjustment(self):
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

        def updateOrtho():
            if hasattr(self, 'orthogonalized_probes'):
                delattr(self, 'orthogonalized_probes')
            self.fullRefresh()
                
        self.checkBox_ortho.stateChanged.connect(updateOrtho)


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

        reset.resetTo = '0'
        def reset_slider():
            textbox.setText(reset.resetTo)
            textbox_finished_callback()
            
        reset.clicked.connect(reset_slider)

        textbox.editingFinished.connect(self.fullRefresh)
        slider.sliderMoved.connect(self.fullRefresh)
        reset.clicked.connect(self.fullRefresh)


    def setupProbeViewer(self):
        # Making the figsize large just makes it so that the figure expands
        # when you increase the window size, instead of the other UI elements
        self.fig = Figure(figsize=(10,10))
        self.fig.set_facecolor((1.0,1.0,1.0,0.0)) # Transparent
                
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;") 
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.groupBox_probeViewer.layout().insertWidget(0,self.canvas)
        self.groupBox_probeViewer.layout().insertWidget(0,self.toolbar)
        self.groupBox_probeViewer.layout().removeWidget(self.graphicsView)
        self.graphicsView.deleteLater()
        self.graphicsView = None

        self.radioButton_real.clicked.connect(self.updateFigure)
        self.radioButton_fourier.clicked.connect(self.updateFigure)
        self.radioButton_amplitude.clicked.connect(self.updateFigure)
        self.radioButton_phase.clicked.connect(self.updateFigure)
        self.spinBox_viewMode.valueChanged.connect(self.updateFigure)


    def setupZMQ(self):
        def whenChecked(state):
            if state:
                if not hasattr(self, 'context'):
                    # We wait to set up zmq until the box is checked for the
                    # first time
                    self.context = zmq.Context()
                    
                    self.port = "tcp://*:5557"
                    # Define the socket to publish the probe calibration on
                    self.pub = self.context.socket(zmq.PUB)
                    self.pub.bind(self.port)
                
                self.emitCalibration()
                
        self.checkBox_zmq.stateChanged.connect(whenChecked)
    
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


    def setupCalibrationHints(self):
        self.lineEdit_a0.setValidator(QDoubleValidator())

        # TODO: Make load and edit mask and background work
        
        
    def loadFile(self):
        to_load = QFileDialog.getOpenFileName(
            caption='Load a Probe Calibration',
            filter='Probe Calibration (*.mat);; CXI ptycho result (*.cxi)')

        # If they cancelled the load
        if to_load[1] == '':
            return

        # First we load the data and check that it's good.
        # Note that we also copy everything over to a separate dictionary.
        # This is done so that we can load from result files that might have
        # lots of extra info we don't need, and we don't need to keep all
        # that extra data in memory

        
        loaded_data = loadmat(to_load[0])
        self.base_data = {}
        if not 'probe' in loaded_data:
            print('Dataset has no probe info, not loading')
            return

        if len(loaded_data['probe'].shape) < 2:
            print('Probe has not enough dimensions, not loading')
            return

        self.base_data['probe'] = loaded_data['probe']
        
        # This always unravels all the modes so the probe is n_modesxNxM
        if len(self.base_data['probe'].shape) == 2:
            self.base_data['probe'] = np.expand_dims(self.base_data['probe'], 0)

        n_modes = np.prod(self.base_data['probe'].shape[:-2])
        self.base_data['probe'] = self.base_data['probe'].reshape(
            (n_modes,) + self.base_data['probe'].shape[-2:])

        if not 'wavelength' in loaded_data:
            print('Dataset has no wavelegth info, not loading')
            return

        # Since this is coming from a .mat file, the constant gets loaded
        # in as part of a 2D matrix
        self.base_data['wavelength'] = loaded_data['wavelength'].ravel()[0]
        
        if not 'basis' in loaded_data:
            print('Dataset has no info on the probe basis, not loading')
            return

        self.base_data['basis'] = loaded_data['basis']

        if 'A0' in loaded_data:
            self.base_data['A0'] = loaded_data['A0']
        elif 'a0' in loaded_data:
            self.base_data['A0'] = loaded_data['a0']
            
        if 'mask' in loaded_data:
            self.base_data['mask'] = loaded_data['mask']

        if 'background' in loaded_data:
            self.base_data['background'] = loaded_data['background']
        
        # This is initializing self.probe, which stores the processed probe
        self.probe = loaded_data['probe']
        
        # Now we update the controls to be initialized with resonable values
        self.horizontalSlider_nmodes.setRange(1, n_modes)
        self.spinBox_nmodes.setRange(1, n_modes)
        self.spinBox_nmodes.setValue(n_modes)

        if hasattr(self, 'orthogonalized_probes'):
            delattr(self, 'orthogonalized_probes')
        self.checkBox_ortho.setCheckState(False)
        
        # The energy slider
        hc = 1.986446e-25 # hc in Joule-meters
        energy = hc / self.base_data['wavelength']
        energy = autoset_from_SI(energy, self.lineEdit_e, self.comboBox_eUnits,
                                 format_string='%0.2f')
        minval = int(np.floor(energy*0.9))
        maxval = int(np.ceil(energy*1.1))
        self.lineEdit_eMin.setText(str(minval))
        self.lineEdit_eMax.setText(str(maxval))
        self.horizontalSlider_e.setRange(minval, maxval)
        self.horizontalSlider_e.setValue(int(np.round(energy)))

        # TODO: This will fail if the units box is changed
        self.pushButton_eReset.resetTo = self.lineEdit_e.text()
        
        # And the Propagation Slider
        pitches = np.linalg.norm(loaded_data['basis'],axis=0)
        min_pitch = np.min(pitches)
        DOF = 2 * min_pitch**2 / loaded_data['wavelength']
        units = [self.comboBox_dzUnits.itemText(idx) for idx in
                 range(self.comboBox_dzUnits.count())]
        
        DOF, unit, idx = convert_to_best_unit(DOF, units)
        prop_limit = int(np.ceil(200 * DOF))

        self.lineEdit_dz.setText('0')
        self.lineEdit_dzMin.setText(str(-prop_limit))
        self.lineEdit_dzMax.setText(str(prop_limit))
        self.horizontalSlider_dz.setRange(-prop_limit, prop_limit)
        self.horizontalSlider_dz.setValue(0)
        self.comboBox_dzUnits.setCurrentIndex(idx)
        
        if 'A0' in loaded_data:
            self.lineEdit_a0.setText(str(A0))

        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title('Amplitude of Mode 1 in Real Space')
        self.axes.set_xlabel('x (um)')
        self.axes.set_ylabel('y (um)')

        basis_norm = np.linalg.norm(self.base_data['basis'], axis=0)
        basis_norm /= 1e-6 # Hard-coded um for now
        extent = [0, self.base_data['probe'].shape[-1]*basis_norm[1], 0,
                  self.base_data['probe'].shape[-2]*basis_norm[0]]
        self.im = self.axes.imshow(np.abs(self.base_data['probe'][0]),
                                   extent=extent)
        divider = make_axes_locatable(self.axes)
        
        cax = divider.append_axes('right', size='5%', pad=0.05)

        self.fig.colorbar(self.im, cax=cax)

        self.updateFigure()
        self.enableControls()

    def fullRefresh(self):
        self.recalculate()
        self.updateFigure()
        if self.checkBox_zmq.checkState():
            self.emitCalibration()
        
    def recalculate(self):
        # Order of operations
        # 1: Orthogonalize and clip probes
        # 2: Update the probe energy
        # 3: Propagate correct distance

        # TODO: Update this so that the probe is saved in Fourier and real
        # space, both before and after propagation, to speed up the calculation
        # and display

        # TODO: Update this to allow for calculations on the GPU, if available,
        # to speed up the processing.

        # This is a pretty expensive operation, and it will rarely change,
        # so I think it's worth it to avoid recalculating it each time.
        # Whenever anything happens that needs to recalculate this, the
        # pattern is to just delete it and it will be recalculated when needed
        if not hasattr(self, 'orthogonalized_probes'):
            if self.checkBox_ortho.checkState():
                self.orthogonalized_probes = \
                    analysis.orthogonalize_probes(
                        t.as_tensor(self.base_data['probe']))
            else:
                self.orthogonalized_probes = t.as_tensor(self.base_data['probe'])

        nmodes = self.spinBox_nmodes.value()
        clipped_probes = self.orthogonalized_probes[:nmodes]

        # TODO: Actually implement the energy update
        
        if not hasattr(self, 'universal_prop'):
            # Whenever something happens that has to change this, for example
            # when the energy changes, then the pattern is to delete the
            # universal propagator and it will be recalculated here.
            
            shape = clipped_probes.shape[-2:]
            spacing = t.as_tensor(np.linalg.norm(self.base_data['basis'], axis=0))
            wavelength = self.base_data['wavelength'].ravel()[0]
            self.universal_prop = make_universal_propagator(shape, spacing, wavelength)
            self.universal_prop = (1j * self.universal_prop)#.to(device='cuda:0')
        z = get_SI_from_lineEdit(self.lineEdit_dz, self.comboBox_dzUnits)

        if z != 0:
            prop = t.exp(z * self.universal_prop)
            self.probe = propagators.near_field(clipped_probes, prop).numpy()
        else:
            self.probe = clipped_probes
            
    def updateFigure(self):
        # TODO: Make it rescale the colorbar on update
        # TODO: Make the axes change to frequency for Fourier space
        # TODO: Make the colormap change between real and Fourier space
        # TODO: Make the units of the axes something sensible depending on
        # the extent
        
        probe_mode = self.spinBox_viewMode.value()
        to_show = self.probe[probe_mode-1]

        if self.radioButton_fourier.isChecked():
            to_show = np.fft.fftshift(np.fft.fft2(to_show, norm='ortho'))

        if self.radioButton_amplitude.isChecked():
            to_show = np.abs(to_show)
        else:
            to_show = np.angle(to_show)
        
        self.im.set_data(to_show)
        self.canvas.draw_idle()

    def emitCalibration(self):
        #TODO: update the energy here
        save_data = copy(self.base_data)
        save_data['probe'] = self.probe
        self.pub.send_pyobj(save_data)
        currentTime = datetime.datetime.now().strftime('%H:%M:%S')
        self.statusBar().showMessage('Published to ØMQ ('
                                     + self.port
                                     + ') at ' + currentTime)
        
    def saveFile(self, filename=None):
        t0 = time()
        if filename is None:
            filename = self.lineEdit_saveLocation.text()

        # Not sure if this is the most sane approach
        if filename[-4:].lower() != '.mat':
            filename += '.mat'

        # TODO : actually update the energy here
        save_data = copy(self.base_data)
        save_data['probe'] = self.probe
        savemat(filename, save_data)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName('PCW7012')
    #app.setStyleSheet(qdarkstyle.load_stylesheet())

    win = Window()
    win.show()
    sys.exit(app.exec())
