import sys
import argparse
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
from matplotlib import pyplot as plt
plt.ion()

import datetime

import numpy as np
import torch as t
from scipy.io import loadmat, savemat
import h5py

from cdtools.tools import propagators, analysis

from probe_calibration_wizard.probe_calibration_wizard_ui import Ui_MainWindow
from probe_calibration_wizard.update_probe_energy import change_energy

# TODO: Add a check that the background and mask match the dimensions of the output probe when saving
# TODO: Implement Lars' method for energy changing, which will allow for energy shifts of non-square probes
# TODO: Get non square probes working
# TODO: Add rotate and flip features
# TODO: Add pad and crop features
# TODO: Add tilt features
# TODO: Allow for save from .cxi files a la ptychocam
# TODO: Check if the final_resolution value is actuall the correct thing for pitch


"""
This section deals with unit conversions
"""

hc = 1.986446e-25 # hc in Joule-meters

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
                  'MeV': 1.602177e-13,
                  'nm/eV': 1e-9/1.602177e-19,
                  'um/eV': 1e-6/1.602177e-19,
                  'mm/eV': 1e-3/1.602177e-19,
                  'cm/eV': 1e-2/1.602177e-19,
                  'm/eV': 1/1.602177e-19,
                  'km/eV': 1e3/1.602177e-19} 

def convert_to_SI(value, unit):
    return value * SI_CONVERSIONS[unit.strip()]

def convert_from_SI(SI_value, unit):
    return SI_value / SI_CONVERSIONS[unit.strip()]

def convert_to_best_unit(SI_value, unit_options, target=50):
    converted = [convert_from_SI(SI_value, unit) for unit in unit_options]
    logs = np.log10(np.array(converted))
    target_log = np.log10(target)

    best = np.argmin(np.abs(logs-target_log))
    return converted[best], unit_options[best], best

def autoset_from_SI(SI_value, textbox, combobox, format_string='%0.2f',
                    target=500):
    units = [combobox.itemText(idx) for idx in range(combobox.count())]
    converted, best_unit, idx = convert_to_best_unit(SI_value, units,
                                                     target=target)
    
    textbox.setText(format_string % converted)
    combobox.setCurrentIndex(idx)
    return converted

def get_SI_from_lineEdit(textbox, combobox):
    value = textbox.text()
    value = 0 if value.strip() == '' else float(value)
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
    def __init__(self, port, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalsSlots()
        self.setupZMQ(port)
        self.disableControls()

    def connectSignalsSlots(self):
        self.setupModeControl()
        self.setupProbeAdjustment()
        self.setupProbeViewer()
        self.setupFileManagement()
        self.setupCalibrationHints()

    def disableControls(self):
        self.groupBox_probeAdjustment.setDisabled(True)
        self.groupBox_probeViewer.setDisabled(True)
        self.groupBox_calibrationHints.setDisabled(True)
        self.pushButton_saveAs.setDisabled(True)
        self.checkBox_zmq.setDisabled(True)
        
    def enableControls(self):
        self.groupBox_probeAdjustment.setDisabled(False)
        self.groupBox_probeViewer.setDisabled(False)
        self.groupBox_calibrationHints.setDisabled(False)
        self.pushButton_saveAs.setDisabled(False)
        self.checkBox_zmq.setDisabled(False)
    
    def setupModeControl(self):
        slider = self.horizontalSlider_nmodes
        spinbox = self.spinBox_nmodes
        label = self.label_nmodeReadout
        currentMode = self.spinBox_viewMode

        def nmodes_slider_finished_callback(val):
            spinbox.setValue(val)
            self.recalculate()
        
        slider.sliderMoved.connect(nmodes_slider_finished_callback)

        def nmodes_spinbox_finished_callback():
            val = spinbox.value()
            slider.setValue(val)
            label.setText('/%d' % val)
            currentMode.setMaximum(val)
            
            if currentMode.value() > val:
                currentMode.setValue(val)
 
            self.recalculate()
                
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
        self.checkBox_updateDzWithE.stateChanged.connect(self.fullRefresh)

    def setupSliderGroup(self, textbox, minbox, slider, maxbox, unit, reset,
                         factor=1000):
        textbox.setValidator(QDoubleValidator())
        minbox.setValidator(QIntValidator())
        maxbox.setValidator(QIntValidator())
        
        # This causes problems because the value is changed when the text
        # box is updated too.
        #slider.valueChanged.connect(update_textbox)
        slider.sliderMoved.connect(lambda val : textbox.setText(str(val/1000)))
        
        
        def update_slider_range():
            value = slider.value()
            minval = int(float(minbox.text()) * factor)
            maxval = int(float(maxbox.text()) * factor)
            if maxval < minval:
                minval, maxval = maxval, minval
                minbox.setText(str(minval/factor))
                maxbox.setText(str(maxval/factor))
                
                   
            slider.setRange(minval, maxval)
            if minval > value:
                slider.setValue(minval)
            elif maxval < value:
                slider.setValue(maxval)
                
            if slider.value() != value:
                textbox.setText(str(slider.value()/factor))

        def textbox_finished_callback():
            value = float(textbox.text())
            if value < float(minbox.text()):
                minbox.setText(str(int(np.floor(value))))
            if value > float(maxbox.text()):
                maxbox.setText(str(int(np.ceil(value))))
                
            update_slider_range()
            slider.setValue(int(np.round(value*factor)))
            
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


    def setupZMQ(self, port):
        self.context = zmq.Context()
        self.port = port
        def whenChecked(state):
            # checking for hasattr just avoids any weirdness if there still is
            # a port already connected for some reason.
            if state and not hasattr(self, 'pub'):
                port, ok = QInputDialog.getText(self, 'Choose Port to Publish on', 'Port:', text=self.port)
                if ok:
                    self.port = port
                    # We wait to set up zmq until the box is checked for the
                    # first time
                    try:
                        # Define the socket to publish the probe calibration on
                        self.pub = self.context.socket(zmq.PUB)
                        self.pub.bind(self.port)
                    except zmq.error.ZMQError as e:
                        if e.errno == 98: # Port already in use
                            if hasattr(self, 'pub'):
                                delattr(self, 'pub')
                            self.checkBox_zmq.setChecked(False)
                            self.statusBar().showMessage(
                                'ØMQError: Address already in use. Perhaps another copy of PCW is running?')
                            return
                        else:
                            raise e
                        
                    self.timer = QTimer()
                    self.timer.timeout.connect(self.emitCalibration)
                    self.timer.start(1000)
                else:
                    self.checkBox_zmq.setChecked(False)
            else:
                self.timer.stop()
                self.pub.unbind(self.pub.last_endpoint)
                delattr(self, 'pub')
                self.statusBar().showMessage(
                    'ØMQ Successfully Disconnected')
                
        self.checkBox_zmq.clicked.connect(whenChecked)
    
    def setupFileManagement(self):
        self.pushButton_load.clicked.connect(self.askAndLoadFile)

        def saveAs():
            saveloc = QFileDialog.getSaveFileName(
                caption='Save Location',
                filter='Probe Calibration (*.mat);;Ptychocam-ish CXI (*.cxi)')
            if saveloc[1] != '':
                self.saveFile(saveloc[0])
                
        self.pushButton_saveAs.clicked.connect(saveAs)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()
        
    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls()]
        # We only support loading one file at once, so I just load the first
        self.loadFile(files[0])

    def setupCalibrationHints(self):
        self.lineEdit_a1.setValidator(QDoubleValidator())

        self.pushButton_loadBackground.clicked.connect(
            self.loadBackground)
        self.pushButton_loadMask.clicked.connect(
            self.loadMask)
        self.checkBox_includeMask.clicked.connect(
            self.emitCalibrationIfChecked)

        self.pushButton_viewBackground.clicked.connect(
            self.viewBackground)
        self.pushButton_viewMask.clicked.connect(
            self.viewMask)
        self.checkBox_includeBackground.clicked.connect(
            self.emitCalibrationIfChecked)

    def loadBackground(self):
        to_load = QFileDialog.getOpenFileName(
            caption='Load a Background from a Probe Calibration',
            filter='Probe Calibration (*.mat);; CXI ptycho result (*.cxi)')

        # If they cancelled the load
        if to_load[1] == '':
            return

        loaded_data = loadmat(to_load[0])
        
        if 'background' in loaded_data:
            if loaded_data['background'].shape != self.probe.shape[-2:]:
                self.statusBar().showMessage(
                    'Background shape ('
                    + str(loaded_data['background'].shape)
                    +') did not match probe.')
                return
            
            self.base_data['background'] = loaded_data['background']
            self.pushButton_viewBackground.setDisabled(False)
            self.checkBox_includeBackground.setDisabled(False)
            self.checkBox_includeBackground.setChecked(True)
        else:
            self.statusBar().showMessage('No background found in file.')

    def loadMask(self):
        to_load = QFileDialog.getOpenFileName(
            caption='Load a Mask from a Probe Calibration',
            filter='Probe Calibration (*.mat);; CXI ptycho result (*.cxi)')

        # If they cancelled the load
        if to_load[1] == '':
            return

        loaded_data = loadmat(to_load[0])
        
        if 'mask' in loaded_data:
            if loaded_data['mask'].shape != self.probe.shape[-2:]:
                self.statusBar().showMessage(
                    'Background shape ('
                    + str(loaded_data['mask'].shape)
                    + ') did not match probe.')
                return

            self.base_data['mask'] = loaded_data['mask']
            self.pushButton_viewMask.setDisabled(False)
            self.checkBox_includeMask.setDisabled(False)
            self.checkBox_includeMask.setChecked(True)
        else:
            self.statusBar().showMessage('No mask found in file.')

    def viewBackground(self):
        plt.figure()
        plt.imshow(self.base_data['background'])
        plt.colorbar()
        plt.title('Detector Background')
        plt.xlabel('j (pixels)')
        plt.ylabel('i (pixels)')
        plt.show()

    def viewMask(self):
        plt.figure()
        plt.imshow(self.base_data['mask'])
        plt.colorbar()
        plt.title('Detector Mask')
        plt.xlabel('j (pixels)')
        plt.ylabel('i (pixels)')
        plt.show()
        pass
        
    def askAndLoadFile(self):
        to_load = QFileDialog.getOpenFileName(
            caption='Load a Probe Calibration',
            filter='Probe Calibration (*.mat);; CXI ptycho result (*.cxi)')

        # If they cancelled the load
        if to_load[1] == '':
            return

        self.loadFile(to_load[0])
    
    def loadFile(self, to_load):
        # First we load the data and check that it's good.
        # Note that we also copy everything over to a separate dictionary.
        # This is done so that we can load from result files that might have
        # lots of extra info we don't need, and we don't need to keep all
        # that extra data in memory
        if to_load.split('.')[-1].lower().strip() == 'cxi':
            print('File to load is a .cxi file, presumably the output of a ptychocam reconstruction')
            loaded_data = {}
            with h5py.File(to_load, "r") as f:
                try:
                    if "entry_1/image_latest/process_1/final_illumination" in f:
                        loaded_data['probe'] = \
                            np.array(f["entry_1/image_latest/process_1/final_illumination"])
                    else:
                        self.statusBar().showMessage('.cxi file has no reconstructed probe in "entry_1/image_latest/process_1/final_illumination')
                        return
                    # maybe also check image_x and image_y?
                    if "final_res" in f:
                        pitch = np.array(f["final_res"])

                        loaded_data['basis'] = np.array([[0,-pitch],
                                                         [-pitch,0],
                                                         [0,0]])
                    else:
                        self.statusBar().showMessage('.cxi file has no infomation on the probe basis, checked "final_res".')
                        return

                    if "entry_1/instrument_1/source_1/energy" in f:
                        loaded_data['wavelength'] = hc / \
                            np.array(f["entry_1/instrument_1/source_1/energy"])
                    else:
                        self.statusBar().showMessage('.cxi file has no infomation on the probe energy, checked "entry_1/instrument_1/source_1/energy".')
                        return
                    
                    if "entry_1/instrument_1/detector_1/detector_mask" in f:
                        loaded_data['mask'] = \
                            np.array(f["entry_1/instrument_1/detector_1/detector_mask"])

                    if "entry_1/image_1/process_1/final_background" in f:
                        loaded_data['background'] = \
                            np.fft.ifftshift(np.array(f["entry_1/image_1/process_1/final_background"]))
                    
                except Exception as e:
                    raise e
            # now I populate 'loaded_data

        else:
            try:
                loaded_data = loadmat(to_load)
            except ValueError as e:
                self.statusBar().showMessage(
                    'File does not appear to be a .mat file, unable to load.')
                return
            
        if not 'probe' in loaded_data:
            self.statusBar().showMessage(
                'Dataset has no probe info, not loading')
            return

        if len(loaded_data['probe'].shape) < 2:
            self.statusBar().showMessage(
                'Probe has not enough dimensions, not loading')
            return

        if not 'wavelength' in loaded_data:
            self.statusBar().showMessage(
                'Dataset has no wavelegth info, not loading')
            return

        
        if not 'basis' in loaded_data:
            self.statusBar().showMessage(
                'Dataset has no info on the probe basis, not loading')
            return

        # All the checks need to go before any loading happens, otherwise
        # it could end up in a bad state during a failed load

        if 'oversampling' in loaded_data:
            self.spinBox_oversampling.setValue(loaded_data['oversampling'].ravel()[0])
        else:
            self.spinBox_oversampling.setValue(1)
        
        self.base_data = {}
        self.base_data['probe'] = loaded_data['probe']
        
        # This always unravels all the modes so the probe is n_modesxNxM
        if len(self.base_data['probe'].shape) == 2:
            self.base_data['probe'] = np.expand_dims(self.base_data['probe'], 0)
            
        n_modes = np.prod(self.base_data['probe'].shape[:-2])
        self.base_data['probe'] = self.base_data['probe'].reshape(
            (n_modes,) + self.base_data['probe'].shape[-2:])

        # Since this is coming from a .mat file, the constant gets loaded
        # in as part of a 2D matrix
        self.base_data['wavelength'] = loaded_data['wavelength'].ravel()[0]
        
        self.base_data['basis'] = loaded_data['basis']
        self.basis = loaded_data['basis']

        if 'mask' in loaded_data:
            self.base_data['mask'] = loaded_data['mask']
            self.pushButton_viewMask.setDisabled(False)
            self.checkBox_includeMask.setDisabled(False)
            self.checkBox_includeMask.setChecked(True)
        else:
            self.pushButton_viewMask.setDisabled(True)
            self.checkBox_includeMask.setDisabled(True)
            self.checkBox_includeMask.setChecked(False)

        if 'background' in loaded_data:
            self.base_data['background'] = loaded_data['background']
            self.pushButton_viewBackground.setDisabled(False)
            self.checkBox_includeBackground.setDisabled(False)
            self.checkBox_includeBackground.setChecked(True)
        else:
            self.pushButton_viewBackground.setDisabled(True)
            self.checkBox_includeBackground.setDisabled(True)
            self.checkBox_includeBackground.setChecked(False)
        
        # This is initializing self.probe, which stores the processed probe
        self.probe = self.base_data['probe']

        # The energy slider
        self.energy = hc / self.base_data['wavelength']
        energy = autoset_from_SI(self.energy, self.lineEdit_e, self.comboBox_eUnits,
                                 format_string='%0.2f')
        minval = int(np.floor(energy*0.9))
        maxval = int(np.ceil(energy*1.1))
        self.lineEdit_eMin.setText(str(minval))
        self.lineEdit_eMax.setText(str(maxval))
        # TODO: This is hard-coded, but it should automatically update
        self.horizontalSlider_e.setRange(minval*1000, maxval*1000)
        self.horizontalSlider_e.setValue(int(np.round(energy*1000)))

        # TODO: This will fail if the units box is changed
        self.pushButton_eReset.resetTo = self.lineEdit_e.text()
        
        # And the Propagation Slider
        pitches = np.linalg.norm(self.base_data['basis'],axis=0)
        min_pitch = np.min(pitches)
        DOF = 2 * min_pitch**2 / self.base_data['wavelength']
        units = [self.comboBox_dzUnits.itemText(idx) for idx in
                 range(self.comboBox_dzUnits.count())]
        
        prop_limit = 200 * DOF
        prop_limit, unit, idx = convert_to_best_unit(prop_limit, units)
        prop_limit = int(np.ceil(prop_limit))

        self.lineEdit_dz.setText('0')
        self.lineEdit_dzMin.setText(str(-prop_limit))
        self.lineEdit_dzMax.setText(str(prop_limit))
        # TODO: Again, this is hard-coded
        self.horizontalSlider_dz.setRange(-prop_limit*1000, prop_limit*1000)
        self.horizontalSlider_dz.setValue(0)
        self.comboBox_dzUnits.setCurrentIndex(idx)

        if 'A1' in loaded_data:
            A1_SI = loaded_data['A1'].ravel()[0]
            autoset_from_SI(A1_SI, self.lineEdit_a1,
                            self.comboBox_a1Units, format_string='%0.3f',
                            target=40)
        elif 'a1' in loaded_data:
            A1_SI = loaded_data['a1'].ravel()[0]
            autoset_from_SI(A1_SI, self.lineEdit_a1,
                            self.comboBox_a1Units, format_string='%0.3f',
                            target=40)
        else:
            self.lineEdit_a1.setText('')
        # NOTE: the spinbox has to be updated last, because it will trigger
        # an automatic recalculation of the probe, and that will fail if the
        # energy is not set (energy defaults to 0). This annoying requirement
        # could be avoided if I can figure out how to make the recalculation
        # only trigger when the user makes a change, but I can't, so we're
        # stuck with this confusing thing for now.
        # Same goes for the orthogonalized mode checkbox
        
        # Now we update the controls to be initialized with resonable values
        if hasattr(self, 'orthogonalized_probes'):
            delattr(self, 'orthogonalized_probes')
        self.checkBox_ortho.setCheckState(False)

        self.horizontalSlider_nmodes.setRange(1, n_modes)
        self.spinBox_nmodes.setRange(1, n_modes)
        self.spinBox_nmodes.setValue(n_modes)
            
        # Now we plot the loaded probe
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
                                   extent=extent, interpolation='none')
        divider = make_axes_locatable(self.axes)
        
        cax = divider.append_axes('right', size='5%', pad=0.05)

        self.fig.colorbar(self.im, cax=cax)

        self.updateFigure()
        self.enableControls()

        self.statusBar().showMessage(
            'File "' + to_load + '" loaded successfully.')

    def fullRefresh(self):
        self.recalculate()
        self.updateFigure()
        self.emitCalibrationIfChecked()
        
    def recalculate(self):
        # Order of operations
        # 1: Orthogonalize and clip probes
        # 2: Update the probe energy
        # 3: Propagate correct distance

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

        self.energy = get_SI_from_lineEdit(self.lineEdit_e,
                                           self.comboBox_eUnits)
        base_energy = hc / self.base_data['wavelength']
        if hasattr(self, 'last_base_energy'):
            if last_base_energy != base_energy:
                delattr(self, 'universal_prop')
        last_base_energy = base_energy
        
        energy_ratio = self.energy / base_energy
        energy_changed_probes = change_energy(clipped_probes, energy_ratio)
        
        A1 = get_SI_from_lineEdit(self.lineEdit_a1, self.comboBox_a1Units)
        energy_propagation_correction = A1 * (base_energy - self.energy)

        self.basis = self.base_data['basis'] / energy_ratio
        
        if not hasattr(self, 'universal_prop') or \
           self.universal_prop.shape[-2:] != self.probe.shape[-2:]:
            # Whenever something happens that has to change this, for example
            # when the energy changes, then the pattern is to delete the
            # universal propagator and it will be recalculated here.
            
            shape = clipped_probes.shape[-2:]
            spacing = t.as_tensor(np.linalg.norm(
                self.basis, axis=0))
            wavelength = hc / self.energy
            self.universal_prop = make_universal_propagator(
                shape, spacing, wavelength)
            self.universal_prop = (1j * self.universal_prop)
        z = get_SI_from_lineEdit(self.lineEdit_dz, self.comboBox_dzUnits)

        if self.checkBox_updateDzWithE.checkState():
            z += energy_propagation_correction
        
        if z != 0:
            prop = t.exp(z * self.universal_prop)
            self.probe = propagators.near_field(
                energy_changed_probes, prop).numpy()
        else:
            self.probe = energy_changed_probes.numpy()
            
    def updateFigure(self):
        probe_mode = self.spinBox_viewMode.value()
        to_show = self.probe[probe_mode-1]
        
        basis_norm = np.linalg.norm(self.basis, axis=0)
        
        if self.radioButton_fourier.isChecked():
            title = ' in Fourier Space'
            to_show = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(to_show), norm='ortho'))

            extent = [-1/(2*basis_norm[0]),1/(2*basis_norm[0]),
                      -1/(2*basis_norm[1]),1/(2*basis_norm[1])]
            _,unit,_ = convert_to_best_unit(1/extent[1],
                                            ['A','nm','um','mm','m','km'],
                                            target=1/100)
            extent = [1/convert_from_SI(1/e, unit) for e in extent]
            #convert_to_best_unit(value, unit_options, target=500):
            self.axes.set_xlabel('kx (cycles/' + unit + ')')
            self.axes.set_ylabel('ky (cycles/' + unit + ')')

        else:
            title = ' in Real Space'
            extent = [0, self.base_data['probe'].shape[-1]*basis_norm[1], 0,
                      self.base_data['probe'].shape[-2]*basis_norm[0]]
            _, unit, _ = convert_to_best_unit(extent[1],
                                              ['A','nm','um','mm','m','km'],
                                              target=100)
            extent = [convert_from_SI(e, unit) for e in extent]
            
            
            self.axes.set_xlabel('x (' + unit + ')')
            self.axes.set_ylabel('y (' + unit + ')')
            
        title = ' of Mode ' + str(probe_mode) + title

        if self.radioButton_amplitude.isChecked():
            title = 'Amplitude' + title
            to_show = np.abs(to_show)
        else:
            title = 'Phase' + title
            to_show = np.angle(to_show)
            
        self.im.set_data(to_show)

        if self.radioButton_amplitude.isChecked():
            # This cmap sets the scale so that one pixel in every row is
            # saturated on average. It's a good compromise between not updating
            #the colormap and having it flicker like crazy due to outliers
            self.im.set(cmap='viridis',
                        clim=[0, np.quantile(to_show,1-1/self.probe.shape[-1])],
                        extent=extent)
            
        else:
            self.im.set(cmap='twilight',
                        clim=[-np.pi,np.pi],
                        extent=extent)

        self.axes.set_title(title)
        
        self.canvas.draw_idle()


    def collect_results(self):
        results = copy(self.base_data)

        results['wavelength'] = hc / self.energy

        results['probe'] = self.probe
        results['basis'] = self.basis

        results['oversampling'] = self.spinBox_oversampling.value()

        if self.lineEdit_a1.text().strip() != '':
            A1 = float(self.lineEdit_a1.text().strip())
            A1_unit = self.comboBox_a1Units.currentText()
            results['A1'] = convert_to_SI(A1, A1_unit)

        if not self.checkBox_includeBackground.checkState():
            if 'background' in results:
                results.pop('background')
        if not self.checkBox_includeMask.checkState():
            if 'mask' in results:
                results.pop('mask')

        return results

    def emitCalibrationIfChecked(self):
        if self.checkBox_zmq.checkState():
            self.emitCalibration()
        
    def emitCalibration(self):
        save_data = self.collect_results()
        
        self.pub.send_pyobj(save_data)
        currentTime = datetime.datetime.now().strftime('%H:%M:%S')
        self.statusBar().showMessage('Published to ØMQ ('
                                     + self.pub.last_endpoint.decode("utf-8")
                                     + ') at ' + currentTime)
        
    def saveFile(self, filename):
        save_data = self.collect_results()
        # Not sure if this is the most sane approach
        if filename[-4:].lower() == '.cxi':
            self.statusBar().showMessage('Save to ptychocam-ish .cxi not yet implemented')
            return

        # The default if no extension is given is a .mat file
        elif filename[-4:].lower() != '.mat':
            filename += '.mat'
        savemat(filename, save_data)
        self.statusBar().showMessage('Saved calibration to '+filename)
        

def main(argv=sys.argv):

    parser = argparse.ArgumentParser(description='Probe Calibration Wizard Seven Thousand Twelve')
    parser.add_argument('filename', nargs='?', type=str, help='A file to load in while opening the app', default='')
    parser.add_argument('--port', '-p', type=str, help='ZeroMQ port to broadcast on', default='tcp://*:5557')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setApplicationName('PCW7012')
    #app.setStyleSheet(qdarkstyle.load_stylesheet())

    win = Window(args.port)
    if args.filename.strip() != '':
        win.loadFile(args.filename)
    win.show()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
