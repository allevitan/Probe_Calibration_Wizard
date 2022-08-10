# Probe_Calibration_Wizard

This is a simple GUI tool to manipulate probe calibrations for RPI and Ptychography. It allows for the manipulation of the number of probe modes, the probe propagation state, and corrections for energy changes under fixed detector positions.

The program can publish a stream of live-updating probe calibrations over ZeroMQ, which allows it to be connected to a live RPI reconstruction program, allowing the calibration tool to essentially act as a focus knob for ongoing RPI reconstructions.


## Installation

