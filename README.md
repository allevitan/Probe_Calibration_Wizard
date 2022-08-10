# Probe_Calibration_Wizard

This is a simple GUI tool to manipulate probe calibrations for RPI and Ptychography. It allows for the manipulation of the number of probe modes, the probe propagation state, and corrections for energy changes under fixed detector positions.

The program can publish a stream of live-updating probe calibrations over ZeroMQ, which allows it to be connected to a live RPI reconstruction program, allowing the calibration tool to essentially act as a focus knob for ongoing RPI reconstructions.


## Installation

Once the various dependencies (listed in setup.py) are installed, PCW can be installed via:

```console
$ pip install -e .
```

The "-e" flag for developer-mode installation is recommended so updates to the git repo can be immediately included in the installed program.

## Usage

To run the program, simply run:

```console
$ pwc
```

This function also takes the port to publish it's ZeroMQ stream on as an optional command line argument, e.g.

```console
$ pwc tcp://*:5557
```

