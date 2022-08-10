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
$ pcw
```

This function also takes the port to publish it's ZeroMQ stream on as an optional command line argument, e.g.

```console
$ pcw tcp://*:5557
```

## Data Format

This tools reads and writes to .mat files, which are expected to contain at minimum the following entries:

- `probe`, an NxMxM complex-valued array. N is number of modes, M is the spatial dimension. Only square probes are currently supported.
- `wavelength`, the wavelength of the light in meters.
- `basis`, a 3x2 or 2x2 matrix describing the real-space basis for the probe array in either 3D or 2D

Optionally, the following additional information can be stored:

- `A0`, the focal length per energy of the zone plate used to form this probe, in meters/Joule.
- `it
- `background`, a background to use for RPI reconstructions based on this probe
- `mask`, a detector mask to use for RPI reconstructions based on this probe
- `iterations`, a suggested number of iterations to use for RPI reconstructions based on this probe
- `resolution`, a suggested resolution (number of pixels) to use for RPI reconstructions based on this probe