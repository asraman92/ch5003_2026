#!/bin/bash
# Default value for nline_per_set
nline_per_set=2000

# Override with the first command-line argument if provided
if [[ $# -ge 1 ]]; then
    nline_per_set=$1
fi

# Clean up previous runs
rm -fr set*

if ! [ -f box.raw ]; then
    echo "Error: box.raw not found. Exiting."
    exit 1
fi

echo "nframe is $(cat box.raw | wc -l)"
echo "nline per set is $nline_per_set"

# Split raw data files into smaller, numbered chunks
split box.raw -l $nline_per_set -d -a 3 box.raw.
split coord.raw -l $nline_per_set -d -a 3 coord.raw.

# Conditionally split other raw files if they exist
test -f energy.raw && split energy.raw -l $nline_per_set -d -a 3 energy.raw.
test -f force.raw && split force.raw -l $nline_per_set -d -a 3 force.raw.
test -f virial.raw && split virial.raw -l $nline_per_set -d -a 3 virial.raw.
test -f atom_ener.raw && split atom_ener.raw -l $nline_per_set -d -a 3 atom_ener.raw.
test -f fparam.raw && split fparam.raw -l $nline_per_set -d -a 3 fparam.raw.
test -f dipole.raw && split dipole.raw -l $nline_per_set -d -a 3 dipole.raw.
test -f polarizability.raw && split polarizability.raw -l $nline_per_set -d -a 3 polarizability.raw.
test -f atomic_dipole.raw && split atomic_dipole.raw -l $nline_per_set -d -a 3 atomic_dipole.raw.
test -f atomic_polarizability.raw && split atomic_polarizability.raw -l $nline_per_set -d -a 3 atomic_polarizability.raw.

# Get the number of sets created by counting all box.raw.*** files
nset=$(ls -1 box.raw.* | wc -l)
nset_1=$((nset - 1))
echo "will make $nset sets"

# Loop through each set
for ii in $(seq 0 $nset_1); do
    pi=$(printf "%03d" $ii)
    echo "making set $ii ($pi)"
    # Create directory with a dot and move files
    mkdir "set.$pi"
    mv "box.raw.$pi" "set.$pi/box.raw"
    mv "coord.raw.$pi" "set.$pi/coord.raw"
    test -f "energy.raw.$pi" && mv "energy.raw.$pi" "set.$pi/energy.raw"
    test -f "force.raw.$pi" && mv "force.raw.$pi" "set.$pi/force.raw"
    test -f "virial.raw.$pi" && mv "virial.raw.$pi" "set.$pi/virial.raw"
    test -f "atom_ener.raw.$pi" && mv "atom_ener.raw.$pi" "set.$pi/atom_ener.raw"
    test -f "fparam.raw.$pi" && mv "fparam.raw.$pi" "set.$pi/fparam.raw"
    test -f "atomic_dipole.raw.$pi" && mv "atomic_dipole.raw.$pi" "set.$pi/atomic_dipole.raw"
    test -f "atomic_polarizability.raw.$pi" && mv "atomic_polarizability.raw.$pi" "set.$pi/atomic_polarizability.raw"
    cd "set.$pi"

    # Convert raw files to .npy format using Python and NumPy with multiline code blocks
    python3 <<END
import os
import numpy as np

if os.path.isfile('box.raw'):
    data = np.loadtxt('box.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('box', data)
if os.path.isfile('coord.raw'):
    data = np.loadtxt('coord.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('coord', data)
if os.path.isfile('energy.raw'):
    data = np.loadtxt('energy.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('energy', data)
if os.path.isfile('force.raw'):
    data = np.loadtxt('force.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('force', data)
if os.path.isfile('virial.raw'):
    data = np.loadtxt('virial.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('virial', data)
if os.path.isfile('atom_ener.raw'):
    data = np.loadtxt('atom_ener.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('atom_ener', data)
if os.path.isfile('fparam.raw'):
    data = np.loadtxt('fparam.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('fparam', data)
if os.path.isfile('dipole.raw'):
    data = np.loadtxt('dipole.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('dipole', data)
if os.path.isfile('polarizability.raw'):
    data = np.loadtxt('polarizability.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('polarizability', data)
if os.path.isfile('atomic_dipole.raw'):
    data = np.loadtxt('atomic_dipole.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('atomic_dipole', data)
if os.path.isfile('atomic_polarizability.raw'):
    data = np.loadtxt('atomic_polarizability.raw', ndmin=2)
    data = data.astype(np.float32)
    np.save('atomic_polarizability', data)
END
    rm *.raw
    cd ..
done
echo "done"
