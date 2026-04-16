# Hands on Session - Day 1 - 16th April 2026 

## Aim 

This tutorial provides a practical introduction to the Quantum-ESPRESSO
(QE), a leading open-source software for electronic structure. Using
Platinum as our primary example, we will focus on the hands-on
application of key computational parameters. Designed specifically for
beginners, this guide bypasses complex theoretical derivations to focus
on the \"how-to\" of benchmarking and conducting ground-state DFT
simulations. Finally, you will learn how to extract the data necessary
for training deep neural network potentials.

## Objectives 

The objectives of this tutorial session are:

-   Learn to create files and scripts for running QE calculations

-   Understand the anatomy of the QE input file

-   Learn to Submit QE jobs

-   Benchmark DFT parameters

-   Learn to perform Geometry relaxation

-   Perform a large number of DFT calculations using Quantum Espresso
    and the PBE functional

-   Learn to prepare data to train a machine-learning model

-   Create configurations suitable to train a model for a crystalline
    solid using random perturbations from the equilibrium atomic
    positions

-   Prepare files in a format appropriate for training a model for the
    PES using DeePMD-kit

## Running DFT Calculations Using QE 

Running simulations with the Quantum ESPRESSO PWSCF module requires two
core components after installation and compilation of the pw.x
executable and the enviroment.

1.  Pseudopotentials in the .upf format.

2.  A structured input file defining your system.

### Pseudopotentials: 
While the theoretical nuances of pseudopotentials are beyond the scope
of this tutorial, choosing a reliable library is critical for accurate
DFT. For the Pt example, we will use an [Optimized Norm-Conserving
Vanderbilt (ONCV)
pseudopotentials](http://quantum-simulation.org/potentials/sg15_oncv/upf/)
tailored for the PBE functional.

### Input file anatomy: 

While a comprehensive, all-in-one guide for PWscf keywords can be found
[here](https://www.quantum-espresso.org/Doc/INPUT_PW.html), this tutorial focuses on the essential parameters needed to get
started.

To begin, let's examine the sample input file, pw.in, located in the
root directory. We will start by breaking down the
[`&control`] namelist:

```fortran
    &control
       restart_mode = 'from_scratch',
       calculation  = 'scf',
       prefix       = 'scf',
       outdir       = './',
       pseudo_dir = './',
       tprnfor = .true.,
    /
```

The formatting of the namelists starts with & and ends with /, while
keywords are separated by commas. The first keyword
[`calculation=‘scf’`], entails that
we will be performing a single-point self-consistent field (SCF)
calculation. [`prefix=‘scf’`]sets
all the output files with this prefix. The
[`pseudo_dir`] keyword provides the
path to the pseudopotential files, while the
[`outdir`]keyword specifies the path
where the output will be written.
[`restart_mode=‘from_scratch’`]
implies that we are starting the calculation from scratch and not
restarting it. While the SCF calculation is generally meant to get only
the energy, for training deep neural network potentials (DP's), atomic
forces are also needed. To evaluate this, the tprnfor keyword is set to
.true.

Next, let's look at the [`&system`]
namelist:

```fortran
    &system
       ibrav = 0,
       ntyp = 1,
       nat = 4,
       ecutwfc = 70,
       ecutrho = 280,
       nbnd = 44,
       occupations = 'smearing', 
       smearing = 'marzari-vanderbilt',
       degauss = 0.02d0,
       input_dft = 'PBE',
    /
```

The next section starts with the specification of the lattice using the
keyword [`ibrav`]. Here, it's set to
0, implying that a free format is adopted where the explicit lattice
vectors are provided in the input. The keyword
[`ntyp`] specifies the number of
types of atoms in the system, while
[`nat`] specifies the total number of
atoms in the system. [`ecutwfc`]
specifies the kinetic energy cutoff for the planewave basis set, while
[`ecutrho`] specifies the cutoff for
the charge density. Note that both of these are in units of Rydberg
(Ry). These parameters are largely determined by the choice of
pseudopotentials (PP's) used in the calculation. Generally, it is
advisable for ecutrho to be 4x the ecutwfc for norm-conserving PP's and
at least 8 to 12x for ultrasoft PP's. The next keyword,
[`nbnd`], specifies the number of
electronic states to be calculated. For an insulator, this is usually \#
valence electrons/2, while for metals, it is generally advisable to add
at least 20% more. The keyword
[`occupations`] specifies how the
occupation of the electronic states is accomplished. This is mainly for
avoiding numerical problems associated with the finite sampling of the
Brillouin zone and the properties of the system itself. For a metallic
system, some kind of smooth function ('smearing') is usually needed to
avoid instabilities, while a fixed occupation may be used for an
insulator with a band gap. The associated keyword
[`smearing`] specifies the type of
smooth function approximation that will be used, with the keyword
[`degauss`] specifying the value of
Gaussian spreading to be used in the Brillouin zone integration.
Finally, the keyword [`input_dft`]
specifies the exchange-correlation functional used in the calculation.
In this example, the PBE functional is used.

Next is [`&electrons`]:

```fortran
    &electrons
       electron_maxstep = 200,
       diagonalization = 'david',
       conv_thr = 1e-06,
       mixing_ndim = 8,
       mixing_beta = 0.1,
    /
```

This next section controls the convergence of the SCF cycles. The
keyword [`electron_maxstep`]
specifies the total number of steps that will be considered in the SCF
cycle. The diagonalization keyword specifies the algorithm to be used
for diagonalizing the Hamiltonian, whereas the
[`conv_thr`] provides the energy
convergence threshold for the SCF calculation.
[`mixing_ndim`] and
[`mixing_beta`] control the mixing
factor between SCF steps for achieving self-consistency. This ends the
sections with the different namelists.

Lastly we come to the cards (note that these are not namelists and have
different syntax) associated with the structure and k-points:
```fortran
    ATOMIC_SPECIES
    Pt 195.084 Pt_ONCV_PBE-1.2.upf
    
    K_POINTS automatic
    8 8 8  0 0 0

    CELL_PARAMETERS angstrom
    3.94315036000000 0.00000000000000 0.00000000000000
    0.00000000000000 3.94315036000000 0.00000000000000
    0.00000000000000 0.00000000000000 3.94315036000000
   
    ATOMIC_POSITIONS angstrom
    Pt 0.0000000000 0.0000000000 0.0000000000
    Pt 0.0000000000 1.9715751800 1.9715751800
    Pt 1.9715751800 0.0000000000 1.9715751800
    Pt 1.9715751800 1.9715751800 0.0000000000

```

The [`ATOMIC_SPECIES`] section lists
the atoms in the system along with the corresponding pseudopotential
(PP) used.This is followed by the
[`CELL_PARAMETERS`] section, where we
provide the explicit cell vectors (needed as we set ibrav=0). Note that
the units are specified to be in Angstroms. The
[`ATOMIC_POSITIONS`] section lists
the explicit atomic positions in the cell, in the following format: Atom
type followed by the corresponding x, y, z coordinates.
[`K_POINTS gamma`] specifies that
only the gamma point will be used to sample the Brillouin zone.

### Typical output file:

In summary, Quantum ESPRESSO performed a self-consistent field (SCF)
cycle, iteratively refining the system's eigenvectors and eigenvalues
from an initial trial guess until the ground-state electronic structure
of the silicon system was achieved.The most important quantities, among
other things in the output file are the total energy of the system and
the atomic forces as shown here:

```fortran

     !    total energy         =    -970.46570737 Ry
     estimated scf accuracy    <       0.00000010 Ry
     smearing contrib. (-TS)   =       0.00312041 Ry
     internal energy E=F+TS    =    -970.46882778 Ry

     The total energy is F=E-TS. E is the sum of the following terms:
     one-electron contribution =    -240.92623707 Ry
     hartree contribution      =     160.78689044 Ry
     xc contribution           =     -92.90590430 Ry
     ewald contribution        =    -797.42357684 Ry

     convergence has been achieved in 8 iterations

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    3 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    4 type  1   force =     0.00000000    0.00000000    0.00000000

     Total force =     0.000000     Total SCF correction =     0.000000

```

To find the Total Energy of your SCF calculation, search the output file
for the exclamation point (!) character. This prefix marks the final
converged energy value. Immediately following this line, you will find a
breakdown of the energy terms, the number of iterations taken to reach
convergence, and the calculated forces acting on each atom.

### Benchmarking and Geometry Optimization 

Benchmarking your DFT parameters is a critical first step. Since the
accuracy of a Machine Learning Potential is fundamentally limited by the
quality of the training data, your DFT \"labels\" must be converged and
reliable.

In this section, we will demonstrate how to benchmark two of the most
influential parameters in Quantum ESPRESSO: the kinetic energy cutoff
(ecutwfc) and the k-point grid density.

-   Kinetic Energy Cutoff (ecutwfc): In plane-wave DFT, the ecutwfc
    parameter determines the size of the basis set. We must select a
    cutoff high enough that the total energy of the system becomes
    stable (converged). Essentially, we are balancing the trade-off
    between computational precision (larger basis set) and
    time-to-solution.

    -   To begin the convergence test:

    -   Navigate to the energycutoffandkpointsoptimization/energycutoff directory.

    -   Locate the Python script named EnergyCutoffandKpointsScript.ipynb.

    Execute the script to automatically generate a series of QE input
    files with plane-wave cutoffs ranging from 40 to 100 Ry.

    Once your calculations are finished, it is time to evaluate the
    convergence of the total energy.You will notice that as the ecutwfc
    increases, the total energy
    typically decreases. However, you will eventually reach a point of
    diminishing returns. A reliable DFT protocol requires selecting a
    cutoff value just beyond this \"plateau\"---the point where further
    increases in the basis set size no longer lead to significant
    changes in the total energy.

    To help you identify this threshold, we have provided an IPython
    script (EnergyCutoffandKpointsScript.ipynb) to visualize the trend. It iterates through each
    cutoff value and uses the ASE read() function to extract the total
    energy from the corresponding output files (pw.out). It generates a
    graph of Total Energy vs. Cutoff Energy and saves the result as
    Convergence_energycutoff_Pt.png. You can run this analysis using the Jupyter Notebook
    environment.

-   K-point Grid Convergence: Similarly, achieving energy convergence
    requires sampling an appropriate number of k-points within the
    periodic Brillouin zone. A grid that is too sparse will fail to
    capture the electronic structure accurately, while a grid that is
    too dense will unnecessarily increase computational costs. To begin
    this benchmarking step:

    -   Navigate to the energycutoffandkpointsoptimization/kpoints directory.

    -   Locate the Python script named EnergyCutoffandKpointsScript.ipynb .

    -   Run the script

    This script will generate and execute a series of input files with
    increasing k-grid densities, ranging from 4x4x4 to 12x12x12.
    
    Similar to the analysis for energy cutoff, we have provided EnergyCutoffandKpointsScript.ipynb script to visualise the kpoint convergence. It will save the graph of Total Energy vs Kpoints as Convergence_kpoints_Pt.png
    
-   Structural Optimization: In Quantum ESPRESSO, structural
    optimization is generally categorized into two types:

    -   relax: Only the atomic positions are allowed to change, while
        the cell dimensions remain fixed.

    -   vc-relax: Both the atomic positions and the lattice constants
        (cell volume/shape) are allowed to vary simultaneously.

    In a relax calculation, a full electronic SCF cycle is converged at
    every ionic step. The algorithm then moves the atoms (and
    potentially the cell) based on those calculated forces, repeating
    the process until the forces fall below your specified
    forc_conv_thr. To obtain an accurate starting point for our platinum
    model, we will perform a vc-relax optimization. Later, we will apply
    random perturbations to this optimized structure to generate diverse
    training configurations for our deep neural network potential.

    If you compare the vc-relax input file to the standard SCF file, you
    will notice several critical additions:

    -   The &control Namelist:

        -   calculation = 'vc-relax': Specifies that the engine should
            perform a variable-cell relaxation.

        -   forc_conv_thr: Sets the convergence threshold for the
            forces; the calculation will stop once all atomic forces
            fall below this value.

    -   Additional Namelists:

        -   &ions: Contains parameters for controlling the movement of
            atoms.

        -   &cell: Contains parameters for the dynamics of the
            simulation box itself.

    By default, Quantum ESPRESSO utilizes the BFGS algorithm (a
    sophisticated quasi-Newton method) to efficiently navigate the
    potential energy surface and find the local minimum.

    After the simulation completes, it is standard practice to validate
    your results against established data:

    -   Lattice Constant: Compare your optimized value with the
        experimental literature value for Pt ($3.92 Å$). A small
        deviation (typically $< 1–2 \%$) is expected depending on your
        choice of functional.

    -   Residual Forces: Verify that the forces on the atoms have
        effectively approached zero. This confirms that the system has
        reached a true local minimum on the potential energy surface.

## Preparing Training Data 

Deep Potential models for the Potential Energy Surface (PES) are built
upon deep neural networks. These models are typically trained on
datasets where the potential energy and atomic forces have been
calculated using Density Functional Theory (DFT).

The standard development pipeline consists of three core stages:

1.  Generating a diverse set of atomistic configurations by sampling the
    relevant configuration space.

2.  Computing the precise ground-state energies and forces for those
    configurations via DFT.

3.  Using the DeepMD-kit to train the DP model and verify its accuracy.

The strategy for building a training database depends heavily on the
intended application of the final model. A robust database consists of
multiple \"frames,\" where each frame contains chemical and
configurational information, DFT-computed forces, and potential energy.

In this section, we will walk you through the process of preparing a
training dataset for DeepMD-kit. For platinum in the fcc structure, we generate configurations by applying random
perturbations to the equilibrium atomic positions and cell vectors. Then
we will perform Quantum ESPRESSO calculations for every sampled frame to
determine its \"ground truth\" potential energy and atomic
forces.Finally, these results will be parsed and converted into the
specific DeepMD-kit input formats, making them ready for the training

To generate our training data, we will use a Python script to automate
the creation of perturbed crystalline configurations. The workflow
follows these steps:

1.  Building the SupercellFirst, the script imports the previously
    optimized 4-atom unit cell of bulk platinum as an ASE Atoms object.
    we expand this unit cell using a $2 \times 2 \times 2$
    transformation matrix, resulting in a 32-atom supercell.

2.  Then, apply random displacements to the atomic positions of a bulk
    Pt supercell and vary the lattice parameters. Random displacements
    within the defined maximum displacement will be added to the
    equilibrium atomic positions, and random fractional changes within
    the defined maximum cell change will be applied to the lattice
    parameters.

3.  The script is configured to generate a total of 100 unique frames,
    each saved as a ready-to-run Quantum ESPRESSO input file. The
    perturbations are drawn from a uniform distribution with the
    following constraints:Maximum Cell Change: $1\%$ deviation from the
    ground-state lattice.Maximum Atomic Displacement: $0.01$ Å from the
    equilibrium positions.

4.  The next step is to label these configurations by calculating the
    energies and forces using DFT.

5.  The raw data from the pw.out file is converted to raw file format explained in Table [1].


### Table 1: Input file format required by deepMD-kit


| Raw file | Property | Dimension |
| :--- | :--- | :--- |
| **type.raw** | Atom type indexes | N atoms |
| **energy.raw** | Frame energies | Nframes in eV |
| **box.raw** | Box dimension | Nframes * 3 * 3 in Å |
| **coord.raw** | Atomic coordinates | Nframes * Natoms * 3 in Å |
| **force.raw** | Atomic forces | Nframes * Natoms * 3 in eV/Å |


It is imperative that these human-readable files are converted to Numpy Binary Data (.npy) files to be used in training by DeePMD-kit. This can be accomplished by a simple shell script.
