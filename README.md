## Installation Guide: 

### 1. Windows Installation

1) Open Command Prompt and run the following commands:
```
curl [https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) -o .\miniconda.exe
start /wait "" .\miniconda.exe /S
del .\miniconda.exe
```

2) Open Anaconda Prompt from your Start Menu and run the following:
```
conda create -n myenv python=3.10 -y
conda activate myenv
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install jupyter
conda install conda-forge::ase
conda install -c conda-forge mdanalysis
```
### 2. macOS Installation

1) Open Terminal and run the following commands:
```
mkdir -p ~/miniconda3
curl [https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh) -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init $(basename $SHELL)
```
2) After restarting your Terminal, run the following:
```
conda create -n myenv python=3.10 -y
conda activate myenv
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install jupyter
conda install conda-forge::ase
conda install -c conda-forge mdanalysis
```
### 3. Launching Jupyter Notebook
To start working with ASE, ensure your environment is active and launch the interface:
```
conda activate myenv
jupyter notebook
```

# Hands on Session - Day 1 - 16th April 2026

### Aim 

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
calculation. [`prefix=‘scf’`] sets
all the output files with this prefix. The
[`pseudo_dir`] keyword provides the
path to the pseudopotential files, while the
[`outdir`] keyword specifies the path
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
        deviation (typically < 1–2 %) is expected depending on your
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
    following constraints:Maximum Cell Change: 1% deviation from the
    ground-state lattice.Maximum Atomic Displacement: $0.01$ Å from the
    equilibrium positions.

4.  The next step is to label these configurations by calculating the
    energies and forces using DFT.

5.  The raw data from the pw.out file is converted to raw file format explained in Table 1.


### Table 1: Input file format required by deepMD-kit


| Raw file | Property | Dimension |
| :--- | :--- | :--- |
| **type.raw** | Atom type indexes | N atoms |
| **energy.raw** | Frame energies | Nframes in eV |
| **box.raw** | Box dimension | Nframes * 3 * 3 in Å |
| **coord.raw** | Atomic coordinates | Nframes * Natoms * 3 in Å |
| **force.raw** | Atomic forces | Nframes * Natoms * 3 in eV/Å |


It is imperative that these human-readable files are converted to Numpy Binary Data (.npy) files to be used in training by DeePMD-kit. This can be accomplished by a simple shell script.


# Hands on Session - Day 2 - 21st April 2026 

## Aim 

The goal of this step is to use the DFT energies and forces generated in the previous tutorial to train a machine-learning model with DeePMD-kit.Additionally, we also demonstrate the active learning protocol used to build a robust training dataset for a converged Deep Potential (DP) model. Basedon the principles established by [\"Zhang et.al., Phys. Rev. Mater., 3, 023804\"](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.3.023804), we will iteratively refine the initial platinum model.

Please note that due to time constraints, this session serves as a functional demonstration of the workflow; the resulting model is a baseline and may require further refinement before use in high-level production simulations.


## Objectives 

The objectives of this tutorial session are:

-   Train a DeePMD model to accurately represent the potential energy
    surface of platinum.

-   Understanding the inputs and ouputs of the training process.

-   Running molecular dynamics simulations using the trained model.

-   Quantify uncertainty by using an ensemble of models to estimate
    errors in predicted interatomic forces and preparing those configurations for
    labeling.


## Anatomy of a typical input file for training 

```
    {
    "_comment": "that's all",
    "model": {
        "type_map": [
            "Pt"
        ],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [
                64
            ],
            "rcut_smth": 3.0,
            "rcut": 6.0,
            "neuron": [
                25,
                50,
                100
            ],
            "resnet_dt": false,
            "axis_neuron": 12,
            "seed": 1,
            "_comment": " that's all"
        },
        "fitting_net": {
            "neuron": [
                120,
                120,
                120
            ],
            "resnet_dt": true,
            "seed": 1,
            "_comment": " that's all"
        },
        "_comment": " that's all"
    },
```

The first block is the model definition, and provides the atom
configurations to be mapped to a set of symmetry invariant features by a
suitable descriptor. Make sure the
order provided here is the same as what is listed in the type.raw input
file. The descriptor used in this example is the se_e2_a descriptor,
which is a Deep Potential-smooth edition that is constructed from both
the angular and radial information of the atomic configurations. What
follows next are the settings of the embedding net that suitably
accomplish the mapping. An important parameter here is the rcut, which
is the short-range cutoff for searching neighbors, while sel gives the
maximum possible neighbors within the defined cutoff radius. This is
followed by the settings of the fitting net, which completes the section
on the model definition. The next two blocks specify how the loss
functions will be optimized through a set of parameters. The first block
specifies the learning rate, while the second block specifies the
initial and limiting prefactors for energies, forces, and the virial in
the loss function.

```
   "learning_rate": {
        "type": "exp",
        "start_lr": 0.005,
        "decay_steps": 10000,
        "_comment": "that's all",
        "stop_lr": 1.752633312441437e-07
    },
    "loss": {
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
        "_comment": " that's all"
    },
```

The last block specifies among other things the number of training steps
and the training and validation data. The specific format in which the
training and validation data should be provided is covered below.

```
    "training": {
        "stop_batch": 2000000,
        "seed": 1,
        "_comment": "that's all",
        "disp_file": "lcurve.out",
        "disp_freq": 100,
        "numb_test": 10,
        "save_freq": 1000,
        "save_ckpt": "model.ckpt",
        "disp_training": true,
        "time_training": true,
        "profiling": false,
        "profiling_file": "timeline.json",
        "training_data": {
            "systems": [
                "/scratch/ch24d003/ActiveLearning_Pt/bulk/Training"
            ],
            "set_prefix": "set",
            "batch_size": "auto"
        },
        "validation_data": {
            "systems": [
                "/scratch/ch24d003/ActiveLearning_Pt/bulk/Training/Testing"
            ],
            "set_prefix": "set",
            "batch_size": "auto"
        }
    }
```
The [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/stable/index.html) is comprehensive and serves as an excellent technical reference

## The Output file 

Among the many output files obtained, the progress of the training can
be monitored by looking at the lcurve.out file, whose contents are shown
below:

```
   #  step   rmse_val    rmse_trn    rmse_e_val  rmse_e_trn    rmse_f_val  rmse_f_trn       lr
   # If there is no available reference data, rmse_*_{val,trn} will print nan
      0      6.60e+00    5.82e+00      7.95e-02    7.83e-02      2.09e-01    1.84e-01    5.0e-03
    100      7.94e-01    6.67e-01      9.84e-02    1.32e-01      2.50e-02    2.08e-02    5.0e-03
    200      6.00e-01    6.00e-01      3.70e-01    2.51e-01      1.65e-02    1.79e-02    5.0e-03
    300      7.12e-01    5.70e-01      1.72e-01    1.03e-01      2.21e-02    1.78e-02    5.0e-03
    400      9.64e-01    9.28e-01      1.93e-01    1.67e-01      3.01e-02    2.91e-02    5.0e-03
    500      4.34e-01    4.49e-01      3.19e-01    1.30e-01      1.11e-02    1.38e-02    5.0e-03
    600      4.31e-01    4.95e-01      1.32e-01    2.89e-01      1.32e-02    1.38e-02    5.0e-03
```

where, the different columns indicate the training step, the total RMSE
of the validation and training, and the RMSE in energy, forces and the
learning rate. You can also plot the number of steps vs the RMS errors
to follow the progress of the training process.

Once training is complete, you can proceed to freeze the model using the
command [`dp freeze -o <name.pb>`] in
the command line. This will generate a deep potential file
[`<name.pb>`] which can be used for
performing DPMD simulations. 

## Sample LAMMPS input for DPMD simulations: 

```
   plugin load     libdeepmd_lmp.so
   units           metal
   boundary        p p p
   atom_style      atomic
   neighbor        2.0 bin
   neigh_modify    every 10 delay 0 check no
   read_data       /scratch/ch24d003/ActiveLearning_Pt/bulk/bulk.lmp
   mass            1 195.084
   
   pair_style deepmd /scratch/ch24d003/ActiveLearning_Pt/DP_models/1/graph1.pb /scratch/ch24d003/ActiveLearning_Pt/DP_models/2/graph2.pb /scratch/ch24d003/ActiveLearning_Pt/DP_models/3/graph3.pb out_file md.out out_freq 100
   pair_coeff      * *
   velocity        all create 300 12345678
   fix             1 all nvt temp 300 300 0.1
   timestep        0.0005
   thermo_style    custom step pe ke etotal temp press vol
   thermo          100
   dump            1 all dcd 100 bulk.dcd
   timer timeout   03:00:00 every 100
   run             200000
   write_restart   lammps.restart
```
one must load the LAMMPS-DeePMD plugin in order to enable the use of the
deepmd [pair_style] option. A suitable file that contains the simulation
cell information and the coordinates of the atoms in the starting
configuration should be furnished. When continuing a simulation, make
sure to read both these and the velocities from the restart file. In
this case, an initial velocity distribution for the atoms should not be
specified. It is important to note the different ensembles and the
corresponding thermostat and/or barostat options available. In this
example, the nvt keyword specifies both the canonical ensemble and the
Nose-Hoover thermostat being used. The [thermo_style] keyword specifies
the quantities that will be printed to the log file, while the dump
keyword specifies how the trajectory will be written. The run keyword
specifies the number of steps for which the simulation will run. It is
generally good practice to break up a long simulation into parts by
writing restart files and starting again by reading them. The
integration timestep is provided by the timestep keyword, which in this
example is 0.0005 ps or 0.5 fs. It should be noted that this number is
dictated by the choice of units specified, which is metal here. Finally,
it is important that the atoms and their masses are listed in the same
order as they are provided when mapped in the DP training. For more details, see the LAMMPS [manual](https://docs.lammps.org/Manual.html)

## The Output File 
The key output files obtained include the log file, which contains
thermodynamic info such as temperature, pressure, and the energies as
described by the [thermo_style] command. The MD trajectory gets printed in
the format and frequency specified by the dump command.

Here, an important output file that is highly relevant during the
training of the DP is described in detail. Typically, we compare the
maximum deviation in the atomic forces over an ensemble of neural
networks that differ only in their initialization. In order to obtain
this, one can specify the different frozen models obtained from the
training in the [pair_style] command. LAMMPS will perform MD with the
first model and provide the deviation between all the models specified
in a readable text file. In the sample input above, this is called
md.out. The contents of md.out look like the following:

```
       step        max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f
       0         1.009449e-02       6.109586e-06       5.154487e-03       7.974324e-04       2.546701e-04       5.464042e-04
       100       1.822111e-02       2.073391e-03       9.891248e-03       1.201322e-01       6.485526e-03       3.477206e-02
       200       1.939164e-02       7.808613e-04       9.880663e-03       1.095691e-01       7.600227e-03       4.101700e-02
       300       1.393546e-02       1.944702e-03       8.996917e-03       1.374295e-01       4.926929e-03       4.221991e-02
       400       1.542865e-02       7.462263e-04       7.273032e-03       9.980609e-02       4.697013e-03       2.334689e-02
       500       2.724647e-02       2.812388e-03       1.427890e-02       1.439207e-01       7.891354e-03       5.124365e-02
       600       2.826217e-02       7.248066e-03       1.595639e-02       2.431786e-01       9.187094e-03       7.896723e-02
```

The first column shows the MD steps, with the next three columns showing
the max, min and average deviation in the system virial between the
different models. The last three columns show the max, min, and average
deviation in the atomic forces between the three models.

Running an MD simulation using one DP potential and comparing the
maximum deviation in the atomic forces over an ensemble of neural
networks that differ only in their initialization begins the Exploration
stage of the Active Learning Process. The second step is using the max,
min, and average deviation in the atomic forces as an indicator for
labeling configurations for DFT calculations which are then used to
update the training dataset. This process is repeated till the average
force deviations fall below $0.05 eV/$Å.

## Active Learning Overview 

The active learning protocol follows an iterative Train-Explore-Label
cycle.

1.  The Exploration Phase: Exploration is the systematic sampling of
    configuration space using the current version of the Deep Potential
    (DP) model. In each iteration, an ensemble of models (Three in this tutorial) runs DPMD simulations to explore the system's potential energy
    surface.

    1.  Model Deviation as an Indicator: To identify regions where the
        model is \"uncertain,\" we monitor the maximum deviation in
        atomic forces between the models in the ensemble. High deviation
        typically indicates low prediction accuracy.

    2.  Data Selection: Standard protocols usually select configurations
        within a specific uncertainty window (e.g., 0.1 to 0.8 eV/Å).
        This range ensures the model learns from without being corrupted
        by non-physical configurations.

2.  Labeling: In this step, we generate \"ground truth\" data for the
    configurations selected during Exploration. DFT is used to calculate the
    exact energies and forces for these structures. These newly labeled
    data points are then merged into the existing training dataset.

3.  Training: The training phase evolves alongside the dataset, ensuring the PES is represented both faithfully and efficiently. Once training is complete, the models are
    \"frozen\" and passed back to the Exploration phase for the next
    iteration.
