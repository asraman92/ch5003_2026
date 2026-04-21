
# Hands on Session - Day 2 - $\textbf{21st}$ April $\textbf{2026}$ {#hands-on-session---day-2---textbf22nd-april-textbf2026 .unnumbered}

## Aim {#aim-1 .unnumbered}

The goal of this step is to use the DFT energies and forces generated in the previous tutorial to train a machine-learning model with DeePMD-kit.Additionally, we also demonstrate the active learning protocol used to build a robust training dataset for a converged Deep Potential (DP) model. Basedon the principles established by [\"Zhang et.al., Phys. Rev. Mater., 3, 023804\"](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.3.023804), we will iteratively refine the initial platinum model.

Please note that due to time constraints, this session serves as a functional demonstration of the workflow; the resulting model is a baseline and may require further refinement before use in high-level production simulations.


## Objectives {#objectives-1 .unnumbered}

The objectives of this tutorial session are:

-   Train a DeePMD model to accurately represent the potential energy
    surface of platinum.

-   Understanding the inputs and ouputs of the training process.

-   Running molecular dynamics simulations using the trained model.

-   Quantify uncertainty by using an ensemble of models to estimate
    errors in predicted interatomic forces and preparing those configurations for
    labeling.


## Anatomy of a typical input file for training {#anatomy-of-a-typical-input-file-for-training .unnumbered}

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

## The Output file {#the-output-file .unnumbered}

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
command [`dp freeze -o <name.pb>`]{style="background-color: gray!20"} in
the command line. This will generate a deep potential file
[`<name.pb>`]{style="background-color: gray!20"} which can be used for
performing DPMD simulations.

## Sample LAMMPS input for DPMD simulations: {#sample-lammps-input-for-dpmd-simulations .unnumbered}

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
deepmd pair_style option. A suitable file that contains the simulation
cell information and the coordinates of the atoms in the starting
configuration should be furnished. When continuing a simulation, make
sure to read both these and the velocities from the restart file. In
this case, an initial velocity distribution for the atoms should not be
specified. It is important to note the different ensembles and the
corresponding thermostat and/or barostat options available. In this
example, the nvt keyword specifies both the canonical ensemble and the
Nose-Hoover thermostat being used. The thermo_style keyword specifies
the quantities that will be printed to the log file, while the dump
keyword specifies how the trajectory will be written. The run keyword
specifies the number of steps for which the simulation will run. It is
generally good practice to break up a long simulation into parts by
writing restart files and starting again by reading them. The
integration timestep is provided by the timestep keyword, which in this
example is 0.0005 ps or 0.5 fs. It should be noted that this number is
dictated by the choice of units specified, which is metal here. Finally,
it is important that the atoms and their masses are listed in the same
order as they are provided when mapped in the DP training.

## The Output File {#the-output-file-1 .unnumbered}

The key output files obtained include the log file, which contains
thermodynamic info such as temperature, pressure, and the energies as
described by the thermo_style command. The MD trajectory gets printed in
the format and frequency specified by the dump command.

Here, an important output file that is highly relevant during the
training of the DP is described in detail. Typically, we compare the
maximum deviation in the atomic forces over an ensemble of neural
networks that differ only in their initialization. In order to obtain
this, one can specify the different frozen models obtained from the
training in the pair_style command. LAMMPS will perform MD with the
first model and provide the deviation between all the models specified
in a readable text file. In the sample input above, this is called
md.out. The contents of md.out look like the following:

::: {.mycodebox}
       step        max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f
       0         1.009449e-02       6.109586e-06       5.154487e-03       7.974324e-04       2.546701e-04       5.464042e-04
       100       1.822111e-02       2.073391e-03       9.891248e-03       1.201322e-01       6.485526e-03       3.477206e-02
       200       1.939164e-02       7.808613e-04       9.880663e-03       1.095691e-01       7.600227e-03       4.101700e-02
       300       1.393546e-02       1.944702e-03       8.996917e-03       1.374295e-01       4.926929e-03       4.221991e-02
       400       1.542865e-02       7.462263e-04       7.273032e-03       9.980609e-02       4.697013e-03       2.334689e-02
       500       2.724647e-02       2.812388e-03       1.427890e-02       1.439207e-01       7.891354e-03       5.124365e-02
       600       2.826217e-02       7.248066e-03       1.595639e-02       2.431786e-01       9.187094e-03       7.896723e-02
:::

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

## Active Learning Overview {#active-learning-overview .unnumbered}

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
