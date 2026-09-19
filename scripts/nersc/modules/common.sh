#!/usr/bin/env bash

# Common Perlmutter software stack. Source this after selecting cpu or gpu.
module load conda
module load cray-mpich
module load PrgEnv-gnu
module load cray-hdf5-parallel

# NERSC guidance for HDF5 writes outside $SCRATCH (includes CFS runtime roots).
# Disabled locking is not a multi-writer solution; require exclusive ownership.
export HDF5_USE_FILE_LOCKING=FALSE
