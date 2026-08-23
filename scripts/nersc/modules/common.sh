#!/usr/bin/env bash

# Common Perlmutter software stack. Source this after selecting cpu or gpu.
module load conda
module load cray-mpich
module load PrgEnv-gnu
module load cray-hdf5-parallel
