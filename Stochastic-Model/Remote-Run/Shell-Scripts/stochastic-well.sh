#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=10gb
#PBS -N Stochastic-Well-Simulation
#PBS -J 1-4

module load anaconda3/personal
source activate personalpy3
date

python $PBS_O_WORKDIR/stochastic-well.py $PBS_ARRAY_INDEX
