#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=96gb
#PBS -N Stochastic-Well-Simulation
#PBS -J 1-20

module load anaconda3/personal
source activate personalpy3
date

python $PBS_O_WORKDIR/stochastic-well.py $PBS_ARRAY_INDEX
