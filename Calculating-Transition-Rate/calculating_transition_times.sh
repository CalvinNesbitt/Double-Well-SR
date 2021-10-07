#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=62gb
#PBS -N Transition-Times
#PBS -J 1-20

module load anaconda3/personal
source activate personalpy3
date

python $PBS_O_WORKDIR/calculating_transition_times.py $PBS_ARRAY_INDEX
