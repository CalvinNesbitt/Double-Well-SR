#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=2:mem=20gb
#PBS -N Double-Well-Timings
#PBS -J 1-4

module load anaconda3/personal
source activate personalpy3
date

python $PBS_O_WORKDIR/transition_time_test.py $PBS_ARRAY_INDEX

