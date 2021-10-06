# Make directory to copy model in to, submit job from there
NOW=$(date +"%Y-%m-%d-%T")
run_directory="$EPHEMERAL/Stochastic-Well/$NOW"
mkdir -p $run_directory
cp -r $HOME/Instantons/Rotated-2D-Well/Stochastic-Model/Remote-Run/ $run_directory
cd $run_directory/Remote-Run
cp $run_directory/Remote-Run/Shell-Scripts/stochastic-well.sh .
qsub stochastic-well.sh
