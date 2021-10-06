# Make directory to copy model in to, submit job from there
NOW=$(date +"%Y-%m-%d-%T")
run_directory="$EPHEMERAL/Double-Well-Transitions/$NOW"
mkdir -p $run_directory
cp -r $HOME/Double-Well-SR/Finding-Transitions/ $run_directory
cd $run_directory/Finding-Transitions/
cp $run_directory/Finding-Transitions/Shell-Scripts/transitions-find.sh .
qsub transitions-find.sh
