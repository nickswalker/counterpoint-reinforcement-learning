#! env sh
rsync -av --exclude ".git" --exclude "results" --exclude ".idea" . nwalker@linux.cs.utexas.edu:~/Documents/counterpoint-reinforcement-learning