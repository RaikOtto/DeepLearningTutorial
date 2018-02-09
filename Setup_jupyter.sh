# connect to server
ssh mdc-XX@farm.parallel-computing.pro
# enter password

# install jupyter
python -m pip install jupyter --user # perhaps already installed

# set the password
~/.local/bin/jupyter notebook password # tensorflow1

# start the server
nohup ~/.local/bin/jupyter notebook --no-browser
### LEAVE THE SERVER AND START LOCAL TERMINAL

