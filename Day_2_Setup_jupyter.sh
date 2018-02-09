# connect to server XX is your specific login
ssh mdc-XX@farm.parallel-computing.pro
# enter password

# install jupyter
python -m pip install jupyter --user # perhaps already installed

# set the password
~/.local/bin/jupyter notebook password # tensorflow1

# start the server
nohup ~/.local/bin/jupyter notebook --no-browser
### LEAVE THE SERVER AND START LOCAL TERMINAL

# connect to server notebook LOCAL
ssh -nNT -L 9999:localhost:8888 mdc-XX@farm.parallel-computing.pro # enter password

# connect to notebook GO TO BROWSER
localhost:9999 # enter password

# finished, use jupyter notebook on NVIDIA GPU server
