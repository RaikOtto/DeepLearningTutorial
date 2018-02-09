# connect to server notebook LOCAL
ssh -nNT -L 9999:localhost:8888 mdc-XX@farm.parallel-computing.pro # enter password

# connect to notebook GO TO BROWSER
localhost:9999 # enter password

# finished, use jupyter notebook on NVIDIA GPU server
