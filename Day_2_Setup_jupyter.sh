
############################
# XX == YOUR LAPTOP-NUMBER #
############################

############################
# PORTNUMBER 10000 - XX    #
# i.e.   5 = 10000 - 5 ->  #
# PORTNUMBER = 9995        #
############################

## Start your Terminal

# connect to NVDIA server
ssh mdc-XX@farm.parallel-computing.pro 

# generate config file
jupyter notebook --generate-config

# set password
jupyter notebook password # set password to tensorflow1

# start server
nohup jupyter notebook --no-browser --port PORTNUMBER 

# connect to server notebook LOCAL
ssh -nNT -L 9999:localhost:PORTNUMBER mdc-XX@farm.parallel-computing.pro # enter password

## start from Browser

localhost:PORTNUMBER # enter password

# finished, use jupyter notebook on NVIDIA GPU server
