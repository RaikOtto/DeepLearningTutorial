USER=$1

# make sure we are on a working environment
ssh $USER@farm.parallel-computing.pro 'python-tensorflow -m pip install --user --upgrade setuptools pip; python-tensorflow -m pip install --user backports.shutil_get_terminal_size; python-tensorflow -m pip install --user --upgrade --force-reinstall ipython; python-tensorflow -m pip install --user --upgrade --force-reinstall tensorflow; python-tensorflow -m pip install --user --upgrade --force-reinstall tensorflow-gpu; ./.local/bin/jupyter notebook --generate-config; wget -O .jupyter/jupyter_notebook_config.py https://raw.githubusercontent.com/RaikOtto/DeepLearningTutorial/master/jupyter_notebook_config.py'

RAND_PORT=$(python -S -c "import random; print random.randrange(2000,63000)")
ssh -L 8000:localhost:$RAND_PORT $USER@farm.parallel-computing.pro './.local/bin/jupyter notebook --port '$RAND_PORT


