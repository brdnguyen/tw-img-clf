# Setup local dev environment
## Setup Kaggle Kernel Docker
```
docker-machine create -d virtualbox --virtualbox-disk-size "50000" --virtualbox-cpu-count "4" --virtualbox-memory "8092" docker2

docker-machine start docker2

eval $(docker-machine env docker2)
```

## Setup aliases to run

To run python and jupyter notebook in the docker container

Save these aliases in .bash_profile
```
kpython()
{
    docker run -v $PWD:/tmp/working -w=/tmp/working --rm -it kaggle/python python "$@"
}

ikpython()
{
    docker run -v $PWD:/tmp/working -w=/tmp/working --rm -it kaggle/python ipython
}


kjupyter() {
    docker run -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it kaggle/python bash -c "pip install jupyter_contrib_nbextensions; pip install jupyter_nbextensions_configurator; jupyter contrib nbextension install --user; jupyter notebook --notebook-dir=/tmp/working --ip='*' --port=8888 --no-browser --allow-root"
}
```
then `source ~/.bash_profile`

then in command line: `kjupyter`

then access Notebook in `http://localhost:8888`

Reference: https://medium.com/@zhang_yang/setup-docker-for-kaggle-b34d04705756
