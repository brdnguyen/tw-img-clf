# Preprequisite
* Python 3.7
* pipenv 
* Data in `input` folder


# Getting started
1. Install `pipenv` for environment managemment if you haven't got one:

```
pip install pipenv
```
2. Install dependency
```
pipenv install
```

# Run tests
```
pipenv run nosetests
```

# Run full model and
```
pipenv run python src/run_all.py
```

# Notebook
Access notebooks in /notebook

## Note: Setup local dev environment using Docker

To lazily get a docker with all the environment setup, try
```
docker pull kaggle/python
```


