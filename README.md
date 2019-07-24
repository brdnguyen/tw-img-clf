# Preprequisite
* Python 3.7
* Data in `input` folder as-is as in Kaggle kernel


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

## Note: Setup local dev environment using Kaggle's Docker
### Setup Kaggle Kernel Docker

To lazily get a docker with all the environment setup that is similar to the one found in Kaggle Kernel, try
```
docker pull kaggle/python
```


Follow the full instructions here http://blog.kaggle.com/2016/02/05/how-to-get-started-with-data-science-in-containers/
