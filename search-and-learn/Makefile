.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := src scripts tests

style:
	ruff check --select I --fix $(check_dirs) setup.py 
	ruff format $(check_dirs) setup.py 

quality:
	ruff check --select I $(check_dirs) setup.py