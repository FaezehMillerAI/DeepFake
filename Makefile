.PHONY: setup test lint figures clean

setup:
	pip install -r requirements.txt

test:
	pytest -q

figures:
	python -m src.eval.make_figures --runs runs/ --out figures/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache
