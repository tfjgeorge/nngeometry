quality:
	black --check nngeometry tests
	flake8 nngeometry tests
	isort --check-only nngeometry tests
