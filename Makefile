check_dirs := "."

quality:
	black -l 119 --check --preview  $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 --max-line-length 119 $(check_dirs)

style:
	black --preview -l 119 $(check_dirs)
	isort $(check_dirs)