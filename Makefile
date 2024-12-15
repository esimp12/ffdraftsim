install: requirements.txt
	pip-sync requirements.txt

requirements.txt: requirements.in
	pip-compile -o requirements.txt requirements.in
