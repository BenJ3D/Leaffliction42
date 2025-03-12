all: venv install

venv:
	python3 -m venv .venv

install: venv
	.venv/bin/pip install -r requirements.txt


clean:
	echo "Cleaning"

fclean: clean
	rm -rf .venv

re: fclean all

.PHONY: all venv install fclean