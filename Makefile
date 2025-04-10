ifeq ($(OS),Windows_NT)
	PYTHON=python
	PIP=.venv/Scripts/pip.exe
else
	PYTHON=python3
	PIP=.venv/bin/pip
endif

all: venv install

venv:
	$(PYTHON) -m venv .venv

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

clean:
	echo "Cleaning"

fclean: clean
	rm -rf .venv

re: fclean all

.PHONY: all venv install clean fclean re
