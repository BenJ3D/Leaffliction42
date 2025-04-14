all: venv install

venv:
	python3 -m venv .venv

install: venv
	.venv/bin/pip install -r requirements.txt

# Règle pour rendre le script exécutable et créer l'archive
zip:
	chmod +x create_archive.sh
	./create_archive.sh

clean:
	echo "Cleaning"
	rm -rf model train
	rm -f predict_results.txt


fclean: clean
	rm -rf .venv
	rm -rf db.zip

re: fclean all

.PHONY: all venv install fclean archive