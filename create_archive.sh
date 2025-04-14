# Script pour archiver les dossiers train et model et générer une signature SHA1

# Vérifier si les dossiers existent
if [ ! -d "train" ] || [ ! -d "model" ]; then
    echo "Erreur: Les dossiers train et/ou model n'existent pas."
    exit 1
fi

# Créer l'archive zip
echo "Création de l'archive db.zip..."
zip -r db.zip train model images

# Générer la signature SHA1
echo "Génération de la signature SHA1..."
sha1sum db.zip > signature.txt

echo "Terminé! L'archive db.zip a été créée et la signature SHA1 a été enregistrée dans signature.txt."