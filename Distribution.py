import os
import matplotlib.pyplot as plt


def analyze_dataset(directory):
    # Dictionnaire pour stocker le nombre d'images par catégorie
    categories = {}

    # On liste les sous-dossiers dans le répertoire 'directory'
    for category in os.listdir(directory):
        path = os.path.join(directory, category)
        if os.path.isdir(path):
            # On récupère tous les fichiers d'image (ici en considérant jpg, jpeg et png)
            images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            categories[category] = len(images)

    # Affichage des résultats dans la console
    print("Répartition des images par catégorie :")
    for cat, count in categories.items():
        print(f"{cat} : {count} images")

    # Préparation des données pour les graphiques
    labels = list(categories.keys())
    counts = list(categories.values())

    # Diagramme à barres
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.title('Distribution des images par catégorie')
    plt.xlabel('Catégories')
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()

    # Diagramme circulaire (camembert)
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Répartition des images')
    plt.axis('equal')
    plt.show()
