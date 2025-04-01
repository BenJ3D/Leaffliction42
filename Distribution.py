import os
import matplotlib.pyplot as plt
import argparse

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
    
    # Définir une palette de couleurs pour les catégories
    import matplotlib.colors as mcolors
    
    # Utiliser des couleurs prédéfinies, ou en créer plus si nécessaire
    colors = list(mcolors.TABLEAU_COLORS.values())
    # S'assurer d'avoir assez de couleurs
    while len(colors) < len(labels):
        colors.extend(colors)
    colors = colors[:len(labels)]  # Limiter au nombre nécessaire
    
    # Créer un dictionnaire pour mapper les catégories aux couleurs
    category_colors = dict(zip(labels, colors))

    # Création d'une figure avec deux sous-graphiques côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Diagramme circulaire (camembert) dans le premier sous-graphe
    ax1.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax1.set_title('Répartition des images')
    ax1.axis('equal')
    
    # Diagramme à barres dans le deuxième sous-graphe
    bars = ax2.bar(labels, counts, color=[category_colors[label] for label in labels])
    ax2.set_title('Distribution des images par catégorie')
    ax2.set_xlabel('Catégories')
    ax2.set_ylabel("Nombre d'images")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45)
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse la répartition des images dans un répertoire.')
    parser.add_argument('directory', type=str, help='Le chemin vers le répertoire à analyser')
    args = parser.parse_args()
    analyze_dataset(args.directory)
