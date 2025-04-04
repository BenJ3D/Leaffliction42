import os
import argparse
import sys
import json

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Erreur: {message}\n")
        sys.exit(2)

def extract_classes(directory):
    """Extract class names from subdirectories."""
    classes = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and not item == "output":
            classes.append(item)
    return sorted(classes)

def save_classes_to_json(classes):
    """Save the class names to a JSON file in the current working directory."""
    classes_json = {"classes": classes}
    json_path = os.path.join(os.getcwd(), "classes.json")
    with open(json_path, 'w') as f:
        json.dump(classes_json, f, indent=4)
    return json_path

def main():
    parser = CustomArgumentParser(
        description="Programme pour lancer le train sur un modèle de détection d'objets."
    )
    parser.add_argument(
        "src",
        help="Chemin du dossier récursif contenant toutes les images d'entrainement"
    )
    args = parser.parse_args()

    if not os.path.exists(args.src):
        parser.error(f"Le dossier source '{args.src}' n'existe pas.")
    
    # Extract classes from subdirectories
    classes = extract_classes(args.src)
    if not classes:
        print("Aucune classe (sous-dossier) trouvée dans le répertoire source.")
    else:
        json_path = save_classes_to_json(classes)
        print(f"Classes disponibles ({len(classes)}):")
        for cls in classes:
            print(f"- {cls}")
        print(f"Liste des classes sauvegardée dans: {json_path}")

    print(f"Training sur les images du dossier: {args.src}...")

if __name__ == "__main__":
    main()