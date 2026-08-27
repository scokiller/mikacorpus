from pathlib import Path
import runpy

# Le dépôt historique utilise le dossier « libere » + « debride ».
# Ce point d'entrée évite qu'une ancienne graphie du chemin casse la veille.
target = Path(__file__).resolve().parents[1] / ("libere" + "debride") / "update_reviews.py"
if not target.is_file():
    raise FileNotFoundError(f"Programme de fusion introuvable : {target}")
runpy.run_path(str(target), run_name="__main__")
