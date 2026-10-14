### load_input.py loads path of "dossier final" from path.yaml
import yaml
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent 

# path to path.yaml
path_yaml = base_dir / "path.yaml"

def load_input():
    with open(path_yaml, 'r') as f:
        path = yaml.safe_load(f)
    return path["dossier_final"]
