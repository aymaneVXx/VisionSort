import sys
from pathlib import Path

# Ajoute le répertoire racine du projet au sys.path pour l'exécution des tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
