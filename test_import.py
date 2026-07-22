import sys
from pathlib import Path
print("Adding current directory:", Path.cwd())
sys.path.insert(0, str(Path.cwd()))
print("sys.path:", sys.path[:3])
try:
    import visionsort
    print("Imported visionsort successfully")
    from visionsort.core.config import load_config
    print("loaded config module successfully")
    cfg = load_config()
    print("load_config ok:", cfg is not None)
    from visionsort.core.enums import CameraStatus, CameraRole, CommandType, CommandStatus
    print("loaded enums")
    print("All core imports passed!")
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    print("Traceback:", traceback.format_exc())
