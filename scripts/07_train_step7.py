from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("10_train_step10_quadls.py")), run_name="__main__")
