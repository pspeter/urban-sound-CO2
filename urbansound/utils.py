import os
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> None:
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
