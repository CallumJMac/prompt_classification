import random

from typing import List, Tuple
from name_of_module import very_good

def generate_x_y() -> Tuple[List[float], List[float]]:
    """Example where you import code from
    the main package and then make a demo."""
    print(very_good.hello_world())
    return (
        [random.random() for _ in range(10)],
        [random.random() for _ in range(10)],
    )