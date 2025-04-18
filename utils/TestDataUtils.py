import json
from pathlib import Path


def load_data_sets(filename):
    root_path = Path(__file__).parent.absolute()
    data_source_path = root_path / "sourcedata" / filename
    with open(data_source_path) as file:
        return json.load(file)
