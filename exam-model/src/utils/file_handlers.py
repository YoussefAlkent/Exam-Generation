def read_json_file(file_path: str):
    import json
    import os

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        data = json.load(file)

    return data