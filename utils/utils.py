import pickle
import yaml

def save_object(file_path: str, obj: object) -> None:
    """Saves an object to a file using pickle."""
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
        
def load_object(file_path: str) -> object:
    """Loads an object from a file using pickle."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def read_yaml(file_path: str) -> dict:
    """Reads a YAML file and returns its content as a dictionary."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)