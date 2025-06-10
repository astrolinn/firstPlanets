import importlib
import os
import sys

def load_input_file(path):
    """
    Dynamically load inputFile.py from the given path.

    Args:
        path (str): Path to the simulation directory containing inputFile.py.

    Returns:
        module: The imported inputFile module.

    Raises:
        FileNotFoundError: If inputFile.py is not found in the given path.
        AttributeError: If the required variable is missing in inputFile.py.
    """
    input_file_dir = os.path.abspath(path)
    input_file_name = "inputFile"

    # Add the directory to the Python search path
    sys.path.insert(0, input_file_dir)

    try:
        # Import the inputFile module dynamically
        return importlib.import_module(input_file_name)
    except ModuleNotFoundError:
        raise FileNotFoundError(f"Error: Could not find {input_file_name}.py in {input_file_dir}")
    finally:
        # Clean up the Python path
        sys.path.pop(0)