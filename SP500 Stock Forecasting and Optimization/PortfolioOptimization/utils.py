from pathlib import (Path)

def checkFolder(path: str) -> None:
    """
    # Description
        -> This function ensures that the path of nested folders provided exists.
    -----------------------------------------------------------------------------
    := param: path - Path of the File.
    := return: None, since we are only making sure a certain path exists. 
    """

    # Update the path according to the formatting being used in this project
    path = "/".join(path.split("/")[:-1])

    # Define the path as a Path object
    folderPath = Path(path)
    
    # Ensure it exists
    folderPath.mkdir(parents=True, exist_ok=True)