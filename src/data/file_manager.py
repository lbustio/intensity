import os
import pandas as pd
import json
import pickle
import logging

class FileManager:
    """
    A class to manage file operations, including loading and saving files in various formats 
    (CSV, XLSX, JSON, Pickle), as well as verifying and removing files.
    """

    def __init__(self, logger=None):
        """
        Initializes the FileManager instance.

        Parameters:
            logger (logging.Logger, optional): A logger instance for logging messages.
        """
        self.logger = logger if logger else logging.getLogger(__name__)

    def load(self, filepath):
        """
        Loads a file based on its extension and returns the data.
        
        The method identifies the file type based on its extension and reads the file accordingly.
        
        Parameters:
            filepath (str): The path to the file to load.
        
        Returns:
            object: The data read from the file (e.g., pandas DataFrame, dict, object).
        
        Raises:
            ValueError: If the file extension is unsupported.
            FileNotFoundError: If the file does not exist.
        """
        if not self.verify_file(filepath):
            raise FileNotFoundError(f"File not found: '{filepath}'")
        
        file_extension = os.path.splitext(filepath)[1].lower()

        try:
            if file_extension == '.csv':
                return pd.read_csv(filepath)
            elif file_extension == '.xlsx':
                return pd.read_excel(filepath)
            elif file_extension == '.json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            elif file_extension in ['.pkl', '.pickle']:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
        except Exception as e:
            self.logger.error(f"An error occurred while loading the file '{filepath}': {e}")
            raise
        

    def save(self, data, filepath):
        """
        Saves data to a file based on its extension.
        
        The method identifies the file type based on its extension and writes the data to the file.
        
        Parameters:
            data (object): The data to save (e.g., pandas DataFrame, dict, object).
            filepath (str): The path to the file to save.
        
        Raises:
            ValueError: If the file extension is unsupported.
        """
        file_extension = os.path.splitext(filepath)[1].lower()

        try:
            if file_extension == '.csv':
                data.to_csv(filepath, index=False)
            elif file_extension == '.xlsx':
                data.to_excel(filepath, index=False)
            elif file_extension == '.json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4)
            elif file_extension in ['.pkl', '.pickle']:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
            self.logger.info(f"File '{filepath}' saved successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while saving the file '{filepath}': {e}")
            raise
        

    def verify_file(self, filepath):
        """
        Verifies if a file exists at the given path.
        
        This method checks if the specified file exists in the filesystem.
        
        Parameters:
            filepath (str): The path to the file to verify.
        
        Returns:
            bool: True if the file exists, False otherwise.
        """
        exists = os.path.exists(filepath)
        if exists:
            self.logger.info(f"File '{filepath}' exists.")
        else:
            self.logger.warning(f"File '{filepath}' does not exist.")
        return exists
    

    def remove_file(self, filepath):
        """
        Removes a file at the given path.
        
        This method attempts to delete the file if it exists.
        
        Parameters:
            filepath (str): The path to the file to remove.
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if self.verify_file(filepath):
            try:
                os.remove(filepath)
                self.logger.info(f"File '{filepath}' removed successfully.")
            except Exception as e:
                self.logger.error(f"An error occurred while removing the file '{filepath}': {e}")
                raise
        else:
            raise FileNotFoundError(f"File '{filepath}' does not exist, cannot remove.")
