import os
import logging
import shutil

class FolderManager:
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)

    def create_base_folders(self, folders):
        """
        Create the necessary base folders for the project if they don't exist.
        :param folders: List of folder paths to create
        """
        try:
            for folder in folders:
                self.logger.info(f"Attempting to create folder '{folder}'...")
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    self.logger.info(f"Folder '{folder}' created successfully.")
                else:
                    self.logger.warning(f"Folder '{folder}' already exists.")
            self.logger.info("All required folders have been created or verified successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while creating folders: {e}")
            raise

    def create_folder(self, parent_folder, folder_name):
        """
        Create a specific folder inside the given parent folder.
        :param parent_folder: The path of the parent folder
        :param folder_name: The name of the folder to create
        """
        try:
            folder_path = os.path.join(parent_folder, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                self.logger.info(f"Folder '{folder_path}' created successfully.")
            else:
                self.logger.warning(f"Folder '{folder_path}' already exists.")
        except Exception as e:
            self.logger.error(f"An error occurred while creating the folder '{folder_name}': {e}")
            raise

    def verify_folder(self, folder_path):
        """
        Verify if a folder exists.
        :param folder_path: Path to the folder to verify
        :return: True if the folder exists, False otherwise
        """
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            self.logger.info(f"Folder '{folder_path}' exists.")
            return True
        else:
            self.logger.warning(f"Folder '{folder_path}' does not exist.")
            return False

    def remove_folder(self, folder_path):
        """
        Remove the specified folder.
        :param folder_path: Path to the folder to remove
        """
        try:
            if self.verify_folder(folder_path):
                shutil.rmtree(folder_path)
                self.logger.info(f"Folder '{folder_path}' removed successfully.")
            else:
                self.logger.warning(f"Cannot remove folder '{folder_path}' because it does not exist.")
        except Exception as e:
            self.logger.error(f"An error occurred while removing the folder '{folder_path}': {e}")
            raise