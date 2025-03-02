# ================================
# IMPORTS
# ================================
import platform  # For retrieving platform-specific info
import psutil  # For CPU and RAM details
import GPUtil  # For GPU information (if available)
import torch  # For CUDA GPU info (if available)
import pandas as pd  # For handling and displaying data in DataFrame format
import time  # Library for time-related functions
from prettytable import PrettyTable  # For displaying information in a nice table format
import os  # For interacting with the file system

# Custom utilities for logging, package installation, and data management
from src.utils.logger import LoggerSingleton  # Custom LoggerSingleton class for logging messages
from src.utils.package_installer import PackageInstaller  # Custom PackageInstaller class for installing packages
from src.data.folder_manager import FolderManager  # Importing FolderManager for folder operations
from src.data.file_manager import FileManager  # Importing FileManager for file operations
from src.data.data_respresentation import DataRepresentation  # Importing DataRepresentation for text embeddings
from src.utils.constants import (
    DATA_FOLDER,
    PRINCIPLE_FOLDER,
    PROCESSED_DATA_FOLDER, 
    POPS,
    POP
)  # Importing constants for data folder paths and POPS columns

# ================================
# GLOBAL VARIABLES (SINGLETON INSTANCES)
# ================================
logger = LoggerSingleton().get_logger()  # Get the logger instance (global)
folder_manager = FolderManager(logger=logger)  # Get the FolderManager instance (global)
file_manager = FileManager(logger=logger)  # Get the FileManager instance (global)

# ================================
# SPLASH SCREEN FUNCTION
# ================================
def splash():
    """
    Displays a welcome screen for the research project on Persuasion Intensity in Phishing Messages.
    
    This function prints out a welcome message with the project details, including the title,
    version, and description of the project. It also simulates a loading process for the program.
    """
    print("\n---------------------------------------------------------------")
    print("  Research Project: Persuasion Intensity in Phishing Messages")
    print("  Developed by: Lázaro Bustio-Martínez and Contributors")
    print("  Version: 1.0.0")
    print("---------------------------------------------------------------")
    print("  This project analyzes the persuasion intensity in phishing messages")
    print("  using semi-supervised learning techniques.")
    print("---------------------------------------------------------------")
    print("🚀 Starting the program...\n")
    time.sleep(1)  # Simulate loading time

# ================================
# INITIALIZE FUNCTION
# ================================
def initialize():
    """
    Initializes the program setup by installing required packages, gathering system info,
    and creating necessary base folders for the project.
    
    This function installs the required Python packages (if needed), logs system information
    like CPU, RAM, and GPU details, and creates the necessary folders for storing data and models.
    """
    logger.info("Initializing program...")  # Log the initialization start
    
    # Initialize and create necessary folders using FolderManager (passing the logger)
    base_folders = [DATA_FOLDER, PROCESSED_DATA_FOLDER, PRINCIPLE_FOLDER]  # Add more folder paths as needed
    folder_manager.create_base_folders(base_folders)  # Create base folders using FolderManager
    
    # Define the flag to install packages; set to True if packages should be installed
    INSTALL_PACKAGES = False  # Change this to True to install the required packages
    # Create an instance of the PackageInstaller, passing the install flag and logger
    installer = PackageInstaller(install_flag=INSTALL_PACKAGES, logger=logger)
    installer.install_packages()  # Call the method to install packages if needed
    
    # Get system information using the get_system_info() function
    logger.info("")
    system_info = get_system_info()  # Call the function to get system information
    logger.info("\nSystem Information:\n" + system_info)  # Log the system information

    

# ================================
# GET HARDWARE INFO FUNCTION
# ================================
def get_system_info():
    """
    Retrieves detailed system information such as Python version, platform, processor details,
    number of CPUs, total RAM, and GPU information (if available).
    
    This function uses libraries like `psutil`, `GPUtil`, and `torch` to gather the system 
    specifications and returns them in a nicely formatted table for logging.
    
    Returns:
        str: A formatted string containing the system information, including Python version, 
             platform, processor, CPU, RAM, and GPU details.
    """
    # Initialize PrettyTable object to store and format the system info
    system_info_table = PrettyTable()
    system_info_table.field_names = ["Info", "Details"]

    # Add CPU and RAM information
    system_info_table.add_row(["Python version", platform.python_version()])
    system_info_table.add_row(["Platform", f"{platform.system()} {platform.release()}"])
    system_info_table.add_row(["Processor", platform.processor()])
    system_info_table.add_row(["Architecture", platform.architecture()[0]])
    system_info_table.add_row(["Number of CPUs", psutil.cpu_count(logical=True)])
    system_info_table.add_row(["Total RAM", f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB"])

    # Retrieve GPU details using GPUtil (GPUs detected by the system)
    gpus = GPUtil.getGPUs()
    if gpus:
        for gpu in gpus:
            system_info_table.add_row([f"GPU: {gpu.name}", f"Memory Total: {gpu.memoryTotal} MB"])
            system_info_table.add_row(["GPU Memory Free", f"{gpu.memoryFree} MB"])
            system_info_table.add_row(["GPU Memory Used", f"{gpu.memoryUsed} MB"])
            system_info_table.add_row(["GPU Utilization", f"{gpu.load * 100:.2f}%"])
    else:
        # If no GPU is detected, add a placeholder entry
        system_info_table.add_row(["GPU", "No GPU detected with GPUtil."])

    # Retrieve GPU details using PyTorch (alternative method for CUDA GPUs)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            system_info_table.add_row([f"CUDA GPU {i}: {torch.cuda.get_device_name(i)}", 
                                      f"GPU Capability: {torch.cuda.get_device_capability(i)}"])
            system_info_table.add_row(["Total Memory", f"{torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB"])
    else:
        # If no CUDA-compatible GPU is found, add a placeholder entry
        system_info_table.add_row(["CUDA GPU", "No GPU detected with PyTorch."])

    # Return the formatted table as a string
    return str(system_info_table)


def data_gathering(data_name):
    """
    Gathers data by reading a CSV file containing phishing email data, filtering and 
    checking for necessary columns, and returning the filtered data.
    
    This function checks if the CSV file exists, reads it into a DataFrame, and verifies
    that all required columns are present. If the data contains phishing emails, it filters
    and returns only those entries.

    Parameters:
        data_name (str): The name of the CSV file to read from the processed data folder.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the necessary columns and phishing data.
    
    Raises:
        FileNotFoundError: If the specified CSV file is not found in the expected folder.
        ValueError: If any required columns are missing in the dataset.
    """
    logger.info(f"Reading data from the CSV file '{data_name}'...")   
     
    path = os.path.join(PROCESSED_DATA_FOLDER, data_name)

    # Validate if the CSV file exists at the given path
    if os.path.exists(path):
        try:
            # Use the read_csv method from DataHandler to load the CSV
            data_df = file_manager.load(path)
        except Exception as e:
            # Log an error if reading the CSV file fails
            raise

        # List of expected columns in the dataset
        expected_columns = ["id", "path", "hash", "subject", "txt"] + POPS + ["class", "label"]

        # Check for missing columns in the loaded dataset
        missing_columns = [col for col in expected_columns if col not in data_df.columns]

        # If any columns are missing, log an error and raise an exception
        if missing_columns:
            logger.error(f"Missing expected columns: '{missing_columns}'")
            raise ValueError(f"Missing columns: '{missing_columns}'")

        # Filter the dataset to keep only phishing emails (where 'class' is 1)
        data_phish = data_df[data_df["class"] == 1]

        # Check if there is any phishing data in the filtered DataFrame
        if data_phish.empty:
            logger.warning("No phishing data found in the dataset.")

        # Select the necessary columns for further processing
        data = data_phish[expected_columns]
        
        return data
            
    else:
        # Log an error if the CSV file is not found in the expected location
        logger.error(f"The CSV file '{data_name}' does not exist in the folder '{PROCESSED_DATA_FOLDER}'. Please check the file name and location.")
        raise FileNotFoundError(f"File not found: '{data_name}'")
    
    
def create_embeddings(data, pop_column):
    """
    Creates embeddings for a specific persuasion principle (POP) from the data.
    
    This function uses a class (DataRepresentation) to either load or create embeddings for
    a specific persuasion principle (POP) based on the data provided.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data to generate embeddings for.
        pop_column (str): The name of the column representing the persuasion principle (POP).

    Returns:
        pd.DataFrame: A DataFrame containing the generated or loaded embeddings.
    """
    # Load or create embeddings for each persuasion principle
    data_rep = DataRepresentation(logger)  # Pass the logger to the class constructor

    # Get or create embeddings for the POP persuasion principle
    embedding_df = data_rep.load_or_create_embeddings(data, pop_column)

    # Return the resulting embeddings DataFrame
    return embedding_df
     

# ================================
# MAIN FUNCTION
# ================================
def main():
    """
    Main function that performs data gathering and embedding creation.
    
    This function orchestrates the data gathering process (by loading a CSV file),
    followed by the creation of embeddings for the specified persuasion principle (POP).
    """
    ##
    ## Step 1: Data Gathering and embedding creation
    ##
    data_name = "pop_dataset_Full(Tiltan).csv"  # File name
    data = data_gathering(data_name)
    
    ##
    ## Step 2: Embedding Creation
    ## 
    embeddings_df = create_embeddings(data, POP)
    print(embeddings_df.head())
    

# ================================
# PROGRAM EXECUTION
# ================================
if __name__ == "__main__":
    # Display the welcome screen
    splash()
     
    # Initialize the program setup
    initialize()
    
    # Execute the main program logic
    main()
