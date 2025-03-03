import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from src.utils.constants import TOKENIZER, EMBEDDINGS_MODEL, RESULT_FOLDER, PRINCIPLE_FOLDER, POP
from src.data.file_manager import FileManager

class DataRepresentation:
    """
    Class for handling the representation of data for machine learning tasks, specifically for obtaining text embeddings.
    
    This class uses the DistilBERT model to generate text embeddings for different representations (e.g., persuasion principles).
    It also handles loading and saving embeddings to/from CSV files, and ensures that embeddings are generated if they don't already exist.
    """

    def __init__(self, logger, tokenizer_model=TOKENIZER, embeddings_model=EMBEDDINGS_MODEL):
        """
        Initialize the DataRepresentation class by loading the tokenizer and embeddings model.
        
        Parameters:
            logger (LoggerSingleton): An already instantiated logger.
            tokenizer_model (str): The pre-trained tokenizer to load (default is DistilBERT).
            embeddings_model (str): The pre-trained embeddings model to load (default is DistilBERT).
        """
        self.logger = logger
        try:
            # Load the tokenizer and embeddings model
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_model)
            self.model = DistilBertModel.from_pretrained(embeddings_model)
            self.logger.info(f"Tokenizer '{tokenizer_model}' and model '{embeddings_model}' have been downloaded/loaded successfully!")
        except Exception as e:
            self.logger.error(f"Error downloading/loading '{tokenizer_model}' tokenizer and model: {e}")
            raise
        
        # Instantiate the DataHandler for file operations
        self.file_manager = FileManager(self.logger)

    def get_embeddings(self, texts):
        """
        Generate embeddings for the provided texts using the DistilBERT model.
        
        Parameters:
            texts (list): A list of text strings for which embeddings will be generated.
        
        Returns:
            numpy.ndarray: The embeddings for each text as a numpy array.
        """
        self.logger.info("Starting to process texts for embeddings...")

        try:
            # Tokenize the input texts
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            self.logger.info(f"Tokenized {len(texts)} texts.")
        
            # Generate embeddings
            with torch.no_grad():
                self.logger.info("Generating embeddings...")
                outputs = self.model(**inputs)
            
            # Use the mean of the word embeddings to obtain a single embedding per text
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            self.logger.info(f"Generated embeddings with shape: '{embeddings.shape}'")

            return embeddings

        except Exception as e:
            # Log the exception and raise it to be handled elsewhere
            self.logger.error(f"An error occurred while generating embeddings: {e}")
            raise


    def load_or_create_embeddings(self, data, pop):
        """
        Check if embeddings for a specific persuasion principle ('pop') exist. If not, generate and save them.
        
        Parameters:
            data (pd.DataFrame): The dataset containing text data and 'pop' column for filtering.
            pop (str): The column representing the persuasion principle (e.g., 'pop').
        
        Returns:
            pd.DataFrame: The embeddings DataFrame.
        """

        # Construct file name for the embeddings based on the persuasion principle
        file_name = f"embeddings_{pop}.csv"

        # Filter the data for the selected principle
        pop_data_df = data.loc[data[pop] == 1, ['id', 'path', 'hash', 'subject', 'txt']]

        # Check if the embeddings file for the selected principle already exists using DataHandler
        if os.path.isfile(os.path.join(PRINCIPLE_FOLDER, file_name)):
            self.logger.info(f"Embeddings for '{pop}' already exist. Loading the embeddings...")
            try:
                # Load the existing embeddings using DataHandler
                embed_df = self.file_manager.load(os.path.join(PRINCIPLE_FOLDER, file_name))
                self.logger.info(f"Embeddings loaded successfully for '{pop}'.")
                
                # Rename columns to 'col_#' format if necessary
                embed_df.columns = [f"col_{i}" if col.startswith("col_") else col for i, col in enumerate(embed_df.columns)]

            except Exception as e:
                self.logger.error(f"Error loading embeddings for '{pop}': {e}")
                raise
        else:
            # If embeddings don't exist, generate new embeddings
            self.logger.info(f"Creating embeddings with '{EMBEDDINGS_MODEL}' representation for '{pop}'...")
            embed = self.get_embeddings(pop_data_df["txt"].tolist())

            # Convert embeddings to DataFrame and add the necessary identifying columns
            embed_df = pd.DataFrame(embed)
            embed_df.columns = [f"col_{i}" for i in range(embed_df.shape[1])]
            embed_df["id"] = pop_data_df["id"].values
            embed_df["path"] = pop_data_df["path"].values

        return embed_df
