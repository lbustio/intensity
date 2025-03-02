import os

# ================================
# PRINCIPLES OF PERSUASION
# ================================
# These principles are based on established psychological theories of influence.
# The principles represent different methods used to persuade or influence individuals.

# Persuasion principle to analyze (current choice is "authority")
POP = "authority"  # Example principle: "authority" - People tend to follow those they consider experts.

# A list of principles of persuasion that can be analyzed
POPS = [
    "authority",                             # Authority principle: people tend to follow those they consider experts
    "distraction",                           # Distraction: diverting the audience's attention to influence their decision
    "liking_similarity_deception",           # Liking and similarity: people are more likely to be persuaded by those they like or who are similar to them, even if deception is involved
    "social_proof",                          # Social proof: the influence that others' behavior has on decision-making
    "commitment_integrity_reciprocation"     # Commitment, integrity, and reciprocity: the desire to be consistent with prior commitments and the tendency to return favors
]

# ================================
# NATURAL LANGUAGE PROCESSING (NLP) MODEL CONFIGURATION
# ================================
# This section configures the models for text processing and analysis.

TOKENIZER = "distilbert-base-uncased"       # DistilBERT-based tokenizer that converts text into tokens for the model
EMBEDDINGS_MODEL = "distilbert-base-uncased"  # DistilBERT model that generates embeddings for words, useful for NLP tasks

# ================================
# CLUSTERING PARAMETERS
# ================================
# These parameters are used for clustering and analyzing the data.
MAX_K = 11  # Set the maximum number of clusters to consider when searching for optimal cluster count. 
            # Here it is set to 11 to allow searching from 2 to 10 clusters.

# ================================
# OUTPUT CONFIGURATION
# ================================
# This section handles output configurations, such as where to store results and what types of plots to generate.

# Types of plots to generate during analysis
PLOT_OBJECTIVES = ("optimum_k", "normal_plot")  # Plots include optimum_k (optimal number of clusters) and normal_plot (standard data plot)

# ================================
# DIRECTORY STRUCTURE
# ================================
# This section defines the working directories for different aspects of the analysis.
# These paths organize the workflow, from storing raw data to saving results.

RESULT_FOLDER = "results"  # Directory for storing analysis results
PRINCIPLE_FOLDER = os.path.join(RESULT_FOLDER, POP)  # Folder path for storing results specific to the persuasion principle being analyzed
DATA_FOLDER = "data"       # Directory for storing input and processed data
EXTERNAL_DATA_FOLDER = os.path.join("data", "external")  # Directory for storing external data sources
PROCESSED_DATA_FOLDER = os.path.join("data", "processed")  # Directory for storing processed data
RAW_DATA_FOLDER = os.path.join("data", "raw")  # Directory for storing raw data
MODEL_FOLDER = "models"    # Directory for storing trained models
TEMP_FOLDER = "temp"       # Directory for storing temporary files during processing
NOTEBOOK_FOLDER = "notebooks"  # Directory for storing Jupyter notebooks

# ================================
# ANALYSIS THRESHOLDS
# ================================
# Threshold values are used to define criteria for valid clustering and analysis results.

SILHOUETTE_THRESHOLD = 0.20  # Minimum silhouette score threshold for accepting clusters
                            # A higher value indicates that the clusters are better defined.

CORRESPONDENCE_THRESHOLD = 0.95  # Similarity threshold for correspondence calculation
                                # Represents the minimum percentage of similarity required for valid matching.
                                # Higher values indicate stricter matching criteria.

# ================================
# DATA PREPROCESSING
# ================================
# These options control how data is processed before clustering.

SCALE_DATA = True  # Whether to scale data before clustering. Scaling helps normalize the features for clustering.

# The following preprocessing options are currently not enabled but can be configured as needed:
# NORMALIZE_DATA = False    # Normalize the data
# STANDARDIZE_DATA = False  # Standardize the data
# LOG_TRANSFORM_DATA = False # Apply log transformation to the data
# MIN_MAX_SCALE = False     # Apply Min-Max scaling to the data

# ================================
# INSTALLATION OF REQUIRED PACKAGES
# ================================
# This section is responsible for ensuring that the required packages are installed in the environment.
# The INSTALL_PACKAGES flag controls whether packages should be installed if they are not already present.

INSTALL_PACKAGES = True  # Flag to determine whether required packages should be installed if they are missing

# List of packages to be installed, covering various data science and machine learning needs:
# These packages cover areas like data manipulation, machine learning, and GPU utilities.
PACKAGES = [
    "pandas",               # Data manipulation and analysis
    "numpy",                # Numerical computing
    "matplotlib",           # Basic plotting library
    "seaborn",              # Statistical data visualization
    "scipy",                # Scientific computing
    "scikit-learn",         # Machine learning algorithms
    "transformers",         # NLP transformers library
    "pandasgui",            # GUI for pandas DataFrames
    "torch",                # Deep learning framework
    "psutil",               # System and process utilities
    "gputil",               # GPU monitoring
    "hdbscan",              # Hierarchical density-based clustering
    "prettytable",          # Pretty-print tabular data
    "gputils",              # GPU utilities
    "plotly"                # Interactive plotting library
]