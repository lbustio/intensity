# Semi-Supervised Analysis of Principles of Persuasion Intensity in Phishing Messages

![Automatic Analysis of Persuasion Principles in Phishing Messages](resources/banner.png)

This project aims to propose a semi-supervised method for detecting the intensity of various principles of persuasion in phishing messages (see [this](https://link.springer.com/chapter/10.1007/978-3-031-49552-6_14) and [this](https://www.sciencedirect.com/science/article/abs/pii/S1084804524001413)). The goal is to categorize and evaluate the messages efficiently using Natural Language Processing (NLP) techniques and machine learning. Clustering will be used to automatically determine the intensity levels of the principles of persuasion, and regression will be employed to predict the intensity of these principles in unknown messages based on the found intensities. This is an ongoing project, and its development may vary.

## Project Structure

```bash
lbustio-intensity/
├── readme.md                # This file
├── intensity.code-workspace  # VS Code workspace
├── intensity.ipynb           # Notebook with analysis and experiments
├── requirements.txt          # Project dependencies
├── data/                     # Phishing dataset
│   ├── pop_dataset_Full(Tiltan).csv
│   └── pop_dataset_Full(Tiltan).xlsx
└── resources/                # Additional resources
```


## Objectives

The main objective of the project is to detect the intensity of the principles of persuasion within phishing messages. These principles, as identified by Ana Ferreira and Lázaro Bustio-Martínez, include:

- **Authority**: Use of authority figures or authoritative instructions.
- **Urgency**: Creation of a sense of urgency or pressure for the recipient to act quickly.
- **Reciprocity**: Appeal to an implied obligation to respond or act in favor of the sender.
- **Scarcity**: Presentation of a limited or exclusive opportunity.
- **Deception**: Use of deceptive tactics to influence decisions.

## Methodology

### Step 1: Text Representation
To represent the messages in a way that is understandable by machines, **word embeddings** will be used. These embeddings are numerical representations of words and phrases that capture semantic context, using pretrained models such as **DistilBERT** or **FastText** (other data representation can be explored as well). These models allow the system to understand the text without predefined rules.

### Step 2: Clustering Analysis
The data will be segmented according to the presence of each principle of persuasion, and the analysis will be conducted for each principle independently in parallel. Several clustering algorithms, such as **k-Means**, **OPTICS**, **DBSCAN**, **GMM**, and others, will be evaluated to identify the groups formed in the data. For algorithms requiring a predetermined number of clusters, different numbers of clusters will be tested to find the optimal configuration. The resulting clusters will be evaluated and analyzed to assess their quality and understand the nature of the created groups. Each cluster is expected to contain messages with similar intensity values for each principle of persuasion. Based on these clusters, a metric will be proposed to measure the intensity of each principle of persuasion.

### Step 3: Intensity Evaluation
The system will assign an *intensity score* to each principle of persuasion in the message, on a scale from 1 to 10. This will allow the evaluation of how strong the presence of each principle is in the message and assess its potential effectiveness in persuading the recipient.

## Semi-Supervised Approach

This project employs a semi-supervised approach, combining clustering and regression techniques to analyze the intensity of principles of persuasion in phishing messages. The data will be segmented according to the presence of each principle of persuasion, and the analysis will be conducted for each principle independently in parallel. Several clustering algorithms, such as **k-Means**, **OPTICS**, **DBSCAN**, **GMM**, and others, will be evaluated to identify the groups formed in the data. For algorithms requiring a predetermined number of clusters, different numbers of clusters will be tested to find the optimal configuration. The resulting clusters will be evaluated and analyzed to assess their quality and understand the nature of the created groups. Each cluster is expected to contain messages with similar intensity values for each principle of persuasion. Based on these clusters, a metric will be proposed to assign an intensity value to each principle.

The approach used for this analysis will be **unsupervised**, meaning that no manual labels are required for training data. Instead, the system will learn the representations of principles of persuasion from the data without direct human intervention. Once the clusters are formed, a regression model is used to predict the intensity of each principle in unknown messages based on the patterns identified in the clustering phase. This approach allows the system to learn the representations of persuasion principles from the data without direct human intervention.

## Challenges and Expectations

- **Message variety**: Phishing messages can vary significantly in structure and content, making it challenging to identify principles of persuasion.
- **Language model**: The performance of the word embeddings model will be crucial, and different approaches such as **DistilBERT**, **FastText**, and other similar models will be tested.
- **Intensity evaluation**: Evaluating the "intensity" of principles of persuasion will be a complex task, requiring continuous refinement of the system.

## Contributions

This is an evolving project, and various techniques and methods will be explored to improve performance and accuracy. Any contributions or suggestions are welcome.

## Requirements

- Python 3.12.8
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Microsoft C++ Build Tools

### Note
Be sure to install Microsoft C++ Build Tools by following the instructions provided in this [Medium article](https://medium.com/@oleg.tarasov/building-fasttext-python-wrapper-from-source-under-windows-68e693a68cbb).

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/lbustio/intensity.git
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Preprocess the phishing messages.
2. Use embedding models to represent the messages.
3. Segment the data based on the presence of each principle of persuasion.
4. Apply clustering algorithms (e.g., k-Means, OPTICS, DBSCAN, GMM, etc.) to identify groups of messages.
5. Evaluate the resulting clusters to assess their quality and understand the nature of the created groups.
6. Propose a metric to assign an intensity value to each principle of persuasion.
7. Use a regression model to predict the intensity of each principle in unknown messages based on the patterns identified in the clustering phase.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
