# Automatic Analysis of Persuasion Principles in Phishing Messages

This project aims to develop an automated system capable of analyzing phishing messages and detecting the presence and intensity of different persuasion principles employed in them. The goal is to categorize and evaluate the messages efficiently using Natural Language Processing (NLP) techniques and machine learning.

## Objectives

The main objective of the project is to identify and classify the persuasion principles within phishing messages. These principles include, but are not limited to:

- **Authority**: Use of authority figures or authoritative instructions.
- **Urgency**: Creation of a sense of urgency or pressure for the recipient to act quickly.
- **Reciprocity**: Appeal to an implied obligation to respond or act in favor of the sender.
- **Scarcity**: Presentation of a limited or exclusive opportunity.
- **Deception**: Use of deceptive tactics to influence decisions.

## Methodology

### Step 1: Text Representation
To represent the messages in a way that is understandable by machines, **word embeddings** will be used. These embeddings are numerical representations of words and phrases that capture semantic context, using pretrained models such as **DistilBERT** or **FastText**. These models allow the system to understand the text without predefined rules.

### Step 2: Similarity Analysis
**Cosine similarity** will be calculated between the embeddings of phishing messages and the representations of persuasion principles. Each principle will have representative phrases, and the system will try to determine how similar a phishing message is to these principles, assigning an intensity score.

### Step 3: Intensity Evaluation
The system will assign an **intensity score** to each persuasion principle in the message, on a scale from 1 to 10. This will allow us to evaluate how strong the presence of each principle is in the message and assess its potential effectiveness in persuading the recipient.

## Unsupervised Approach

The approach used for this analysis will be **unsupervised**, meaning that no manual labels are required for training data. Instead, the system will learn the representations of persuasion principles from the data without direct human intervention.

## Challenges and Expectations

- **Message variety**: Phishing messages can vary significantly in structure and content, making it challenging to identify persuasion principles.
- **Language model**: The performance of the word embeddings model will be crucial, and different approaches such as DistilBERT, FastText, and other similar models will be tested.
- **Intensity evaluation**: Evaluating the "intensity" of persuasion principles will be a complex task, requiring continuous refinement of the system.

## Contributions

This is an evolving project, and various techniques and methods will be explored to improve performance and accuracy. Any contributions or suggestions are welcome.

## Requirements

- **Python 3.x**
- **Hugging Face Transformers**
- **PyTorch**
- **Scikit-learn**

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/persuasion-analysis-phishing.git

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt

## Usage
1. Preprocess the phishing messages.
2. Use embedding models to represent the messages.
3. Evaluate the similarity between the messages and the representative phrases of persuasion principles.
4. Assign the intensity of persuasion principles in the message.

## License

This project is licensed under the MIT License - see the LICENSE file for details