# Deep Learning for Information Retrieval on the Cranfield Dataset
## Introduction
This project explores the application of deep learning techniques to enhance information retrieval systems' performance using the Cranfield dataset. This dataset includes abstracts from aeronautical research papers and is a classic benchmark for evaluating IR models.

## Dataset Description
The Cranfield dataset is at the heart of this project. It consists of 1,400 document abstracts, 225 queries, and a set of relevance judgments linking queries to relevant documents. The dataset was preprocessed through tokenization, stopwords removal, and stemming to normalize the texts, making them suitable for deep learning model input.

## Environment and Dependencies
An environment suitable for deep learning experimentation was established, primarily using Python 3.8 or newer. Essential dependencies include TensorFlow for building and training the neural models and NLTK for natural language processing tasks like tokenization and stemming. Scikit-learn is crucial for computing metrics and evaluating models.

## Installation Instructions
To prepare the project environment:

Clone the project repository.
Install necessary Python packages using pip install -r requirements.txt.
Ensure TensorFlow installation is compatible with the hardware, especially for GPU acceleration.

## Data Preprocessing
The Cranfield dataset was meticulously prepared for deep learning applications. The initial step was tokenization, where text data were broken down into individual words, facilitating the identification of unique terms. Following this, stopwords—commonly used words contributing minimal to overall meaning—were removed to focus on more significant terms. Stemming was then applied to reduce words to their root forms, ensuring that variations of a word (e.g., 'connect', 'connecting', 'connected') were treated as the same term.

The final step in preprocessing involved converting texts into Term Frequency-Inverse Document Frequency (TF-IDF) vectors. This process transformed the textual data into a numerical format, highlighting the importance of each term within documents relative to the entire dataset. The TF-IDF approach helped diminish the effect of commonly used words across documents, amplifying the significance of more unique terms, thereby preparing the dataset for efficient neural network processing.

## Model Architecture
The project explored two distinct neural network architectures to evaluate their efficacy in information retrieval tasks:

### Convolutional Neural Network (CNN): 
This model leverages convolutional layers to process text, analogous to their application in image recognition. In text analysis, these layers help identify local patterns or features (e.g., phrase or word groupings) within documents, which are crucial for understanding document content and context. The CNN's ability to capture these local structures makes it particularly effective for tasks requiring the identification of relevant documents based on specific query terms.

### Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM): 
RNNs are tailored to handle sequential data, making them well-suited for text processing. The LSTM variant of RNNs was employed to overcome traditional RNNs' limitations in learning long-term dependencies. By remembering information for extended periods, LSTM units are particularly effective in understanding the context and flow of texts in documents and queries, enhancing the model's ability to match queries with relevant documents based on the overall context rather than isolated terms.

## Training and Evaluation
The dataset was divided into training (70%) and testing (30%) sets, following standard practices to ensure a robust evaluation of model performance. The models were trained over several epochs, with a batch size of 32, to iteratively improve their understanding of the dataset's text structures and relevance patterns.

Model performance was rigorously evaluated using metrics such as precision@k, recall@k, and F1 scores. These metrics provided a multi-faceted view of each model's retrieval effectiveness, balancing the trade-offs between returning many relevant documents (recall) and ensuring the documents returned are relevant (precision).

## Results and Discussion
The evaluation revealed distinct strengths for each model. The CNN model demonstrated higher precision, likely due to its effectiveness in identifying key local text features relevant to specific queries. In contrast, the RNN model, with LSTM units, showed improved recall rates, attributed to its superior capability in understanding the overall context and flow of both documents and queries.

These outcomes suggest different use cases for each model within information retrieval systems and highlight deep learning's potential to enhance traditional text processing techniques.

## Conclusion and Future Work
This project underscores deep learning's transformative potential in information retrieval, showcased by the experiments with the Cranfield dataset. The results offer a promising avenue for future work, which could include exploring more sophisticated neural network architectures, incorporating semantic understanding technologies to grasp deeper textual meanings, and refining models based on iterative user feedback to continually improve performance.

Future initiatives could also look into hybrid models that combine CNN and RNN strengths or delve into transfer learning and unsupervised learning approaches to leverage external datasets and knowledge bases, further enhancing retrieval accuracy and relevance.

## How to Use
For application of this project's findings:

Adapt the preprocessing steps to your dataset.
Execute model training using the provided scripts.
Evaluate the performance of the trained model and apply it to your information retrieval tasks.

## References:
1. Data Source: http://ir.dcs.gla.ac.uk/resources/test_collections/cran/
2. DSSM:https://www.microsoft.com/en-us/research/wpcontent/uploads/2016/02/cikm2013_DSSM_fullversion.pdf
3. Neural ranking models for document retrieval: 
https://link.springer.com/article/10.1007/s10791-021-09398-0
4. Foundations and Trends R© in Information Retrieval
Deep Learning for Matching in Search and Recommendation, by Jun Xu, Xiangnan He, 
Hang Li.
5. For data set collection Cranfield - ir_datasets (ir-datasets.com)
6. Cranfield experiments information at Cranfield experiments - Wikipedia
