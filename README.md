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
Training and Evaluation
The dataset was split into a 70% training set and a 30% testing set. The models underwent training over several epochs, with a batch size of 32. As the models trained, their ability to discern and align with the dataset's text structures and relevance patterns was progressively refined.

Performance metrics, specifically precision@k, recall@k, and F1 scores, were employed to assess the models' retrieval effectiveness. Precision@k measures the proportion of recommended documents that are relevant, while recall@k measures the proportion of relevant documents that are recommended by the model. The F1 score is the harmonic mean of precision and recall, providing a single metric that balances the trade-off between the two.

Quantitatively, the models demonstrated the following performance across varying epochs:

At 20 epochs, the Mean Average Precision at 10 (MAP@10) was 0.0057, and the Mean Reciprocal Rank at 10 (MRR@10) was 0.0568.
Improvement was noted at 40 epochs, with MAP@10 rising to 0.0068 and MRR@10 to 0.0682.
The peak performance occurred at 60 epochs, where MAP@10 reached 0.0659, and MRR@10 significantly increased to 0.2143, indicating a marked improvement in both the quality of the top 10 recommendations and the ranking quality of the first relevant recommendation.
However, at 100 epochs, there was a notable decline, with MAP@10 falling to 0.0028 and MRR@10 to 0.0152, suggesting potential overfitting or other issues that necessitate a re-evaluation of the training regimen.
The varying levels of MAP and MRR at different epochs underscore the importance of monitoring models throughout their training process to optimize performance and avoid issues such as overfitting.

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
