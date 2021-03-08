## Text Classification

  Text Classification (TC) and Named-Entity Recognition (NER) are two fundamental tasks for many Natural Language Processing (NLP) applications, which involve understanding, extracting information, and categorizing the text. In order to achieve such goals, we utilized AutoPhrase (Jingbo Shang, 2018) and a pre-trained language NER model to extract quality phrases. Using these as part of our features, we are able to achieve high performance for a five class and a twenty class text classification dataset. Our project follows a similar setting as previous works with train, validation, and test datasets and comparing the results across different methods.
  
  Text classification is an important NLP task, which can be understood as a given set of text and labels. We want to create a classifier that can classify them in addition to other texts. Text classification tasks mainly involve understanding the text and extracting high quality phrases for model training. For example, if a text has "government" or "minister" as a frequent phrase or word, it is more likely to belong to 'Politics'. As such, it is important for us to extract quality phrases and make sure they represent these documents well.

## Data Sets

  For our project, we have decided to use two news data sets: a BBC News data set and a 20 News data set.
  
  **BBC News dataset:** 
  
  - includes 2,225 documents
  - spans 2004-2005 
  - composed of five categories
    - entertainment
    - technology
    - politics
    - business
    - sports
  
  **20 News Groups dataset:** 
  - includes 18,000 news groups posts
  - composed of 20 categories e.g. Computer, Science, Politics, Religion, etc.

## Models

### AutoPhrase-based Model

  AutoPhrase has two major modules: Robust Positive-Only Distant Training and POS-Guided Phrasal Segmentation. The first module trains the model and determines the quality score for phrases. The second module determines which tokens should be combined together and constitute a phrase. AutoPhrase first estimates the quality score from frequent n-gram phrases. With these results, it then utilizes the segmentation module to revise the segmentation. Rather than using the n-gram based phrases, AutoPhrase estimates the final quality score based on the segmentation results. Since the AutoPhrase method is applicable to any domain and language, we utilized this method on both of our datasets to extract quality phrases. With these quality phrases, we adopt the same Bag-of-Words and TF-IDF processes to encode them into vectors.

### Pre-trained NER-based Model

BERT (Bidirectional Encoder Representations from Transformers) is a general-purpose language model trained on the large dataset. This pre-trained model can be fine-tuned and used for different tasks such as sentiment analysis, question answering systems, sentence classification, and Named-Entity Recognition. Named-Entity Recognition is the process of extracting noun entity from text data and classifies them into predefined categories e.g. person, location, organization and others. Hence, we can use a BERT-based Named-Entity Recognition model, fine-tuned on the CoNLL 2003 dataset, to extract noun entities in the BBC News data set and 20 News group datasets.

For our experiment, we have used the BERT-based uncased model as a baseline trained by the HuggingFace library with 110M parameters, 12 layers, 768-hidden, and 12-heads. For fine-tuning, we used the suggested parameters of max-seq-length=128, training epoch=3, and warm-up proportion=0.1. Then, we created the dataframe for BBC News summary data and used the model to predict the entity by each sentence of the document. We followed the same procedure for the 20 News dataset.

_reference:_ https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/

## Experiment

**Logistic Regression** is a binary classifier model that is widely adopted for many research projects and real-world applications. As such, we included this model in our experiment as well. This model is optimized by minimizing the **Logistic Loss** (Equation 1). 

A **Support Vector Machine (SVM)** is a supervised model intended for solving classification problems. The SVM algorithm creates a line or a hyper-plane, which separates the data into classes. This model is optimized by minimizing the **Hinge Loss** (Equation 2)

The architecture of BERT's transfer learning is made up by a fully-connected layer, a drop-out layer, a **Rectified Linear Unit (ReLU)** activation layer, a second fully-connected layer, and a **soft-max** activation layer. For the optimizer, we used **AdamW**, an improved version of **Adam**, and opted to use the _negative_ log-likelihood loss, which is well-suited for multiple-class classification. For training, we used a learning rate of exp(-4) for 40 epochs. Due to GPU resources, we were only able to perform training and evaluation on the BBC News dataset.


## Result

### Result Matrix

## Conclusion

## Reference
