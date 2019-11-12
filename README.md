# StackOverflowTag-Classcification
This is a text-Classification task to classify the tags of [azure, aws, gcp], based on the title of StackOverflow questions.  

Data is scraped from StackOverflow.  

## Dataset Details
* Azure: 77216 titles
* AWS: 82182 titles
* GCP: 17576 titles

## Requirements:
* Python 3.7
* TensorFlow 2.0
* Scikit-Learn
## Machine-Learning Models

## Deep-Learning Models
Several Deep Models are implemented  
### Training Details
* steps: 2 epochs
* batch_size: 64
* embedding_size: 128
* max_seq_length: 50
* RNN_dimension: 128
* dropout_rate: 0.5
* testing_size: 0.2
### Models
#### FastText
* Accuracy on training: 0.9574
* Accuracy on testing: 0.9411
#### TextCNN
* Accuracy on training: 0.9622
* Accuracy on testing: 0.9457
#### TextRNN
* Accuracy on training: 0.9599
* Accuracy on testing: 0.9410
#### TextRCNN
* Accuracy on training: 0.9603
* Accuracy on testing: 0.9427
