# Human written vs bot-generated news

This work relates to news classification into the ones written by a human vs written by a bot / generated sentences. As part of the classification task, we create transition matrix Q for true sentences and use likelihood function in order to estimate the likelihood of a particular sentence being written by a human. We then use neural network that we train to predict the next word given current words. 

This work doesn't incorporate all of the details of the final algorithm, and is mostly used for demostration purposes of training neural networks for classification tasks within the space of natural language processing. 


# How to Run

## Dependencies
- pandas 
- numpy 
- scipy
- matplotlib
- sklearn 
- random
- seaborn 
- tensorflow 

## Setup:

• Clone the repository to your local machine \
• Ensure you've installed all required dependencies (see the "Dependencies" section) \
• Download data and place it in your working directory \
• Note: Some libraries are imported with specific abbreviations in the code, like import numpy as np
