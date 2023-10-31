# Import libraries
import pandas as pd
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import sklearn
import random
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# Load data
data = pd.read_csv("data")
data.head(3)
data.info()

# Creating a list of news
news= data["News"].values
news= data["News"].values
isinstance(news, list)

news_list = []
for item in news:
  news_list.append(item)
news = news_list
isinstance(news, list)

# Preprocessing
MAX_VOCAB = 9999
tokenizer = Tokenizer(num_words = MAX_VOCAB,
                      filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      oov_token = 'UNK',
                      lower="True"
                      )
tokenizer.fit_on_texts(news)
seq = tokenizer.texts_to_sequences(news)

# Checks
len(seq) == len(news)
tokenizer.word_index["hi"]

word_index = {}
for k,v in tokenizer.word_index.items():
    if v< MAX_VOCAB:
        word_index[k] = v
word_index["START"] = 0
word_index["END"] = MAX_VOCAB

# Check
word_index["START"]

# Creating dictionary
index_word = { v : k for k,v in word_index.items()}

random.shuffle(seq)

sequences = []
for sequence in seq:
    sequences.append([0] + sequence + [MAX_VOCAB])

# Creating transition matrix
sequences_flat_list = []
for sublist in sequences:
    for item in sublist:
        sequences_flat_list.append(item)

def transition_matrix(V, sequence):
    Q = [[0]*(V+1) for _ in range(V+1)]
    for (i,j) in zip(sequence,sequence[1:]):
        Q[i][j] += 1

    for row in Q:
        n = sum(row)
        if n > 0:
            row[:] = [f/n for f in row]

    return Q

Q = transition_matrix(MAX_VOCAB, sequences_flat_list)
Q = np.array(Q)

# Checks
Q.shape
np.sum(Q[1])
np.sum(Q[0:10000,])
probabilities = np.sum(Q, axis = 1)
print(len(probabilities)-sum((probabilities >0.99)))

# Working with sentences
random.seed(2021)
true_sentences = random.sample(sequences,5)

def likelihood_test(sentences, T_matrix):
    n = len(sentences)
    likelihood = [0]*n
    for sentence in range(n):
        for (i,j) in zip(sentences[sentence],sentences[sentence][1:]):
            if T_matrix[i,j] != 0:
                likelihood[sentence] += np.log(T_matrix[i,j])
            else:
                likelihood[sentence] += np.log(10**(-18))
        likelihood[sentence] = likelihood[sentence]/len(sentences[sentence])
    return likelihood

true_likelihood = likelihood_test(true_sentences,Q)
print(true_likelihood)

# Average log-likelihood for true sentences
average_loglikelihood = np.mean(true_likelihood)
average_loglikelihood

def max_min(sequences):
    lengths = [len(seq) for seq in sequences]
    maxim = max(lengths)
    minim = min(lengths)
    return minim, maxim

max_min(seq)

# Create fake sentences as a check
fake_sentences = []

for i in range(5):
    length_random = random.randint(2,65)
    sentence_fake = random.sample(range(1,MAX_VOCAB-1), length_random)
    fake_sentences.append(sentence_fake)

# Check
len(fake_sentences[0])

fake_sequences_comparison = []
for sentence in fake_sentences:
    fake_sequences_comparison.append([0] + sentence + [MAX_VOCAB])

# Checks
fake_sequences_comparison[0]
fake_likelihood = likelihood_test(fake_sequences_comparison,Q)

# Average log-likelihood for fake sentences
average_loglikelihood_fake = np.mean(fake_likelihood)
average_loglikelihood_fake

def create_matrix(sequences, V):
  N = len(sequences)
  tensor_data = np.zeros((N, V))
  for i, sequence in enumerate(sequences):
    tensor_data[i, sequence] = 1.
  return tensor_data

# Hyperparameters
D = 8
learning_rate = 0.0001
epochs = 1 # to check, for computational efficiency. In reality should be larger

W_1 = np.array([ [ np.random.rand() for i in range(D) ] for j in range(MAX_VOCAB+2) ])
W_2 = np.array([ [ np.random.rand() for i in range(MAX_VOCAB+2) ] for j in range(D) ])

def new_sgd(sentence, learning_rate, W_1, W_2):

  sentence_matrix = create_matrix(sentence, MAX_VOCAB+2)
  feature_matrix = sentence_matrix[:len(sentence_matrix)-1,:]
  target_matrix = sentence_matrix[1:, :]


  hidden_matrix = np.tanh(W_1[sentence[:-1]])
  prediction_matrix = softmax(hidden_matrix.dot(W_2))

  # Gradients
  N_matrix = np.array([ [ 1 for i in range(D) ] for j in range(len(sentence_matrix)-1) ])
  gW_2 = np.transpose(hidden_matrix).dot(prediction_matrix - target_matrix)
  gW_1 = np.transpose(feature_matrix).dot(np.multiply((prediction_matrix - target_matrix).dot(np.transpose(W_2)), (N_matrix-np.multiply(hidden_matrix, hidden_matrix))))

  W_1 -= learning_rate*gW_1
  W_2 -= learning_rate*gW_2

  cost = tf.keras.losses.categorical_crossentropy(target_matrix, prediction_matrix).numpy()

  return W_1, W_2, cost.sum()


full_costs = []

random.shuffle(sequences)

for i in range(0,len(sequences)):
  W_1, W_2, tmp_cost = new_sgd(sequences[i], learning_rate, W_1, W_2)
  full_costs.append(tmp_cost)

# Exponentially weighted moving average, with a smoothing factor of 0.01
ewma = pd.Series(full_costs).ewm(alpha=0.01).mean()

# Plotting the loss function output as a function of batch size
plotting_data = pd.DataFrame({'Loss': full_costs, "EWMA Loss": ewma, 'Batch': range(1, len(full_costs)+1)})
plotting_data.head()

sns.lineplot(x = 'Batch', y = 'EWMA Loss', data = plotting_data, color = "coral").set_title('EWMA Loss')
