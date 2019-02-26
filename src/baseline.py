__author__ = 'Serina Chang <sc3003@columbia.edu'
__date__ = 'Feb 25, 2019'

from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import numpy as np
from process_data import PATH_TO_PROCESSED_CORPUS, VALID_LEVELS
from sklearn.metrics import f1_score

TAG2IDX = {'<PAD>':0, '<UNK>':1, 'B':2, 'I':3, 'O':4}

def get_data(level, mode='train'):
    """
    Retrieves the preprocessed text files for the given level and
    mode (train or test), and parses them to create BIO sequences in
    the form of (word, tag) tuples.
    """
    assert(level in VALID_LEVELS)
    words_txt_fn = PATH_TO_PROCESSED_CORPUS + '{}.{}-level.words.txt'.format(mode, level)
    tags_txt_fn = PATH_TO_PROCESSED_CORPUS + '{}.{}-level.tags.txt'.format(mode, level)
    data = []
    with open(words_txt_fn, 'r') as words_f, open(tags_txt_fn, 'r') as tags_f:
        words_lines = words_f.readlines()
        tags_lines = tags_f.readlines()
        for wl, tl in zip(words_lines, tags_lines):
            words = wl.strip().split()
            tags = tl.strip().split()
            assert(len(words) == len(tags))
            data.append(list(zip(words, tags)))
    return data

def index_words(data):
    """
    Indexes the words in the data tuples, with indices according to the
    frequency of the words (from most to least frequent). Space is reserved
    in the vocabulary for padding and unknown words.
    """
    all_words = []
    for sample in data:
        all_words += [tup[0] for tup in sample]
    word2idx = {'<PAD>':0, '<UNK>':1}
    word_cts = Counter(all_words).items()
    sorted_word_cts = sorted(word_cts, key=lambda x:x[1], reverse=True)
    for i, (word, ct) in enumerate(sorted_word_cts):
        word2idx[word] = i+2
    return word2idx

def baseline(level):
    """
    Trains a baseline bidirectional LSTM on the BIO data. The max-length
    of each sample is determined dynamically, particularly since max-length
    should differ across different levels, i,e, paragraph-level should have
    shorter max-length than essay-level. After training, the model is
    evaluated on unseen data and the macro and per-class F1 scores are printed.
    """
    train_data = get_data(level, mode='train')
    test_data = get_data(level, mode='test')
    max_len = np.max([len(sample) for sample in train_data + test_data])
    word2idx = index_words(train_data)
    n_words = len(word2idx)
    n_tags = len(TAG2IDX)

    X_tr = [[word2idx[tup[0]] for tup in s] for s in train_data]
    X_tr = pad_sequences(maxlen=max_len, sequences=X_tr, padding='post', value=0)
    X_te = [[word2idx[tup[0]] if tup[0] in word2idx else word2idx['<UNK>'] for tup in s] for s in test_data]
    X_te = pad_sequences(maxlen=max_len, sequences=X_te, padding='post', value=0)

    y_tr = [[TAG2IDX[tup[1]] if tup[1] in TAG2IDX else TAG2IDX['<UNK>'] for tup in s] for s in train_data]
    y_tr = pad_sequences(maxlen=max_len, sequences=y_tr, padding='post', value=0)
    y_tr = np.array([to_categorical(i, num_classes=n_tags) for i in y_tr])
    y_te = [[TAG2IDX[tup[1]] if tup[1] in TAG2IDX else TAG2IDX['<UNK>'] for tup in s] for s in test_data]
    y_te = pad_sequences(maxlen=max_len, sequences=y_te, padding='post', value=0)
    y_te = np.array([to_categorical(i, num_classes=n_tags) for i in y_te])
    print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)

    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=10, validation_split=0.1, verbose=1)
    scores = model.evaluate(X_te, y_te)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    compute_f1(y_te, model.predict(X_te))

def compute_f1(gold, pred):
    gold_tags = []
    for sample in gold:
        tags = np.argmax(sample, axis=1)
        gold_tags.append(tags)
    gold_tags = np.concatenate(gold_tags, axis=0)
    pred_tags = []
    for sample in pred:
        tags = np.argmax(sample, axis=1)
        pred_tags.append(tags)
    pred_tags = np.concatenate(pred_tags, axis=0)
    f1s = f1_score(gold_tags, pred_tags, average=None)
    print('F1 per class:', f1s)
    print('Macro F1:', round(np.mean(f1s), 3))

if __name__ == '__main__':
    baseline('essay')