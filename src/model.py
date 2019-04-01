__author__ = 'Serina Chang <sc3003@columbia.edu'
__date__ = 'Feb 25, 2019'

from collections import Counter
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Input, model_from_json
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import os
from sklearn.metrics import f1_score

CORPORA_KEYS = {'essays':'0.stab_essays', 'editorials':'1.webis_editorials'}
TAG2IDX = {'<PAD>':0, 'B':1, 'I':2, 'O':3}
PATH_TO_GLOVE = '../../../glove.6B/'
PATH_TO_MODEL = '../mdl/'

def get_data(corpus_key, level, group, lowercase=True):
    """
    Retrieves the preprocessed text files for the given level and
    group (train, dev, or test), and parses them to create BIO sequences in
    the form of (word, tag) tuples.
    """
    assert(corpus_key in CORPORA_KEYS)
    words_txt_fn = '../{}/processed/{}.{}-level.words.txt'.format(CORPORA_KEYS[corpus_key], group, level)
    assert(os.path.isfile(words_txt_fn))
    tags_txt_fn = '../{}/processed/{}.{}-level.tags.txt'.format(CORPORA_KEYS[corpus_key], group, level)
    assert(os.path.isfile(tags_txt_fn))
    data = []
    with open(words_txt_fn, 'r') as words_f, open(tags_txt_fn, 'r') as tags_f:
        words_lines = words_f.readlines()
        tags_lines = tags_f.readlines()
        for wl, tl in zip(words_lines, tags_lines):
            if lowercase:
                wl = wl.lower()
            words = wl.strip().split()
            tags = tl.strip().split()
            assert(len(words) == len(tags))
            data.append(list(zip(words, tags)))
    return data

def load_glove_embs(dim, max_words, verbose=False):
    """
    Loads max_words GloVe embeddings with the given dimension.
    """
    word2emb = {}
    fn = PATH_TO_GLOVE + 'glove.6B.{}d.txt'.format(dim)
    with open(fn, 'r') as f:
        for i, line in enumerate(f.readlines()[:max_words]):
            line = line.strip()
            toks = line.split()
            word = toks[0]
            if word.isalpha() and word.lower() == word and len(word) < 20:
                emb = np.array([float(e) for e in toks[1:]])
                assert(len(emb) == dim)
                word2emb[word] = emb
            if verbose and i % 1000 == 0:
                print(i, word)
    return word2emb

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

def prep_data_matrices(train_corpus_keys, test_corpus_keys, level):
    """
    Preps the data matrices for training and testing.
    """
    train_corpora = {}
    test_corpora = {}  # corpus_key to test set
    train_data = []
    test_data = []
    for ck in train_corpus_keys:
        train_corpora[ck] = get_data(ck, level, group='train')
        train_data += train_corpora[ck]
    for ck in test_corpus_keys:
        if ck == 'editorials':
            test_corpora[ck] = get_data(ck, level, group='dev')
        else:
            test_corpora[ck] = get_data(ck, level, group='test')
        test_data += test_corpora[ck]
    max_len = np.max([len(sample) for sample in train_data + test_data])
    word2idx = index_words(train_data)

    train_matrices = {}
    for ck, train_data in train_corpora.items():
        X_tr = [[word2idx[tup[0]] for tup in s] for s in train_data]
        X_tr = pad_sequences(maxlen=max_len, sequences=X_tr, padding='post', value=0)
        y_tr = [[TAG2IDX[tup[1]] if tup[1] in TAG2IDX else TAG2IDX['<UNK>'] for tup in s] for s in train_data]
        y_tr = pad_sequences(maxlen=max_len, sequences=y_tr, padding='post', value=0)
        y_tr = np.array([to_categorical(i, num_classes=len(TAG2IDX)) for i in y_tr])
        print('{} train:'.format(ck), X_tr.shape, y_tr.shape)
        train_matrices[ck] = (X_tr, y_tr)
    test_matrices = {}
    for ck, test_data in test_corpora.items():
        X_te = [[word2idx[tup[0]] if tup[0] in word2idx else word2idx['<UNK>'] for tup in s] for s in test_data]
        X_te = pad_sequences(maxlen=max_len, sequences=X_te, padding='post', value=0)
        y_te = [[TAG2IDX[tup[1]] if tup[1] in TAG2IDX else TAG2IDX['<UNK>'] for tup in s] for s in test_data]
        y_te = pad_sequences(maxlen=max_len, sequences=y_te, padding='post', value=0)
        y_te = np.array([to_categorical(i, num_classes=len(TAG2IDX)) for i in y_te])
        print('{} test:'.format(ck), X_te.shape, y_te.shape)
        test_matrices[ck] = (X_te, y_te)
    return train_matrices, test_matrices, word2idx

def prep_embedding_matrix(word2idx, dim):
    """
    Creates an embedding matrix for the given words and indices.
    """
    word2emb = load_glove_embs(dim, max_words=50000)
    matrix = np.random.uniform(size=(len(word2idx), dim))
    missed = []
    num_matched, num_missed = 0, 0
    for word, idx in word2idx.items():
        if word in word2emb:
            matrix[idx] = word2emb[word]
            num_matched += 1
        else:
            missed.append(word)
            num_missed += 1
    print('found embeddings for {} words, missed {}'.format(num_matched, num_missed))
    return matrix

def compile_new_model(n_words, n_tags, max_len, emb_weights = None, emb_dim = 100):
    """
    Compiles a new LSTM.
    """
    input = Input(shape=(max_len,))
    if emb_weights is not None:
        model = Embedding(input_dim=n_words, output_dim=emb_dim, input_length=max_len, weights=[emb_weights], trainable=True)(input)
    else:
        model = Embedding(input_dim=n_words, output_dim=emb_dim, input_length=max_len)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def load_pretrained_model(json_fn, weights_fn):
    """
    Loads a pre-trained model and its weights.
    """
    model_json = open(json_fn, 'r').read()
    model = model_from_json(model_json)
    model.load_weights(weights_fn)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def compute_f1(gold, pred):
    """
    Computes the F1 score at the word-tag level (instead of paragraph level).
    """
    gold_tags, pred_tags = [], []
    for gold_sample, pred_sample in zip(gold, pred):
        gold_tags.append(np.argmax(gold_sample, axis=1))
        pred_tags.append(np.argmax(pred_sample, axis=1))
    gold_tags = np.concatenate(gold_tags, axis=0)
    pred_tags = np.concatenate(pred_tags, axis=0)
    f1s = f1_score(gold_tags, pred_tags, average=None)
    print('F1 per class:', f1s)
    print('Macro F1:', round(np.mean(f1s[1:]), 3))

def baseline(level = 'paragraph', pretrained_embs = True, emb_dim = 100, epochs = 30):
    """
    Train on essays, test on essays. Can used pretrained GloVe embeddings
    or randomly initialized.
    """
    train_matrices, test_matrices, word2idx = prep_data_matrices(['essays'], ['essays'], level)
    X_tr, y_tr = train_matrices['essays']
    X_te, y_te = test_matrices['essays']
    max_len = X_tr.shape[1]
    n_words = len(word2idx)
    n_tags = len(TAG2IDX)
    if pretrained_embs:
        embedding_matrix = prep_embedding_matrix(word2idx, emb_dim)
    else:
        embedding_matrix = None
    model = compile_new_model(n_words, n_tags, max_len, emb_weights=embedding_matrix, emb_dim=emb_dim)
    weights_fn = PATH_TO_MODEL + 'baseline.weights.hdf5'
    es = EarlyStopping(monitor='val_loss', patience=3)
    mc = ModelCheckpoint(weights_fn, monitor='val_loss', save_best_only=True, verbose=1)
    model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=epochs, callbacks=[es, mc], validation_split=0.1, verbose=1)
    model.load_weights(weights_fn)
    compute_f1(y_te, model.predict(X_te))

def pretrain_model_on_editorials(level = 'paragraph', pretrained_embs = True, emb_dim = 100, pretrain_epochs = 10, epochs = 30):
    """
    Pretrain on editorials, then train on essays and test on essays.
    Again, can use pretrained GloVe embeddings or randomly initialized.
    """
    train_matrices, test_matrices, word2idx = prep_data_matrices(['essays', 'editorials'], ['essays', 'editorials'], level)
    es = EarlyStopping(monitor='val_loss', patience=3)

    print('PRETRAINING ON EDITORIALS...')
    X_ptr, y_ptr = train_matrices['editorials']
    X_pte, y_pte = test_matrices['editorials']
    max_len = X_ptr.shape[1]
    n_words = len(word2idx)
    n_tags = len(TAG2IDX)
    if pretrained_embs:
        embedding_matrix = prep_embedding_matrix(word2idx, emb_dim)
    else:
        embedding_matrix = None
    model = compile_new_model(n_words, n_tags, max_len, emb_weights=embedding_matrix, emb_dim=emb_dim)
    pretrained_weights_fn = PATH_TO_MODEL + 'pretrained.weights.hdf5'
    mc_ptr = ModelCheckpoint(pretrained_weights_fn, monitor='val_loss', save_best_only=True, verbose=1)
    model.fit(X_ptr, np.array(y_ptr), batch_size=32, epochs=pretrain_epochs, callbacks=[es, mc_ptr], validation_split=0.1, verbose=1)
    model.load_weights(pretrained_weights_fn)
    compute_f1(y_pte, model.predict(X_pte))

    print('REAL TRAINING...')
    X_tr, y_tr = train_matrices['essays']
    X_te, y_te = test_matrices['essays']
    with_pretrained_weights_fn = PATH_TO_MODEL + 'with_pretrained.weights.hdf5'
    mc = ModelCheckpoint(with_pretrained_weights_fn, monitor='val_loss', save_best_only=True, verbose=1)
    model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=epochs, callbacks=[es, mc], validation_split=0.1, verbose=1)
    model.load_weights(with_pretrained_weights_fn)
    compute_f1(y_te, model.predict(X_te))

if __name__ == '__main__':
    # X_tr, y_tr, test_matrices, word2idx = prep_data_matrices(['essays', 'editorials'], ['essays', 'editorials'], 'paragraph')
    # y_te = test_matrices['essays'][1]
    # compute_f1(y_te, y_te)
    # train_matrices, test_matrices, word2idx = prep_data_matrices(['essays', 'editorials'], ['essays', 'editorials'], 'paragraph')
    # max_len = X_tr.shape[1]
    # n_words = len(word2idx)
    # n_tags = len(TAG2IDX)
    # baseline(epochs=10)
    pretrain_model_on_editorials(pretrain_epochs=10, epochs=10)
