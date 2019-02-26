__author__ = 'Serina Chang <sc3003@columbia.edu'
__date__ = 'Feb 25, 2019'

import csv
import os
import string

PATH_TO_TRAIN_TEST_SPLIT = '../raw_stab_essays.0/train-test-split.csv'
PATH_TO_RAW_CORPUS = '../raw_stab_essays.0/brat-project-final/'
PATH_TO_PROCESSED_CORPUS = '../segmentation_data/'
ADU_LABELS = {'MajorClaim', 'Claim', 'Premise'}
VALID_LEVELS = {'essay', 'paragraph'}

def create_processed_files(level):
    """
    Parses the essays and annotation files from the Stab corpus
    and converts them into formats that will be convenient for the
    BIO tagging task. Four files are created:
    - train.<level>-level.words.txt
    - train.<level-level.tags.txt
    - test.<level>-level.words.txt
    - test.<level>-level.tags.txt
    The words.txt files contain the words from the essays and tags.txt
    files contain the corresponding BIO tags for each of the words.
    <level> refers to whether the data samples are separated at the
    essay or paragraph level.
    """
    assert(level in VALID_LEVELS)
    pairs = {}
    for fn in os.listdir(PATH_TO_RAW_CORPUS):
        if fn.endswith('.txt') or fn.endswith('.ann'):
            essay_id = fn[:len(fn)-4]
            if essay_id not in pairs:
                pairs[essay_id] = []
            pairs[essay_id].append(PATH_TO_RAW_CORPUS + fn)

    train_words_fn = PATH_TO_PROCESSED_CORPUS + 'train.{}-level.words.txt'.format(level)
    train_tags_fn = PATH_TO_PROCESSED_CORPUS + 'train.{}-level.tags.txt'.format(level)
    test_words_fn = PATH_TO_PROCESSED_CORPUS + 'test.{}-level.words.txt'.format(level)
    test_tags_fn = PATH_TO_PROCESSED_CORPUS + 'test.{}-level.tags.txt'.format(level)
    train_ids, test_ids = get_train_test_split()

    with open(train_words_fn, 'w') as train_words_f, open(train_tags_fn, 'w') as train_tags_f, \
        open(test_words_fn, 'w') as test_words_f, open(test_tags_fn, 'w') as test_tags_f:
        for essay_id, fns in pairs.items():
            words_f = train_words_f if essay_id in train_ids else test_words_f
            tags_f = train_tags_f if essay_id in train_ids else test_tags_f
            if len(fns) == 2:  # both .ann and .txt were found
                txt_fn, ann_fn = sorted(fns, reverse=True)
                try:
                    all_words, all_tags = convert_to_words_and_tags(txt_fn, ann_fn, level)
                    if level == 'paragraph':
                        for words_per_paragraph, tags_per_paragraph in zip(all_words, all_tags):
                            words_f.write(' '.join(words_per_paragraph) + '\n')
                            tags_f.write(' '.join(tags_per_paragraph) + '\n')
                    else:
                        words_f.write(' '.join(all_words) + '\n')
                        tags_f.write(' '.join(all_tags) + '\n')
                except AssertionError:
                    print('Failed:', essay_id)

def convert_to_words_and_tags(txt_fn, ann_fn, level):
    """
    Given the filename to the essay text and the filename to the essay
    annotation, this function returns the words in the essay and the
    corresponding BIO tags for those words. If the level is essay, a
    flat list is returned for words and tags each; otherwise, a list of
    lists is returned for each, separated into paragraphs.
    """
    content = open(txt_fn, 'r').read()
    char_tag_tups = [[c, 'O'] for c in content]
    spans = extract_adu_spans(ann_fn)
    for beg, end in spans:
        assert(beg > 0 and end < len(char_tag_tups))
        for i in range(beg, end):
            char_tag_tups[i][1] = 'I'
    all_words = []
    all_tags = []
    prompt, _, start = _convert_next_paragraph_to_words_and_tags(char_tag_tups, start=0)
    start += 1  # two newlines after prompt
    while start < len(char_tag_tups):
        words, tags, start = _convert_next_paragraph_to_words_and_tags(char_tag_tups, start=start)
        if len(words) > 0:
            if level == 'paragraph':
                all_words.append(words)
                all_tags.append(tags)
            else:
                all_words += words
                all_tags += tags
    return all_words, all_tags

def _convert_next_paragraph_to_words_and_tags(char_tag_tups, start):
    """
    This is a helper function for returning the words and tags in
    a given pair of files. char_tag_tups contains the file's characters
    in order, tagged by either 'O' or 'I' depending on whether they
    are outside or inside of a labeled ADU span from the annotation
    file. start indicates which char tag tuple to begin at. The
    tuples are processed in order and converted into words and word-level
    tags. When a newline is reached, i.e. a new paragraph, the function
    returns and records the index of the tuple containing the newline.
    """
    words = []
    tags = []
    i = start
    while i < len(char_tag_tups):
        consecutive_chars = []
        word_tag = None
        j = i
        punc_break = False
        while j < len(char_tag_tups):
            curr_char, curr_tag = char_tag_tups[j]
            if curr_char == '\n':  # new paragraph
                return words, tags, j+1
            if curr_char == ' ':
                break
            if curr_char in string.punctuation:
                punc_break = True
                break
            else:
                consecutive_chars.append(curr_char)
                if word_tag is None:
                    word_tag = curr_tag
                else:
                    assert(word_tag == curr_tag)
                j += 1
        if len(consecutive_chars) > 0:
            words.append(''.join(consecutive_chars))
            tags = append_tag(tags, word_tag)
        if punc_break:
            punc, punc_tag = char_tag_tups[j]
            words.append(punc)
            tags = append_tag(tags, punc_tag)
        i = j+1
    return words, tags, i

def append_tag(tags, word_tag):
    """
    Appends a desired word_tag to tags, converting it from 'I' to 'B' if
    necessary.
    """
    if word_tag == 'O':
        tags.append(word_tag)
    else:  # word_tag == 'I'
        if len(tags) > 0 and tags[-1] in {'B', 'I'}:  # already inside
            tags.append(word_tag)
        else:
            tags.append('B')
    return tags

def extract_adu_spans(fn):
    """
    Parses an annotation file and extracts the beginning and end indices
    of each labeled argumentative span.
    """
    spans = []
    with open(fn, 'r') as f:
        for line in f.readlines():
            elem, remainder = line.strip().split('\t', 1)
            if any(remainder.startswith(u) for u in ADU_LABELS):
                label, text = remainder.split('\t', 1)
                label, beg, end = label.split()
                spans.append((int(beg), int(end)))
    return sorted(spans)

def get_train_test_split():
    """
    Retrieves train/test split from Stab corpus.
    """
    train_ids = set()
    test_ids = set()
    with open(PATH_TO_TRAIN_TEST_SPLIT, 'r') as train_test_csv:
        reader = csv.DictReader(train_test_csv, delimiter=';')
        for row in reader:
            if row['SET'] == 'TRAIN':
                train_ids.add(row['ID'])
            else:
                test_ids.add(row['ID'])
    return train_ids, test_ids

if __name__ == '__main__':
    create_processed_files('paragraph')