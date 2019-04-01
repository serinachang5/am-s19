__author__ = 'Serina Chang <sc3003@columbia.edu'
__date__ = 'Mar 30, 2019'

import os
import re

PATH_TO_RAW_CORPUS = '../1.webis_editorials/raw_download/annotated-txt/split-for-evaluation-final/'
PATH_TO_PROCESSED_CORPUS = '../1.webis_editorials/processed/'
VALID_LEVELS = {'essay', 'paragraph'}
ARGUMENT_LABELS = {'common-ground', 'assumption', 'testimony', 'statistics', 'anecdote'}
NONARGUMENT_LABELS = {'no-unit', 'other'}

def create_processed_files(level):
    """
    Parses the essays and annotation files from the Al-Khatib corpus
    and converts them into formats that will be convenient for the
    BIO tagging task. Six files are created:
    - train.<level>-level.words.txt
    - train.<level>-level.tags.txt
    - dev.<level>-level.words.txt
    - dev.<level>-level.tags.txt
    - test.<level>-level.words.txt
    - test.<level>-level.tags.txt
    The words.txt files contain the words from the articles and tags.txt
    files contain the corresponding BIO tags for each of the words.
    <level> refers to whether the data samples are separated at the
    essay or paragraph level.
    """
    assert(level in VALID_LEVELS)
    for group in ['train', 'dev', 'test']:
        words_fn = PATH_TO_PROCESSED_CORPUS + '{}.{}-level.words.txt'.format(group, level)
        tags_fn = PATH_TO_PROCESSED_CORPUS + '{}.{}-level.tags.txt'.format(group, level)
        with open(words_fn, 'w') as words_f, open(tags_fn, 'w') as tags_f:
            group_dir = PATH_TO_RAW_CORPUS + group + '/'
            for data_fn in os.listdir(group_dir):
                if data_fn.endswith('.txt'):
                    essay_id = data_fn[:-4]
                    try:
                        all_words, all_tags = convert_to_words_and_tags(group_dir + data_fn, level)
                        if level == 'paragraph':
                            for words_per_paragraph, tags_per_paragraph in zip(all_words, all_tags):
                                words_f.write(' '.join(words_per_paragraph) + '\n')
                                tags_f.write(' '.join(tags_per_paragraph) + '\n')
                        else:
                            words_f.write(' '.join(all_words) + '\n')
                            tags_f.write(' '.join(all_tags) + '\n')
                        print('Success:', essay_id)
                    except:
                        print('Failed:', essay_id)

def convert_to_words_and_tags(txt_fn, level):
    """
    Given the filename to the annotated text, this function returns the
    words in the essay and the corresponding BIO tags for those words.
    If the level is essay, a flat list is returned for words and tags
    each; otherwise, a list of lists is returned for each, separated
    into paragraphs.
    """
    lines = open(txt_fn, 'r').readlines()
    labeled_texts = [('par-sep', '')]  # end with par-sep
    for i in range(1, len(lines)):  # traverse backwards, ignore first line b/c title
        line = lines[i*-1]
        elements = line.strip().split('\t')
        assert(len(elements) >= 2)
        label = elements[1]
        if label == 'par-sep':
            labeled_texts.insert(0, (label, ''))
        else:
            assert(len(elements) >= 3)
            text = elements[2]
            if label == 'continued':
                assert(len(labeled_texts) > 0)
                subsequent_label = labeled_texts[0][0]
                if not subsequent_label.startswith('cont-'):
                    subsequent_label = 'cont-' + subsequent_label
                labeled_texts.insert(0, (subsequent_label, text))
            else:
                labeled_texts.insert(0, (label, text))

    all_words, all_tags = [], []
    paragraph_words, paragraph_tags = [], []
    for i, (label, text) in enumerate(labeled_texts):
        if label == 'par-sep':
            if len(paragraph_words) > 0:
                if level == 'paragraph':
                    all_words.append(paragraph_words)
                    all_tags.append(paragraph_tags)
                else:
                    all_words += paragraph_words
                    all_tags += paragraph_tags
                paragraph_words, paragraph_tags = [], []
        else:
            words = re.findall(r"[\w']+|[.,!?;:\"-_]", text)  # separate punctuation from words
            if label.startswith('cont-'):
                label = label[5:]
            if label in ARGUMENT_LABELS:
                tags = ['I'] * len(words)
                if i == 0 or not labeled_texts[i-1][0].startswith('cont-'):
                    tags[0] = 'B'
                paragraph_words += words
                paragraph_tags += tags
            elif label in NONARGUMENT_LABELS:
                tags = ['O'] * len(words)
                paragraph_words += words
                paragraph_tags += tags
            else:
                print('missing:', label)
    return all_words, all_tags

if __name__ == '__main__':
    create_processed_files('essay')
    # all_words, all_tags = convert_to_words_and_tags('../1.webis_editorials/test-data.txt', 'paragraph')
    # for paragraph_words, paragraph_tags in zip(all_words, all_tags):
    #     print('NEW PARAGRAPH')
    #     for w, t in zip(paragraph_words, paragraph_tags):
    #         print(w,t)