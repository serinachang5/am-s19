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
    labeled_texts = []
    for i in range(1, len(lines)):  # traverse backwards, ignore first line b/c title
        line = lines[i*-1]
        elements = line.strip().split('\t')
        if len(elements) >= 3:
            label, text = elements[1], elements[2]
            if label in ARGUMENT_LABELS:
                labeled_texts.insert(0, ('I', text))
            elif label in NONARGUMENT_LABELS:
                labeled_texts.insert(0, ('O', text))
            elif label == 'par-sep':
                labeled_texts.insert(0, ('SEP', ''))
            elif label == 'continued':
                assert(len(labeled_texts) > 0)
                subsequent_label = labeled_texts[0][0]
                labeled_texts.insert(0, (subsequent_label, text))
            else:
                print('Non-covered label:', label)
        elif len(elements) == 2 and elements[1] == 'par-sep':
            labeled_texts.insert(0, ('SEP', ''))
        else:
            print(elements)
    all_words, all_tags = [], []
    paragraph_words, paragraph_tags = [], []
    for i, (label, text) in enumerate(labeled_texts):
        if label == 'SEP' and len(paragraph_words) > 0:
            if level == 'paragraph':
                all_words.append(paragraph_words)
                all_tags.append(paragraph_tags)
            else:
                all_words += paragraph_words
                all_tags += paragraph_tags
            paragraph_words, paragraph_tags = [], []
        else:
            words = re.findall(r"[\w']+|[.,!?;:\"-_]", text)  # separate punctuation from words
            if len(words) > 0:
                tags = [label] * len(words)
                if label == 'I' and i > 0 and labeled_texts[i-1][0] != 'I':
                    tags[0] = 'B'  # beginning of inside
                paragraph_words += words
                paragraph_tags += tags
            else:
                print(txt_fn, i, text)
    return all_words, all_tags

if __name__ == '__main__':
    create_processed_files('paragraph')