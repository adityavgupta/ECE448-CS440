import collections

tagset = {'NOUN', 'VERB', 'ADJ', 'ADV',
          'PRON', 'DET', 'IN', 'NUM',
          'PART', 'UH', 'X', 'MODAL',
          'CONJ', 'PERIOD', 'PUNCT', 'TO'}

def evaluate_accuracies(predicted_sentences, tag_sentences):
    """
    :param predicted_sentences:
    :param tag_sentences:
    :return: (Accuracy, correct word-tag counter, wrong word-tag counter)
    """
    assert len(predicted_sentences) == len(tag_sentences)

    correct_wordtagcounter = {}
    wrong_wordtagcounter = {}
    correct = 0
    wrong = 0
    for pred_sentence, tag_sentence in zip(predicted_sentences, tag_sentences):
        assert len(pred_sentence) == len(tag_sentence)
        for pred_wordtag, real_wordtag in zip(pred_sentence, tag_sentence):
            #assert pred_wordtag[0] == real_wordtag[0]
            word = pred_wordtag[0]
            #print(pred_wordtag,real_wordtag)
            if pred_wordtag[1] == real_wordtag[1]:
                if word not in correct_wordtagcounter.keys():
                    correct_wordtagcounter[word] = collections.Counter()
                correct_wordtagcounter[word][real_wordtag[1]] += 1
                correct += 1
            else:
                if word not in wrong_wordtagcounter.keys():
                    wrong_wordtagcounter[word] = collections.Counter()
                wrong_wordtagcounter[word][real_wordtag[1]] += 1
                wrong += 1

    accuracy = correct / (correct + wrong)

    return accuracy, correct_wordtagcounter, wrong_wordtagcounter


def specialword_accuracies(train_sentences, predicted_sentences, tag_sentences):
    """
    :param train_sentences:
    :param predicted_sentences:
    :param tag_sentences:
    :return: Accuracy on words with multiple tags, and accuracy on words that do not occur in the training sentences
    """
    seen_words, words_with_multitags_set = get_word_tag_statistics(train_sentences)
    multitags_correct = 0
    multitags_wrong = 0
    unseen_correct = 0
    unseen_wrong = 0
    for i in range(len(predicted_sentences)):
        for j in range(len(predicted_sentences[i])):
            word = tag_sentences[i][j][0]
            tag = tag_sentences[i][j][1]

            if predicted_sentences[i][j][1] == tag:
                if word in words_with_multitags_set:
                    multitags_correct += 1
                if word not in seen_words:
                    unseen_correct += 1
            else:
                if word in words_with_multitags_set:
                    multitags_wrong += 1
                if word not in seen_words:
                    unseen_wrong += 1
    multitag_accuracy = multitags_correct / (multitags_correct + multitags_wrong)
    total_unseen = unseen_correct + unseen_wrong
    unseen_accuracy = unseen_correct / total_unseen if total_unseen > 0 else 0

    return multitag_accuracy, unseen_accuracy


def topk_wordtagcounter(wordtagcounter, k):
    top_items = sorted(wordtagcounter.items(), key=lambda item: sum(item[1].values()), reverse=True)[:k]
    top_items = list(map(lambda item: (item[0], dict(item[1])), top_items))
    return top_items


def load_dataset(data_file):
    sentences = []
    if data_file=='cleaner-data/pentree-training.txt' or data_file=='cleaner-data/pentree-test.txt':
        tagsets=hidden_tagset
    else:
        tagsets=tagset
    with open(data_file, 'r', encoding='UTF-8') as f:
        for line in f:
            sentence=[]
            raw = line.split()
            for pair in raw:
                splitted = pair.split('=')
                if (len(splitted) < 2):
                    continue
                else:
                    # find tag
                    if (splitted[-1] not in tagset):
                        tag = 'X'
                    else:
                        tag = splitted[-1]

                    # find word
                    word = splitted[0]
                    for element in splitted[1:-1]:
                        word += '/' + element
                    sentence.append((word.lower(), tag))
            sentences.append(sentence)
    return sentences

def strip_tags(sentences):
    '''
    Strip tags
    input:  list of sentences
            each sentence is a list of (word,tag) pairs
    output: list of sentences
            each sentence is a list of words (no tags)
    '''

    sentences_without_tags = []

    for sentence in sentences:
        sentence_without_tags = []
        for i in range(len(sentence)):
            pair = sentence[i]
            sentence_without_tags.append(pair[0])
        sentences_without_tags.append(sentence_without_tags)

    return sentences_without_tags


def get_word_tag_statistics(data_set):
    # get set of all seen words and set of words with multitags
    word_tags = collections.defaultdict(lambda: set())
    word_set = set()
    for sentence in data_set:
        for word, tag in sentence:
            word_tags[word].add(tag)
            word_set.add(word)
    return word_set, set(map(lambda elem: elem[0], filter(lambda elem: len(elem[1]) > 1, word_tags.items())))
