import argparse
import sys

from viterbi import viterbi_p1, viterbi_p2, baseline
from extra import extra
import utils

"""
This file contains the main application that is run for this MP.
"""


def main(args):
    print("Loading dataset...")
    train_set = utils.load_dataset(args.training_file)
    test_set = utils.load_dataset(args.test_file)
    print("Loaded dataset")
    print()

    for algorithm, name in zip([baseline, viterbi_p1, viterbi_p2, extra], ['Baseline', 'Viterbi_p1', 'Viterbi_p2', 'extra']):
        print("Running {}...".format(name))
        testtag_predictions = algorithm(train_set, utils.strip_tags(test_set))
        baseline_acc, correct_wordtagcounter, wrong_wordtagcounter = utils.evaluate_accuracies(test_set,
                                                                                               testtag_predictions)
        multitags_acc, unseen_acc, = utils.specialword_accuracies(train_set, test_set, testtag_predictions)

        print("Accuracy: {:.2f}%".format(baseline_acc * 100))
        print("\tTop K Wrong Word-Tag Predictions: {}".format(utils.topk_wordtagcounter(wrong_wordtagcounter, k=4)))
        print("\tTop K Correct Word-Tag Predictions: {}".format(utils.topk_wordtagcounter(correct_wordtagcounter, k=4)))
        print("\tMultitags Accuracy: {:.2f}%".format(multitags_acc * 100))
        print("\tUnseen words Accuracy: {:.2f}%".format(unseen_acc * 100))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP4 HMM')
    parser.add_argument('--train', dest='training_file', type=str,
                        help='the file of the training data')
    parser.add_argument('--test', dest='test_file', type=str,
                        help='the file of the testing data')
    args = parser.parse_args()
    if args.training_file == None or args.test_file == None:
        sys.exit('You must specify training file and testing file!')

    main(args)
