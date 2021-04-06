#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import os
import traceback

# Import student submission
import submission
import pickle, torch, json
import torch.nn as nn
from run import compute_corpus_level_bleu_score as bleu

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0

H_BATCH_SIZE = 5
H_WORD_EMBEDDING_SIZE = 7
H_HIDDEN_SIZE = 128
H_DROPOUT_RATE = 0.3

G_BATCH_SIZE = 50
G_HIDDEN_SIZE = 128
G_WORD_EMBEDDING_SIZE = 20

BATCH_SIZE_2B = 5
HIDDEN_SIZE_2B = 7
CHAR_EMBEDDING_SIZE_2B = 20
MAX_WORD_LENGTH_2B = 21

BATCH_SIZE_2C = 50
HIDDEN_SIZE_2C = 128
CHAR_EMBEDDING_SIZE_2C = 20
EMBED_SIZE_2C = 31
MAX_WORD_LENGTH_2C = 21

BATCH_SIZE_2D = 5
CHAR_EMBED_SIZE_2D = 3
HIDDEN_SIZE_2D = 3
DROPOUT_RATE_2D = 0.0

class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

def print_hyper_params(
        vocab,
        batch_size=H_BATCH_SIZE,
        word_embedding_size=H_WORD_EMBEDDING_SIZE,
        hidden_size=H_HIDDEN_SIZE,
        dropout_rate=H_DROPOUT_RATE,
        nmt_model=False
):
    print("\tSetting batch size to: {}".format(batch_size))
    if nmt_model:
        print("Initializing Student & Solution NMT Models with following hyperparams:")
        print("\tWord Embedding Size: {}".format(word_embedding_size))
        print("\tHidden Size: {}".format(hidden_size))
        print("\tDropout Rate: {}".format(H_DROPOUT_RATE))
        print("\tWord-level source vocab has size: {}".format(len(vocab.src)))
        print("\tWord-level target vocab has size: {}".format(len(vocab.tgt)))
    else:
        print("Initializing Student & Solution ModelEmbeddings Models with following params:")
        print("\tWord Embedding Size: {}".format(word_embedding_size))
        print("\tWord-level source vocab has size: {}".format(len(vocab.src)))
    print("-" * 40)


def reinitialize_layers_saved_files(model, student=True, init_with_bias=True):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """

    def init_weights(m):
        global STUDENT_CNN_HAS_BIAS
        if type(m) == nn.Linear:
            m.weight.data = torch.load('./outputs/linear_layer_weights_soln.pkl')
            m.bias.data = torch.load('./outputs/linear_layer_bias_soln.pkl')
        elif type(m) == nn.Embedding:
            m.weight.data = torch.load('./outputs/char_embedding_soln.pkl')
        elif type(m) == nn.Conv1d:
            m.weight.data = torch.load('./outputs/conv_weights_soln.pkl')
            if student:
                if m.bias is None:
                    STUDENT_CNN_HAS_BIAS = False
                else:
                    STUDENT_CNN_HAS_BIAS = True
                    m.bias.data = torch.load('./outputs/conv_bias_soln.pkl')
            else:
                if init_with_bias:
                    m.bias.data = torch.load('./outputs/conv_bias_soln.pkl')
                else:
                    m.bias.data.fill_(0)

    with torch.no_grad():
        model.apply(init_weights)

def save_files(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """

    def save(m):
        if type(m) == nn.Linear:
            torch.save(m.weight.data, './outputs/linear_layer_weights_soln.pkl')
            torch.save(m.bias.data, './outputs/linear_layer_bias_soln.pkl')
        elif type(m) == nn.Embedding:
            torch.save(m.weight.data, './outputs/char_embedding_soln.pkl')
        elif type(m) == nn.Conv1d:
            torch.save(m.weight.data, './outputs/conv_weights_soln.pkl')
            torch.save(m.bias.data, './outputs/conv_bias_soln.pkl')

    with torch.no_grad():
        model.apply(save)

def print_hyper_params_g(
        vocab,
        batch_size=G_BATCH_SIZE,
        word_embedding_size=G_WORD_EMBEDDING_SIZE,
        hidden_size=G_HIDDEN_SIZE,
):
    print("Hyperparameters:")
    print("\tBatch Size: {}".format(batch_size))
    print("\tWord Embedding Size: {}".format(word_embedding_size))
    print("\tHidden Size: {}".format(hidden_size))
    print("\tSource Vocab Length: {}".format(len(vocab.src)))
    print("\tTarget Vocab Length: {}".format(len(vocab.tgt)))
    print("-" * 40)


def init_seeds_g():
    random.seed(35436)
    np.random.seed(4355)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

def setup_2a(self):
    self.batch_size = 5
    self.char_embed_size = 3
    self.hidden_size = 3

    self.vocab = DummyVocab()
    self.vocab_size = len(self.vocab.char2id)
    self.padding_idx = self.vocab.char2id['<pad>']

    print(
        'Initializing CharDecoder with hidden_size=%s, char_embedding_size=%s, target_vocab from sanity_check_en_es_data/char_vocab_sanity_check.json' % (
        self.hidden_size, self.char_embed_size))
    print('Using batch_size=%i' % self.batch_size)

    print('Initializing student\'s CharDecoder...')
    self.student_result = submission.CharDecoder(
        hidden_size=self.hidden_size,
        char_embedding_size=self.char_embed_size,
        target_vocab=self.vocab)

def print_hyper_params_2b(
        vocab,
        batch_size=BATCH_SIZE_2B,
        char_embedding_size=CHAR_EMBEDDING_SIZE_2B,
        hidden_size=HIDDEN_SIZE_2B,
        running_forward=False,
        target_chars=None,
        decoder_init_hidden_state=None,
):
    print("Hyperparameters:")
    print("\tBatch Size: {}".format(batch_size))
    print("Initializing Student & Solution CharDecoder with following parameters:")
    print("\tHidden Size: {}".format(hidden_size))
    print("\tCharacter Embedding Size: {}".format(char_embedding_size))
    if hasattr(vocab, 'tgt'):
        # All other checks
        print("\tTarget Vocab of Length: {}".format(len(vocab.tgt)))
    else:
        # Sanity check
        print("\tTarget Vocab of Length: {}".format(len(vocab.char2id)))
    if running_forward:
        print("Running Model Forward with:")
        print("\tTarget Chars: {}".format(target_chars))
        print("\tDecoder Initial Hidden State: {}".format(decoder_init_hidden_state))
    print("-" * 40)


def weight_copy(soln, student):
    """ Copy the weights of the solution's character decoder into the student's
        character decoder.
    """
    print(
        'Copying model solution\'s target_vocab, charDecoder, char_output_projection and decoderCharEmb to student\'s CharDecoder')
    student.target_vocab = soln.target_vocab

    # if the student set batch_first=True (non-standard) in their self.charDecoder, then we need to account for that
    assert not soln.charDecoder.batch_first
    if not student.charDecoder.batch_first:
        student.charDecoder = soln.charDecoder
    else:  # if student set batch_first=True then we just copy the weights so the batch_first setting stays the same
        student.charDecoder.weight_hh_l0 = soln.charDecoder.weight_hh_l0
        student.charDecoder.weight_ih_l0 = soln.charDecoder.weight_ih_l0
        student.charDecoder.bias_hh_l0 = soln.charDecoder.bias_hh_l0
        student.charDecoder.bias_ih_l0 = soln.charDecoder.bias_ih_l0

    student.char_output_projection = soln.char_output_projection
    student.decoderCharEmb = soln.decoderCharEmb

def print_hyper_params_2c(
        vocab,
        sequence_length,
        sequence,
        batch_size=BATCH_SIZE_2C,
        hidden_size=HIDDEN_SIZE_2C,
        char_embedding_size=CHAR_EMBEDDING_SIZE_2C,
        running_forward=False,
):
    print("Batch Size: {}".format(batch_size))
    print("Character Embedding Size: {}".format(char_embedding_size))
    print("Hidden Size: {}".format(hidden_size))
    print("Character-level target vocab with size: {}".format(len(vocab.char2id)))
    if running_forward:
        print('Input to train_forward is:')
        print("\t Sequence of Length: {}".format(sequence_length))
        print("\t Sequence: {}".format(sequence))
    print("-" * 40)

#########
# TESTS #
#########

class Test_1a(GradedTestCase):
    @graded()
    def test_0(self):
        """1a-0-basic:  Sanity check for words2charindices function."""
        vocab = submission.VocabEntry()

        print('Running test on small list of sentences')
        sentences = [["a", "b", "c?"], ["~d~", "c", "b", "a"]]
        small_ind = vocab.words2charindices(sentences)
        small_ind_gold = [[[1, 30, 2], [1, 31, 2], [1, 32, 70, 2]],
                          [[1, 85, 33, 85, 2], [1, 32, 2], [1, 31, 2], [1, 30, 2]]]
        self.assertEqual(small_ind, small_ind_gold, "small test resulted in indices list {:}, expected {:}".format(small_ind, small_ind_gold))

        print('Running test on large list of sentences')
        tgt_sents = [
            ['<s>', "Let's", 'start', 'by', 'thinking', 'about', 'the', 'member', 'countries', 'of', 'the', 'OECD,', 'or',
             'the', 'Organization', 'of', 'Economic', 'Cooperation', 'and', 'Development.', '</s>'],
            ['<s>', 'In', 'the', 'case', 'of', 'gun', 'control,', 'we', 'really', 'underestimated', 'our', 'opponents.',
             '</s>'],
            ['<s>', 'Let', 'me', 'share', 'with', 'those', 'of', 'you', 'here', 'in', 'the', 'first', 'row.', '</s>'],
            ['<s>', 'It', 'suggests', 'that', 'we', 'care', 'about', 'the', 'fight,', 'about', 'the', 'challenge.', '</s>'],
            ['<s>', 'A', 'lot', 'of', 'numbers', 'there.', 'A', 'lot', 'of', 'numbers.', '</s>']]
        tgt_ind = vocab.words2charindices(tgt_sents)
        tgt_ind_gold = pickle.load(open('./sanity_check_en_es_data/1a_tgt.pkl', 'rb'))
        self.assertEqual(tgt_ind, tgt_ind_gold, "target vocab test resulted in indices list {:}, expected {:}".format(tgt_ind,
                                                                                                                    tgt_ind_gold))

    @graded(is_hidden=True, timeout=15)
    def test_1(self):
        """1a-1-hidden:  test output of words2charindices (public)"""
        # Set Seeds
        random.seed(35436)
        np.random.seed(4355)

        # Create Inputs
        # Load vocabulary
        stu_vocab = submission.VocabEntry()
        soln_vocab = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).VocabEntry()
        sentences = [['<s>', "Let's", 'start', 'by', 'thinking', 'about', 'the', 'member', 'countries', 'of', 'the', 'OECD,', 'or', 'the', 'Organization', 'of', 'Economic', 'Cooperation', 'and', 'Development.', '</s>'], ['<s>', 'In', 'the', 'case', 'of', 'gun', 'control,', 'we', 'really', 'underestimated', 'our', 'opponents.', '</s>'], ['<s>', 'Let', 'me', 'share', 'with', 'those', 'of', 'you', 'here', 'in', 'the', 'first', 'row.', '</s>'], ['<s>', 'It', 'suggests', 'that', 'we', 'care', 'about', 'the', 'fight,', 'about', 'the', 'challenge.', '</s>'], ['<s>', 'A', 'lot', 'of', 'numbers', 'there.', 'A', 'lot', 'of', 'numbers.', '</s>']]
        print("Test Input: {}".format(sentences))
        student_outputs = stu_vocab.words2charindices(sentences)
        soln_outputs = soln_vocab.words2charindices(sentences)
        print("Student Output: {}".format(student_outputs))
        print("Solution Output: {}".format(soln_outputs))
        self.assertEqual(student_outputs, soln_outputs)

class Test_1b(GradedTestCase):
    @graded()
    def test_0(self):
        """1b-0-basic:  Sanity check for pad_sents_char() function. """
        vocab = submission.VocabEntry()

        print("Running test on a list of sentences")
        sentences = [['Human:', 'What', 'do', 'we', 'want?'], ['Computer:', 'Natural', 'language', 'processing!'],
                     ['Human:', 'When', 'do', 'we', 'want', 'it?'], ['Computer:', 'When', 'do', 'we', 'want', 'what?']]
        word_ids = vocab.words2charindices(sentences)

        padded_sentences = submission.pad_sents_char(word_ids, 0)
        gold_padded_sentences = torch.load('./sanity_check_en_es_data/gold_padded_sentences.pkl')
        self.assertEqual(padded_sentences, gold_padded_sentences, "Sentence padding is incorrect: it should be:\n {} but is:\n{}".format(
            gold_padded_sentences, padded_sentences))

    @graded(is_hidden=True, timeout=30)
    def test_1(self):
        """1b-1-hidden:  Test output of pad_sents_char (public)"""
        # Set Seeds
        random.seed(35436)
        np.random.seed(4355)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Create Inputs
        vocab = submission.VocabEntry()
        # Prep for Test
        padded_sentences_result = False
        sentences = [['Human:', 'What', 'do', 'we', 'want?'], ['Computer:', 'Natural', 'language', 'processing!'], ['Human:', 'When', 'do', 'we', 'want', 'it?'], ['Computer:', 'When', 'do', 'we', 'want', 'what?']]
        word_ids = vocab.words2charindices(sentences)
        padded_sentences = submission.pad_sents_char(word_ids, 0)  # Padding token is 0
        gold_padded_sentences = torch.load('./sanity_check_en_es_data/gold_padded_sentences.pkl')

        print("Input: {}".format(sentences))
        print("Student Output: {}".format(padded_sentences))
        print("Solution Output: {}".format(gold_padded_sentences))

        self.assertEqual(padded_sentences, gold_padded_sentences)

class Test_1c(GradedTestCase):
    @graded(is_hidden=True, timeout=20)
    def test_0(self):
        """1c-0-hidden:  Test shape of output of to_input_tensor_char (hidden)"""
        # Load vocabulary
        stu_vocab = submission.VocabEntry()
        soln_vocab = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).VocabEntry()
        sentences = [
            ['<s>', "Let's", 'start', 'by', 'thinking', 'about', 'the', 'member', 'countries', 'of', 'the', 'OECD,', 'or',
             'the', 'Organization', 'of', 'Economic', 'Cooperation', 'and', 'Development.', '</s>'],
            ['<s>', 'In', 'the', 'case', 'of', 'gun', 'control,', 'we', 'really', 'underestimated', 'our', 'opponents.',
             '</s>'],
            ['<s>', 'Let', 'me', 'share', 'with', 'those', 'of', 'you', 'here', 'in', 'the', 'first', 'row.', '</s>'],
            ['<s>', 'It', 'suggests', 'that', 'we', 'care', 'about', 'the', 'fight,', 'about', 'the', 'challenge.', '</s>'],
            ['<s>', 'A', 'lot', 'of', 'numbers', 'there.', 'A', 'lot', 'of', 'numbers.', '</s>']]
        print('Input: {}'.format(sentences))
        student_output = stu_vocab.to_input_tensor_char(sentences, device=torch.device('cpu'))
        soln_output = soln_vocab.to_input_tensor_char(sentences, device=torch.device('cpu'))
        print('Student Output Shape: {}'.format(student_output.shape))
        print('Solution Output Shape: {}'.format(soln_output.shape))
        self.assertEqual(list(student_output.shape), list(soln_output.shape))

class Test_1f(GradedTestCase):
    def setUp(self):
        random.seed(35436)
        np.random.seed(4355)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Initialize Model Embeddings
        src_sents = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus('./en_es_data/train.es', source='src')
        tgt_sents = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus('./en_es_data/train.en', source='tgt')
        self.vocab = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).Vocab.build(src_sents, tgt_sents, 50000, 2)

        self.student_embeddings = submission.ModelEmbeddings(H_WORD_EMBEDDING_SIZE, self.vocab.src)
        self.soln_embeddings = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).ModelEmbeddings(H_WORD_EMBEDDING_SIZE, self.vocab.src)
        self.student_embeddings.eval()
        self.soln_embeddings.eval()

        # Read Data
        train_data_src = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus('./en_es_data/data.es', 'src')
        train_data_tgt = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus('./en_es_data/data.en', 'tgt')
        train_data = list(zip(train_data_src, train_data_tgt))
        for src_sents, tgt_sents in self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).batch_iter(train_data, batch_size=H_BATCH_SIZE, shuffle=False):
            src_sents = src_sents
            tgt_sents = tgt_sents
            break
        self.src_padded_chars = self.vocab.src.to_input_tensor_char(src_sents, device=torch.device('cpu'))

    @graded()
    def test_0(self):
        """1f-0-basic:  Sanity check for model_embeddings.py.  Also basic shape check"""
        vocab = submission.Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

        # Create NMT Model
        model = submission.NMT(
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            dropout_rate=DROPOUT_RATE,
            vocab=vocab)
        sentence_length = 10
        max_word_length = 21
        inpt = torch.zeros(sentence_length, BATCH_SIZE, max_word_length, dtype=torch.long)
        ME_source = model.model_embeddings_source
        output = ME_source.forward(inpt)
        output_expected_size = [sentence_length, BATCH_SIZE, EMBED_SIZE]
        self.assertEqual(list(
            output.size()), output_expected_size, "output shape is incorrect: it should be:\n {} but is:\n{}".format(
            output_expected_size, list(output.size())))

    @graded(is_hidden=True)
    def test_1(self):
        """1f-1-hidden:  Test shape of output of ModelEmbeddings.forward"""
        BATCH_SIZE = 5
        WORD_EMBEDDING_SIZE = 3
        HIDDEN_SIZE = 3
        DROPOUT_RATE = 0.3
        SENTENCE_LENGTH = 10
        MAX_WORD_LENGTH = 21

        print_hyper_params(
            self.vocab,
            batch_size=BATCH_SIZE,
            word_embedding_size=WORD_EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            nmt_model=True
        )
        model = submission.NMT(
            embed_size=WORD_EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            dropout_rate=DROPOUT_RATE,
            vocab=self.vocab)
        model.eval()

        print("Sentence Length: {}".format(SENTENCE_LENGTH))
        print("Max Word Length: {}".format(MAX_WORD_LENGTH))
        print("Input of Dimensions: ({}, {}, {})".format(SENTENCE_LENGTH, BATCH_SIZE, MAX_WORD_LENGTH))
        inpt = torch.zeros(SENTENCE_LENGTH, BATCH_SIZE, MAX_WORD_LENGTH, dtype=torch.long)

        output = model.model_embeddings_source.forward(inpt)
        output_expected_size = [SENTENCE_LENGTH, BATCH_SIZE, WORD_EMBEDDING_SIZE]
        print("ModelEmbeddings.forward() expected output size:{}".format(output_expected_size))
        print("ModelEmbeddings.forward() output size:{}".format(list(output.size())))
        self.assertTrue(list(output.size()) == output_expected_size)

    @graded(is_hidden=True)
    def test_2(self):
        """1f-2-hidden:  Test shape of weights in Character Embeddings in ModelEmbeddings"""
        result = True
        print_hyper_params(
            self.vocab,
        )

        def check(layer):
            if type(layer) == nn.Embedding:
                print("Student ModelEmbeddings Vocab Size: {}".format(layer.weight.shape[0]))
                print("Student ModelEmbeddings Character Embedding Size Size: {}".format(layer.weight.shape[1]))
                if (
                        layer.weight.shape[0] != len(self.vocab.src.char2id) or
                        layer.weight.shape[1] != 50
                ):
                    result = False

        self.student_embeddings.apply(check)
        if (result):
            print("pass!")
        else:
            print("fail!")
        self.assertTrue(result)

    @graded(is_hidden=True)
    def test_3(self):
        """1f-3-hidden:  Test shape of weights in Linear Layers in Highway"""
        result = True
        print_hyper_params(
            self.vocab
        )
        print("Highway's Linear Layers' `in_features` and `out_features` should be: 20")
        print("Highway's Linear Layers' `bias` size should be: 20")

        def check(layer):
            if type(layer) == nn.Linear:
                print("Linear Layer's `in_features: {}".format(layer.weight.shape[0]))
                print("Linear Layer's `out_features: {}".format(layer.weight.shape[1]))
                print("Linear Layer's `bias` size: {}".format(layer.bias.shape[0]))
                if (
                        layer.weight.shape[0] != 20 or
                        layer.weight.shape[1] != 20 or
                        layer.bias.shape[0] != 20
                ):
                    result = False

        self.student_embeddings.apply(check)
        if (result):
            print("pass!")
        else:
            print("fail!")
        self.assertTrue(result)

    @graded(is_hidden=True)
    def test_4(self):
        """1f-4-hidden:  Test shape of Conv1D weights in CNN layer"""
        result = True
        print_hyper_params(
            self.vocab
        )
        print("`in_channels` should be: 50")
        print("`out_channels` should Be: 20")
        print("`kernel_size` should be: 5")
        print("`bias` size should be: 20")

        def check(layer):
            if type(layer) == nn.Conv1d:
                print("Conv1D `in_channels`: {}".format(layer.weight.shape[1]))
                print("Conv1D `out_channels`: {}".format(layer.weight.shape[0]))
                print("Conv1D `kernel_size`: {}".format(layer.weight.shape[2]))
                print("Conv1D `bias` size: {}".format(layer.bias.shape[0]))
                if (
                        layer.weight.shape[1] != 50 or
                        layer.weight.shape[0] != 20 or
                        layer.weight.shape[2] != 5 or
                        layer.bias.shape[0] != 20
                ):
                    result = False

        self.student_embeddings.apply(check)
        if (result):
            print("pass!")
        else:
            print("fail!")
        self.assertTrue(result)

    @graded(is_hidden=True)
    def test_5(self):
        """1f-5-hidden:  Test output of ModelEmbeddings.forward"""
        result = False

        print_hyper_params(
            self.vocab
        )

        print("Using same weights in student code and solution code.")
        with torch.no_grad():
            reinitialize_layers_saved_files(self.student_embeddings)
            print('-' * 80)
            print(STUDENT_CNN_HAS_BIAS)
            reinitialize_layers_saved_files(self.soln_embeddings, False, STUDENT_CNN_HAS_BIAS)
            print("Running solution\'s ModelEmbeddings.forward()...")
            soln_output = self.soln_embeddings(self.src_padded_chars)
            try:
                print("Running student\'s ModelEmbeddings.forward()...")
                student_output = self.student_embeddings(self.src_padded_chars)
            except Exception as e:
                traceback.print_exc()
                raise (e)

        print("Solution Output:")
        print(soln_output)
        print("Student Output:")
        print(student_output)
        if np.allclose(soln_output.numpy(), student_output.numpy(), atol=1e-3):
            result = True
            print('Passed!')
        else:
            print('Failed!')
        self.assertTrue(result)

class Test_1g(GradedTestCase):
    def setUp(self):
        # Read Data
        init_seeds_g()
        self.vocab = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).Vocab.load('./vocabs/vocab_test.json')
        train_data_src = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus('./en_es_data/data.es', 'src')
        train_data_tgt = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus('./en_es_data/data.en', 'tgt')
        train_data = list(zip(train_data_src, train_data_tgt))
        for src_sents, tgt_sents in self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).batch_iter(train_data, batch_size=BATCH_SIZE, shuffle=False):
            self.src_sents = src_sents
            self.tgt_sents = tgt_sents
            break

    @graded(is_hidden=True)
    def test_0(self):
        """1g-0-hidden:  Test output of NMT.forward"""
        print_hyper_params_g(vocab=self.vocab)
        with torch.no_grad():
            # Soln NMT
            print("Initialize solutions\'s NMT instance with output embedding size {} and hidden size {}.".format(
                G_WORD_EMBEDDING_SIZE, G_HIDDEN_SIZE))
            init_seeds_g()
            soln_nmt = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).NMT(G_WORD_EMBEDDING_SIZE, G_HIDDEN_SIZE, self.vocab, no_char_decoder=True)
            soln_nmt.eval()
            soln_output = soln_nmt(self.src_sents, self.tgt_sents)

            # Student NMT
            print("Initialize student\'s NMT instance with output embedding size {} and hidden size {}.".format(
                G_WORD_EMBEDDING_SIZE, G_HIDDEN_SIZE))
            init_seeds_g()
            student_nmt = submission.NMT(G_WORD_EMBEDDING_SIZE, G_HIDDEN_SIZE, self.vocab, no_char_decoder=True)
            print("Substitute student\'s ModelEmbeddings instance with solution\'s ModelEmbeddings")
            student_nmt.model_embeddings_source = soln_nmt.model_embeddings_source
            student_nmt.model_embeddings_target = soln_nmt.model_embeddings_target

            print("Copying other weights from solution to student")
            student_nmt.encoder = soln_nmt.encoder
            student_nmt.decoder = soln_nmt.decoder
            student_nmt.h_projection = soln_nmt.h_projection
            student_nmt.c_projection = soln_nmt.c_projection
            student_nmt.att_projection = soln_nmt.att_projection
            student_nmt.combined_output_projection = soln_nmt.combined_output_projection
            student_nmt.target_vocab_projection = soln_nmt.target_vocab_projection
            student_nmt.dropout = soln_nmt.dropout
            student_nmt.eval()

            print('Running student\'s NMT.forward() function...')

            try:
                student_output = student_nmt(self.src_sents, self.tgt_sents)
            except Exception as e:
                traceback.print_exc()
                raise (e)

            print("Student output from NMT.forward():")
            print(student_output.numpy())
            print("Solution output from NMT.forward():")
            print(soln_output.numpy())

            self.assertLessEqual(abs(soln_output.numpy() - student_output.numpy()), 0.004)

class Test_1h(GradedTestCase):
    @graded(is_hidden=True)
    def test_0(self):
        """1h-0-hidden:  BLEU score on tiny test set is over 99"""
        hyp_file = os.path.join('./outputs', 'test_outputs_local_q1_soln.txt')
        ref_file = os.path.join('./en_es_data', 'test_tiny.en')

        if os.path.exists(os.path.join('./submission', 'test_outputs_local_q1_soln.txt')):
            hyp_file = os.path.join('./submission', 'test_outputs_local_q1_soln.txt')

        self.assertTrue(os.path.exists(hyp_file),
                        f'Output test file (outputs/test_outputs_local_q1_soln.txt) does not exist.')

        ref = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus(ref_file, source='tgt')
        hyp = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus(hyp_file, source='tgt')

        ref = [line[1:-1] for line in ref]  # remove <s>...</s>
        hyp = [line[1:-1] for line in hyp]  # remove <s>...</s>

        hyp = [submission.Hypothesis(value=line, score=1) for line in hyp]
        self.ref, self.hyp = ref, hyp

        score = bleu(self.ref, self.hyp) * 100
        print("BLEU " + str(score))
        self.assertGreaterEqual(score, 99.0, "Your BLEU score ({}) is below 99".format(score))


class Test_2a(GradedTestCase):
    @graded()
    def test_0(self):
        """2a-0-basic:  Sanity check for CharDecoder.__init__().  Also basic shape check"""
        char_vocab = DummyVocab()

        # Initialize CharDecoder
        decoder = submission.CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=EMBED_SIZE,
            target_vocab=char_vocab)
        self.assertEqual(
                decoder.charDecoder.input_size, EMBED_SIZE, "Input dimension is incorrect:\n it should be {} but is: {}".format(
            EMBED_SIZE, decoder.charDecoder.input_size))
        self.assertEqual(
                decoder.charDecoder.hidden_size, HIDDEN_SIZE, "Hidden dimension is incorrect:\n it should be {} but is: {}".format(
            HIDDEN_SIZE, decoder.charDecoder.hidden_size))
        self.assertEqual(
                decoder.char_output_projection.in_features, HIDDEN_SIZE, "Input dimension is incorrect:\n it should be {} but is: {}".format(
            HIDDEN_SIZE, decoder.char_output_projection.in_features))
        self.assertEqual(decoder.char_output_projection.out_features, len(
            char_vocab.char2id), "Output dimension is incorrect:\n it should be {} but is: {}".format(
            len(char_vocab.char2id), decoder.char_output_projection.out_features))
        self.assertEqual(decoder.decoderCharEmb.num_embeddings, len(
            char_vocab.char2id), "Number of embeddings is incorrect:\n it should be {} but is: {}".format(
            len(char_vocab.char2id), decoder.decoderCharEmb.num_embeddings))
        self.assertEqual(
                decoder.decoderCharEmb.embedding_dim, EMBED_SIZE, "Embedding dimension is incorrect:\n it should be {} but is: {}".format(
            EMBED_SIZE, decoder.decoderCharEmb.embedding_dim))

    @graded(is_hidden=True)
    def test_1(self):
        """2a-1-hidden:  Test shapes of initialized layers"""
        setup_2a(self)
        correct = False
        decoder = self.student_result

        if (
                decoder.charDecoder.input_size == self.char_embed_size and
                decoder.charDecoder.hidden_size == self.hidden_size and
                decoder.char_output_projection.in_features == self.hidden_size and
                decoder.char_output_projection.out_features == self.vocab_size and
                decoder.decoderCharEmb.num_embeddings == self.vocab_size and
                decoder.decoderCharEmb.embedding_dim == self.char_embed_size
        ):
            correct = True

        print('Expected charDecoder.input_size=%s, got %s' % (self.char_embed_size, decoder.charDecoder.input_size))
        print('Expected charDecoder.hidden_size=%s, got %s' % (self.hidden_size, decoder.charDecoder.hidden_size))
        print('Expected char_output_projection.in_features=%s, got %s' % (
        self.hidden_size, decoder.char_output_projection.in_features))
        print('Expected char_output_projection.out_features=%s, got %s' % (
        self.vocab_size, decoder.char_output_projection.out_features))
        print('Expected decoderCharEmb.num_embeddings=%s, got %s' % (
        self.vocab_size, decoder.decoderCharEmb.num_embeddings))
        print('Expected decoderCharEmb.embedding_dim=%s, got %s' % (
        self.char_embed_size, decoder.decoderCharEmb.embedding_dim))

        self.assertTrue(correct)

    @graded(is_hidden=True)
    def test_2(self):
        """2a-2-hidden:  Test correct initialization of self.target_vocab"""
        setup_2a(self)
        print('Testing student\'s initialization of target_vocab...')
        student = self.student_result.target_vocab
        print('Checking self.target_vocab is equal to the target_vocab passed in...')
        self.assertEqual(student, self.vocab,
                         'Failure: target_vocab should be equal to the target_vocab passed in.'
                         )

    @graded(is_hidden=True)
    def test_3(self):
        """2a-3-hidden:  Test correct initialization of self.charDecoder"""
        setup_2a(self)
        print('Testing student\'s initialization of charDecoder...')
        student = self.student_result.charDecoder
        print('Checking charDecoder is a nn.LSTM object...')
        self.assertIsInstance(student, nn.LSTM,
                              'Failure: charDecoder should be nn.LSTM object, but student\'s is type {}'.format(
                                  type(student))
                              )
        print('Checking charDecoder.input_size...')
        self.assertEqual(student.input_size, self.char_embed_size,
                         'Failure: charDecoder.input_size should be {}, but student\'s is {}'.format(
                             self.char_embed_size, student.input_size)
                         )
        print('Checking charDecoder.hidden_size...')
        self.assertEqual(student.hidden_size, self.hidden_size,
                         'Failure: charDecoder.hidden_size should be {}, but student\'s is {}'.format(self.hidden_size,
                                                                                                      student.hidden_size)
                         )
        print('Checking charDecoder.bias...')
        self.assertEqual(student.bias, True,
                         'Failure: charDecoder.bias should be True, but student\'s is {}'.format(student.bias)
                         )
        print(
            'Note: student\'s charDecoder set batch_first={}. Standard solution assumes batch_first=False (default setting).'.format(
                student.batch_first))

    @graded(is_hidden=True)
    def test_4(self):
        """2a-4-hidden:  Test correct initialization of self.char_output_projection (hidden)"""
        setup_2a(self)
        print('Testing student\'s initialization of char_output_projection...')
        student = self.student_result.char_output_projection
        print('Checking char_output_projection is a nn.Linear object...')
        self.assertIsInstance(student, nn.Linear,
                              'Failure: char_output_projection must be a nn.Linear object. Student\'s is type {}'.format(
                                  type(student))
                              )
        print('Checking char_output_projection.in_features...')
        self.assertEqual(student.in_features, self.hidden_size,
                         'Failure: char_output_projection.in_features should be {}. Student\'s is {}'.format(
                             self.hidden_size, student.in_features)
                         )
        print('Checking char_output_projection.out_features...')
        self.assertEqual(student.out_features, self.vocab_size,
                         'Failure: char_output_projection.out_features should be {}. Student\'s is {}'.format(
                             self.vocab_size, student.out_features)
                         )
        print('Checking char_output_projection.bias...')
        self.assertIsInstance(student.bias, torch.Tensor,
                              'Failure: char_output_projection.bias must be a torch.Tensor. Student\'s is type {}'.format(
                                  type(student.bias))
                              )
        print('Checking char_output_projection.bias shape...')
        self.assertEqual(student.bias.size()[0], self.vocab_size,
                         'Failure: char_output_projection.bias should be {}. Student\'s is {}'.format(self.vocab_size,
                                                                                                      student.bias.size()[
                                                                                                          0])
                         )

    @graded(is_hidden=True)
    def test_5(self):
        """2a-5-hidden:  Test correct initialization of self.decoderCharEmb (hidden)"""
        setup_2a(self)
        print('Testing student\'s initialization of self.decoderCharEmb...')
        student = self.student_result.decoderCharEmb
        print('Checking decoderCharEmb is a nn.Embedding object...')
        self.assertIsInstance(student, nn.Embedding,
                              'Failure: decoderCharEmb must be a nn.Embedding object, but student\'s is type {}'.format(
                                  str(type(student)))
                              )
        print('Checking decoderCharEmb.num_embeddings...')
        self.assertEqual(student.num_embeddings, self.vocab_size,
                         'Failure: decoderCharEmb.num_embeddings should be {}, but student\'s is {}'.format(
                             self.vocab_size, student.num_embeddings)
                         )
        print('Checking decoderCharEmb.embedding_dim...')
        self.assertEqual(student.embedding_dim, self.char_embed_size,
                         'Failure: decoderCharEmb.embedding_dim should be {}, but student\'s is {}'.format(
                             self.char_embed_size, student.embedding_dim)
                         )
        print('Checking decoderCharEmb.padding_idx...')
        self.assertEqual(student.padding_idx, self.padding_idx,
                         'Failure: decoderCharEmb.padding_idx should be {}, but student\'s is {}'.format(
                             self.padding_idx, student.padding_idx)
                         )

class Test_2b(GradedTestCase):
    def runForwardFull(self):
        """After setup, init student and solution, copy solution weights to student, then run both forward and save results"""

        # Initialize Models
        print('Initializing solution\'s CharDecoder...')
        soln = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).CharDecoder(
            hidden_size=HIDDEN_SIZE_2B,
            char_embedding_size=CHAR_EMBEDDING_SIZE_2B,
            target_vocab=self.vocab
        )

        print('Initializing student\'s CharDecoder...')
        student = submission.CharDecoder(
            hidden_size=HIDDEN_SIZE_2B,
            char_embedding_size=CHAR_EMBEDDING_SIZE_2B,
            target_vocab=self.vocab
        )

        # Run models forward
        weight_copy(soln, student)
        soln.eval()
        student.eval()
        self.dummy_dec_hidden = (
            torch.randn(1, 21, HIDDEN_SIZE_2B, dtype=torch.float),
            torch.randn(1, 21, HIDDEN_SIZE_2B, dtype=torch.float),
        )
        torch.manual_seed(20199)
        self.target_chars = torch.from_numpy(self.random_batch_char_sequence(BATCH_SIZE_2B))
        with torch.no_grad():
            try:
                print('Running student\'s forward function...')
                self.student_logits, self.student_dec_hidden = student(self.target_chars, self.dummy_dec_hidden)
                print('Running solution\'s forward function...')
                self.soln_logits, self.soln_dec_hidden = soln(self.target_chars, self.dummy_dec_hidden)
            except Exception as e:
                traceback.print_exc()
                raise (e)

    def random_char_sequence(self):
        sequence_length = np.random.randint(1, MAX_WORD_LENGTH_2B - 1)
        seq = np.zeros(MAX_WORD_LENGTH_2B, dtype=np.int64)
        seq[0] = 1
        for i in range(1, sequence_length + 1):
            seq[i] = np.random.choice(self.char_ids)
        seq[i] = 2
        return seq

    def random_batch_char_sequence(self, batch_size):
        return np.array([self.random_char_sequence() for b in range(batch_size)])

    def setUp(self):
        # Set Seeds
        random.seed(35436)
        np.random.seed(4355)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Create vocabulary
        class DummyVocab():
            def __init__(self):
                self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
                self.id2char = {id: char for char, id in self.char2id.items()}
                self.char_unk = self.char2id['<unk>']
                self.start_of_word = self.char2id["{"]
                self.end_of_word = self.char2id["}"]

        self.vocab = DummyVocab()
        self.char_ids = [4, 6, 7, 9, 11, 12, 16, 17, 19, 22, 23, 26]

    @graded()
    def test_0(self):
        """2b-0-basic:  Sanity check for CharDecoder.forward()"""
        char_vocab = DummyVocab()

        # Initialize CharDecoder
        decoder = submission.CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=EMBED_SIZE,
            target_vocab=char_vocab)
        sequence_length = 4
        inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
        logits, (dec_hidden1, dec_hidden2) = decoder.forward(inpt)
        logits_expected_size = [sequence_length, BATCH_SIZE, len(char_vocab.char2id)]
        dec_hidden_expected_size = [1, BATCH_SIZE, HIDDEN_SIZE]
        self.assertEqual(list(
            logits.size()), logits_expected_size, "Logits shape is incorrect:\n it should be {} but is:\n{}".format(
            logits_expected_size, list(logits.size())))
        self.assertEqual(list(
            dec_hidden1.size()), dec_hidden_expected_size, "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(
            dec_hidden_expected_size, list(dec_hidden1.size())))
        self.assertEqual(list(
            dec_hidden2.size()), dec_hidden_expected_size, "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(
            dec_hidden_expected_size, list(dec_hidden2.size())))

    @graded(is_hidden=True)
    def test_1(self):
        """2b-1-hidden:  Test logits returned by forward"""
        self.runForwardFull()
        result = False
        print_hyper_params_2b(
            self.vocab,
            running_forward=True,
            target_chars=self.target_chars,
            decoder_init_hidden_state=self.dummy_dec_hidden
        )
        print("Checking logits...")
        print("Student logits: {}".format(self.student_logits))
        print("Solution logits: {}".format(self.soln_logits))
        if np.allclose(self.student_logits.numpy(), self.soln_logits.numpy(), atol=1e-4):
            result = True
            print('Passed logits test')
        else:
            print('Failed logits test')
        self.assertTrue(result)

    @graded(is_hidden=True)
    def test_2(self):
        """2b-2-hidden:  Test decoder hidden state returned by forward"""
        self.runForwardFull()
        result = False
        print_hyper_params_2b(
            self.vocab,
            running_forward=True,
            target_chars=self.target_chars,
            decoder_init_hidden_state=self.dummy_dec_hidden
        )
        print("Checking decoder hidden state...")
        print("Student decoder hidden state: {}".format(self.student_dec_hidden[0]))
        print("Solution decoder hidden state: {}".format(self.soln_dec_hidden[0]))
        if np.allclose(self.student_dec_hidden[0].numpy(), self.soln_dec_hidden[0].numpy(), atol=1e-4):
            result = True
            print('Passed decoder hidden state test')
        else:
            print('Failed decoder hidden state test')
        self.assertTrue(result)

    @graded(is_hidden=True)
    def test_3(self):
        """2b-3-hidden:  Test decoder cell state returned by forward"""
        self.runForwardFull()
        result = False
        print_hyper_params_2b(
            self.vocab,
            running_forward=True,
            target_chars=self.target_chars,
            decoder_init_hidden_state=self.dummy_dec_hidden
        )
        print("Checking decoder hidden cell state...")
        print("Student decoder hidden cell state: {}".format(self.student_dec_hidden[1]))
        print("Solution decoder hidden cell state: {}".format(self.soln_dec_hidden[1]))
        if np.allclose(self.student_dec_hidden[1].numpy(), self.soln_dec_hidden[1].numpy(), atol=1e-4):
            result = True
            print('Passed decoder cell state test')
        else:
            print('Failed decoder cell state test')
        self.assertTrue(result)

    @graded(is_hidden=True)
    def test_4(self):
        """2b-4-hidden:  Test shapes of outputs returned by forward"""
        BATCH_SIZE = 5
        CHAR_EMBEDDING_SIZE = 3
        HIDDEN_SIZE = 3

        class DummyVocab():
            def __init__(self):
                self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
                self.id2char = {id: char for char, id in self.char2id.items()}
                self.char_unk = self.char2id['<unk>']
                self.start_of_word = self.char2id["{"]
                self.end_of_word = self.char2id["}"]

        char_vocab = DummyVocab()

        # Initialize Models
        print_hyper_params_2b(
            vocab=char_vocab,
            batch_size=BATCH_SIZE,
            char_embedding_size=CHAR_EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
        )
        print('Initializing student\'s CharDecoder...')
        decoder = submission.CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=CHAR_EMBEDDING_SIZE,
            target_vocab=char_vocab
        )

        sequence_length = 4
        inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
        with torch.no_grad():
            try:
                print('Running student\'s forward function...')
                logits, (dec_hidden1, dec_hidden2) = decoder.forward(inpt)
            except Exception as e:
                traceback.print_exc()
                raise (e)
        logits_expected_size = [sequence_length, BATCH_SIZE, len(char_vocab.char2id)]
        dec_hidden_expected_size = [1, BATCH_SIZE, HIDDEN_SIZE]

        correct = False
        if (
                list(logits.size()) == logits_expected_size and
                list(dec_hidden1.size()) == dec_hidden_expected_size and
                list(dec_hidden2.size()) == dec_hidden_expected_size
        ):
            correct = True
            print('Passed!')
        else:
            print('Failed!')
        self.assertTrue(correct)

class Test_2c(GradedTestCase):
    def setUp(self):
        # Set Seeds
        random.seed(35436)
        np.random.seed(4355)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Create vocabulary
        class DummyVocab():
            def __init__(self):
                self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
                self.id2char = {id: char for char, id in self.char2id.items()}
                self.char_unk = self.char2id['<unk>']
                self.start_of_word = self.char2id["{"]
                self.end_of_word = self.char2id["}"]

        self.char_vocab = DummyVocab()
        self.char_ids = [4, 6, 7, 9, 11, 12, 16, 17, 19, 22, 23, 26]

    def random_char_sequence(self):
        sequence_length = np.random.randint(1, MAX_WORD_LENGTH_2C - 1)
        seq = np.zeros(MAX_WORD_LENGTH_2C, dtype=np.int64)
        seq[0] = 1
        for i in range(1, sequence_length + 1):
            seq[i] = np.random.choice(self.char_ids)
        seq[i] = 2
        return seq

    def random_batch_char_sequence(self, batch_size):
        return np.array([self.random_char_sequence() for b in range(batch_size)])

    @graded()
    def test_0(self):
        """2c-0-basic:  Sanity check for CharDecoder.train_forward()."""
        char_vocab = DummyVocab()

        # Initialize CharDecoder
        decoder = submission.CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=EMBED_SIZE,
            target_vocab=char_vocab)
        sequence_length = 4
        inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
        loss = decoder.train_forward(inpt)
        self.assertEqual(list(loss.size()), [], "Loss should be a scalar but its shape is: {}".format(list(loss.size())))

    @graded(is_hidden=True)
    def test_1(self):
        """2c-1-hidden:  Test shape of output of train_forward"""
        BATCH_SIZE = 5
        HIDDEN_SIZE = 3
        CHAR_EMBEDDING_SIZE = 3

        sequence_length = 4
        inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)

        print_hyper_params_2c(
            vocab=self.char_vocab,
            sequence_length=sequence_length,
            sequence=inpt,
            batch_size=BATCH_SIZE,
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=CHAR_EMBEDDING_SIZE,
            running_forward=True
        )

        print('Initializing student\'s CharDecoder...')
        decoder = submission.CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=CHAR_EMBEDDING_SIZE,
            target_vocab=self.char_vocab
        )

        with torch.no_grad():
            try:
                print('Running student\'s train_forward()...')
                loss = decoder.train_forward(inpt)
            except Exception as e:
                traceback.print_exc()
                raise (e)
        print('Expected loss with shape:', 0)
        if isinstance(loss, int):
            print('Student\'s loss has shape: 0')
            self.assertTrue(True)
        else:
            print('Student\'s loss has shape:', len(list(loss.size())))
            self.assertTrue(list(loss.size()) == [])

    @graded(is_hidden=True)
    def test_2(self):
        """2c-2-hidden:  Test output of train_forward"""
        BATCH_SIZE = 5

        random_input = torch.from_numpy(self.random_batch_char_sequence(BATCH_SIZE))
        random_input = random_input.t()  # make sure this is (length x batch) as in our spec

        print_hyper_params_2c(
            vocab=self.char_vocab,
            sequence_length=MAX_WORD_LENGTH_2C,
            sequence=random_input,
            batch_size=BATCH_SIZE_2C,
            hidden_size=HIDDEN_SIZE_2C,
            char_embedding_size=CHAR_EMBEDDING_SIZE_2C,
            running_forward=True
        )

        print('Initializing solution\'s CharDecoder...')
        soln_decoder = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=EMBED_SIZE_2C,
            target_vocab=self.char_vocab)

        print('Initializing student\'s CharDecoder...')
        student_decoder = submission.CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=EMBED_SIZE_2C,
            target_vocab=self.char_vocab)

        weight_copy(soln_decoder, student_decoder)

        print('Using model solution\'s forward() function in student\'s CharDecoder')
        student_decoder.forward = soln_decoder.forward

        soln_decoder.eval()
        student_decoder.eval()

        with torch.no_grad():
            try:
                print('Running solution\'s train_forward()...')
                soln_loss = soln_decoder.train_forward(random_input)
                print('Running student\'s train_forward()...')
                student_loss = student_decoder.train_forward(random_input)
            except Exception as e:
                traceback.print_exc()
                raise (e)

        print('Solution loss value:', soln_loss.item())
        print('Student loss value:', student_loss.item())
        self.assertAlmostEqual(student_loss.item(), soln_loss.item(), places=3)

class Test_2d(GradedTestCase):
    def set_random_seeds(self):
        random.seed(35436)
        np.random.seed(4355)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    def setUp(self):
        # init student decoder
        self.set_random_seeds()

        print('Setting batch size to %s' % BATCH_SIZE_2D)
        self.char_vocab = DummyVocab()

        print(
            'Initializing student\'s CharDecoder with hidden_size=%s, char_embedding_size=%s, target_vocab from sanity_check_en_es_data/char_vocab_sanity_check.json' % (
            HIDDEN_SIZE_2D, CHAR_EMBED_SIZE_2D))
        print('Initializing student\'s CharDecoder...')
        self.student_decoder = submission.CharDecoder(
            hidden_size=HIDDEN_SIZE_2D,
            char_embedding_size=CHAR_EMBED_SIZE_2D,
            target_vocab=self.char_vocab)

    def setUpSolnDecoderAndRun(self):
        # init solution decoder
        self.set_random_seeds()

        print('Initializing solution\'s CharDecoder...')
        self.soln_decoder = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).CharDecoder(
            hidden_size=HIDDEN_SIZE_2D,
            char_embedding_size=CHAR_EMBED_SIZE_2D,
            target_vocab=self.char_vocab)

        # copy solution decoder's parameters over to student decoder
        weight_copy(self.soln_decoder, self.student_decoder)

        # following 2 lines to ensure students are not double-penalized for incorrect forward or train_forward
        print('Using model solution\'s forward() and train_forward() functions in student\'s CharDecoder')
        self.student_decoder.forward = self.soln_decoder.forward
        self.student_decoder.train_forward = self.soln_decoder.train_forward

        # testing random input
        inpt1 = 100 * torch.rand(1, BATCH_SIZE_2D, HIDDEN_SIZE_2D, dtype=torch.float) - 50.
        inpt2 = 100 * torch.rand(1, BATCH_SIZE_2D, HIDDEN_SIZE_2D, dtype=torch.float) - 50.
        initialStates = (inpt1, inpt2)
        device_student = self.student_decoder.char_output_projection.weight.device
        device_soln = self.soln_decoder.char_output_projection.weight.device

        print('Passing decode_greedy these initialStates:')
        print(initialStates)
        print('Running solution\'s decode_greedy function...')
        self.decodedWords_soln = self.soln_decoder.decode_greedy(initialStates, device_soln)
        print('Running student\'s decode_greedy function...')
        try:
            self.decodedWords_student = self.student_decoder.decode_greedy(initialStates, device_student)
        except Exception as e:
            traceback.print_exc()
            raise (e)

        print('Solution decode_greedy output: %s' % self.decodedWords_soln)
        print('Student decode_greedy output: %s' % self.decodedWords_student)

    @graded()
    def test_0(self):
        """2d-0-basic:  Sanity check for CharDecoder.decode_greedy()"""
        char_vocab = DummyVocab()

        # Initialize CharDecoder
        decoder = submission.CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=EMBED_SIZE,
            target_vocab=char_vocab)
        sequence_length = 4
        inpt = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float)
        initialStates = (inpt, inpt)
        device = decoder.char_output_projection.weight.device
        decodedWords = decoder.decode_greedy(initialStates, device)
        self.assertEqual(len(decodedWords), BATCH_SIZE, "Length of decodedWords should be {} but is: {}".format(BATCH_SIZE,
                                                                                                          len(decodedWords)))

    @graded(is_hidden=True)
    def test_1(self):
        """2d-1-hidden:  Test shape of output of decode_greedy"""
        inpt = torch.zeros(1, BATCH_SIZE_2D, HIDDEN_SIZE_2D, dtype=torch.float)
        initialStates = (inpt, inpt)
        device = self.student_decoder.char_output_projection.weight.device
        print('Passing decode_greedy this initialStates:')
        print(initialStates)
        print('Running student\'s decode_greedy function...')
        try:
            decodedWords = self.student_decoder.decode_greedy(initialStates, device)
        except Exception as e:
            traceback.print_exc()
            raise (e)
        print('Student\'s output from decode_greedy: ', decodedWords)
        print('Expect decode_greedy output to be a list length %s' % BATCH_SIZE_2D)
        print('Student\'s decode_greedy output is a list length %s' % len(decodedWords))
        self.assertTrue(len(decodedWords) == BATCH_SIZE_2D, msg='Failure: list is wrong length.')

    @graded(is_hidden=True)
    def test_2(self):
        """2d-2-hidden:  Test output of decode_greedy for partial match"""
        """
        1. Ensures students are not double penalized for incorrect init, forward, and train_forward
        2. creates random initial state to feed into decode_greedy function
        3. feeds same initial state to both soln and student decode_greedy and checks that outputs are the same
        """
        self.setUpSolnDecoderAndRun()  # sets up solution decoder, copies weights, and runs both forward

        print('Just looking at first 18 characters...')
        decodedWords_soln_partial = [i[:18] for i in self.decodedWords_soln]
        decodedWords_student_partial = [i[:18] for i in self.decodedWords_student]
        print('First 18 chars of solution decode_greedy output: %s' % decodedWords_soln_partial)
        print('First 18 chars of student\'s decode_greedy output: %s' % decodedWords_student_partial)
        self.assertTrue(decodedWords_student_partial == decodedWords_soln_partial,
                        msg='Failure: incorrect decode_greedy partial output (first 18 characters of each string in decodedWords do not match solution)')
        print(
            'Passed - receiving credit for correct partial match (first 18 characters of each string in decodedWords match solution)')

    @graded(is_hidden=True)
    def test_3(self):
        """2d-3-hidden:  Test output of decode_greedy for exact match (hidden)"""
        """
        1. Ensures students are not double penalized for incorrect init, forward, and train_forward
        2. creates random initial state to feed into decode_greedy function
        3. feeds same initial state to both soln and student decode_greedy and checks that outputs are the same
        """
        self.setUpSolnDecoderAndRun()  # sets up solution decoder and copies parts across to student decoder (to avoid double penalization)

        self.assertTrue(
            self.decodedWords_student == self.decodedWords_soln or self.decodedWords_student == [i[:-1] for i in
                                                                                                 self.decodedWords_soln],
            msg='Failure: decode_greedy output not exactly the same as solution\'s (some strings do not match)')
        print(
            'Passed - receiving credit for correct exact match (all characters of each string in decodedWords match solution)')

class Test_2e(GradedTestCase):
    @graded(is_hidden=True)
    def test_0(self):
        """2e-0-hidden:  BLEU score on tiny test set is over 99"""
        # Read files
        hyp_file = os.path.join('./outputs', 'test_outputs_local_q2_soln.txt')
        ref_file = os.path.join('./en_es_data', 'test_tiny.en')

        if os.path.exists(os.path.join('./submission', 'test_outputs_local_q2_soln.txt')):
            hyp_file = os.path.join('./submission', 'test_outputs_local_q2_soln.txt')

        self.assertTrue(os.path.exists(hyp_file),
                        f'Output test file (outputs/test_outputs_local_q2_soln.txt) does not exist.')

        ref = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus(ref_file, source='tgt')
        hyp = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus(hyp_file, source='tgt')

        ref = [line[1:-1] for line in ref]  # remove <s>...</s>
        hyp = [line[1:-1] for line in hyp]  # remove <s>...</s>

        hyp = [submission.Hypothesis(value=line, score=1) for line in hyp]
        self.ref, self.hyp = ref, hyp

        score = bleu(self.ref, self.hyp) * 100
        print("BLEU " + str(score))
        self.assertGreaterEqual(score, 99.0, "Your BLEU score ({}) is below 99".format(score))

class Test_2f(GradedTestCase):
    @graded(is_hidden=True)
    def test_0(self):
        """2f-0-hidden:  BLEU score above 10"""
        # Read files
        hyp_file = os.path.join('./outputs', 'test_outputs_soln.txt')
        ref_file = os.path.join('./en_es_data', 'test.en')

        if os.path.exists(os.path.join('./submission', 'test_outputs_soln.txt')):
            hyp_file = os.path.join('./submission', 'test_outputs_soln.txt')

        self.assertTrue(os.path.exists(hyp_file),
                        f'Output test file (outputs/test_outputs_soln.txt) does not exist. To generate this file, follow these steps:\n'
                        '1. Generate vocab.py (run.sh vocab)\n'
                        '2. Generate and train a model (run.sh train)\n'
                        '3. Generate model outputs on the autograder test set (python envaluation_output.py)')

        ref = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus(ref_file, source='tgt')
        hyp = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus(hyp_file, source='tgt')

        ref = [line[1:-1] for line in ref]  # remove <s>...</s>
        hyp = [line[1:-1] for line in hyp]  # remove <s>...</s>

        hyp = [submission.Hypothesis(value=line, score=1) for line in hyp]
        self.ref, self.hyp = ref, hyp
        self.bleu = bleu(self.ref, self.hyp) * 100
        print("Your BLEU score: " + str(self.bleu))
        self.assertGreaterEqual(self.bleu, 10, "Your BLEU score ({}) is below 10".format(self.bleu))

    @graded(is_hidden=True)
    def test_1(self):
        """2f-1-hidden:  BLEU score above 16"""
        # Read files
        hyp_file = os.path.join('./outputs', 'test_outputs_soln.txt')
        ref_file = os.path.join('./en_es_data', 'test.en')

        if os.path.exists(os.path.join('./submission', 'test_outputs_soln.txt')):
            hyp_file = os.path.join('./submission', 'test_outputs_soln.txt')

        self.assertTrue(os.path.exists(hyp_file),
                        f'Output test file (outputs/test_outputs_soln.txt) does not exist. To generate this file, follow these steps:\n'
                        '1. Generate vocab.py (run.sh vocab)\n'
                        '2. Generate and train a model (run.sh train)\n'
                        '3. Test the model (sh run.sh test)')

        ref = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus(ref_file, source='tgt')
        hyp = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol).read_corpus(hyp_file, source='tgt')

        ref = [line[1:-1] for line in ref]  # remove <s>...</s>
        hyp = [line[1:-1] for line in hyp]  # remove <s>...</s>

        hyp = [submission.Hypothesis(value=line, score=1) for line in hyp]
        self.ref, self.hyp = ref, hyp
        self.bleu = bleu(self.ref, self.hyp) * 100
        print("Your BLEU score: " + str(self.bleu))
        self.assertGreaterEqual(self.bleu, 16, "Your BLEU score ({}) is below 16".format(self.bleu))

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)
