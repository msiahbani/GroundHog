def prototype_state():
    state = {}

    # Random seed
    state['seed'] = 1234
    # Logging level
    state['level'] = 'DEBUG'

    # ----- DATA -----
    # (all Nones in this section are placeholders for required values)

    # Source sequences (must be singleton list for backward compatibility)
    state['source'] = [None]
    # Target sequences (must be singleton list for backard compatiblity)
    state['target'] = [None]
    # index -> word dict for the source language
    state['indx_word'] = None
    # index -> word dict for the target language
    state['indx_word_target'] = None
    # word -> index dict for the source language
    state['word_indx'] = None
    # word -> index dict for the target language
    state['word_indx_trgt'] = None

    # ----- VOCABULARIES -----
    # (all Nones in this section are placeholders for required values)

    # A string representation for the unknown word placeholder for both language
    state['oov'] = 'UNK'
    # These are unknown word placeholders
    state['unk_sym_source'] = 1
    state['unk_sym_target'] = 1
    # These are end-of-sequence marks
    state['null_sym_source'] = None
    state['null_sym_target'] = None
    # These are vocabulary sizes for the source and target languages
    state['n_sym_source'] = None
    state['n_sym_target'] = None

    # ----- MODEL STRUCTURE -----

    # The components of the annotations produced by the Encoder
    state['last_forward'] = True
    state['last_backward'] = False
    state['forward'] = False
    state['backward'] = False
    # Turns on "search" mechanism
    state['search'] = False
    # Turns on using the shortcut from the previous word to the current one
    state['bigram'] = True
    # Turns on initialization of the first hidden state from the annotations
    state['bias_code'] = True
    # Turns on using the context to compute the next Decoder state
    state['decoding_inputs'] = True
    # Turns on an intermediate maxout layer in the output
    state['deep_out'] = True
    # Heights of hidden layers' stacks in encoder and decoder
    # WARNING: has not been used for quite while and most probably
    # doesn't work...
    state['encoder_stack'] = 1
    state['decoder_stack'] = 1
    # Use the top-most recurrent layer states as annotations
    # WARNING: makes sense only for hierachical RNN which
    # are in fact currently not supported
    state['take_top'] = True
    # Activates age old bug fix - should always be true
    state['check_first_word'] = True

    state['eps'] = 1e-10

    # ----- MODEL COMPONENTS -----

    # Low-rank approximation activation function
    state['rank_n_activ'] = 'lambda x: x'
    # Hidden-to-hidden activation function
    state['activ'] = 'lambda x: TT.tanh(x)'
    # Nonlinearity for the output
    state['unary_activ'] = 'Maxout(2)'

    # Hidden layer configuration for the forward encoder
    state['enc_rec_layer'] = 'RecurrentLayer'
    state['enc_rec_gating'] = True
    state['enc_rec_reseting'] = True
    state['enc_rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['enc_rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'
    # Hidden layer configuration for the decoder
    state['dec_rec_layer'] = 'RecurrentLayer'
    state['dec_rec_gating'] = True
    state['dec_rec_reseting'] = True
    state['dec_rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['dec_rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'
    # Default hidden layer configuration, which is effectively used for
    # the backward RNN
    # TODO: separate back_enc_ configuration and convert the old states
    # to have it
    state['rec_layer'] = 'RecurrentLayer'
    state['rec_gating'] = True
    state['rec_reseting'] = True
    state['rec_gater'] = 'lambda x: TT.nnet.sigmoid(x)'
    state['rec_reseter'] = 'lambda x: TT.nnet.sigmoid(x)'

    # ----- SIZES ----

    # Dimensionality of hidden layers
    state['dim'] = 1000
    # Dimensionality of low-rank approximation
    state['rank_n_approx'] = 100
    # k for the maxout stage of output generation
    state['maxout_part'] = 2.

    # ----- WEIGHTS, INITIALIZATION -----

    # This one is bias applied in the recurrent layer. It is likely
    # to be zero as MultiLayer already has bias.
    state['bias'] = 0.

    # Weights initializer for the recurrent net matrices
    state['rec_weight_init_fn'] = 'sample_weights_orth'
    state['rec_weight_scale'] = 1.
    # Weights initializer for other matrices
    state['weight_init_fn'] = 'sample_weights_classic'
    state['weight_scale'] = 0.01

    # ---- REGULARIZATION -----

    # WARNING: dropout is not tested and probably does not work.
    # Dropout in output layer
    state['dropout'] = 1.
    # Dropout in recurrent layers
    state['dropout_rec'] = 1.

    # WARNING: weight noise regularization is not tested
    # and most probably does not work.
    # Random weight noise regularization settings
    state['weight_noise'] = False
    state['weight_noise_rec'] = False
    state['weight_noise_amount'] = 0.01

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    # A magic gradient clipping option that you should never change...
    state['cutoff_rescale_length'] = 0.

    # ----- TRAINING METHOD -----

    # Turns on noise contrastive estimation instead maximum likelihood
    state['use_nce'] = False

    # Choose optimization algorithm
    state['algo'] = 'SGD_adadelta'

    # Adadelta hyperparameters
    state['adarho'] = 0.95
    state['adaeps'] = 1e-6

    # Early stopping configuration
    # WARNING: was never changed during machine translation experiments,
    # as early stopping was not used.
    state['patience'] = 1
    state['lr'] = 1.
    state['minlr'] = 0

    # Batch size
    state['bs']  = 64
    # We take this many minibatches, merge them,
    # sort the sentences according to their length and create
    # this many new batches with less padding.
    state['sort_k_batches'] = 10

    # Maximum sequence length
    state['seqlen'] = 30
    # Turns on trimming the trailing paddings from batches
    # consisting of short sentences.
    state['trim_batches'] = True
    # Loop through the data
    state['use_infinite_loop'] = True
    # Start from a random entry
    state['shuffle'] = False

    # ----- TRAINING PROCESS -----

    # Prefix for the model, state and timing files
    state['prefix'] = 'phrase_'
    # Specifies whether old model should be reloaded first
    state['reload'] = True
    # When set to 0 each new model dump will be saved in a new file
    state['overwrite'] = 1

    # Number of batches to process
    state['loopIters'] = 3000000
    # Maximum number of minutes to run
    state['timeStop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1

    # Reset data iteration every this many epochs
    state['reset'] = -1
    # Frequency of training error reports (in number of batches)
    state['trainFreq'] = 1
    # Frequency of running hooks
    state['hookFreq'] = 13
    # Validation frequency
    state['validFreq'] = 500
    # Model saving frequency (in minutes)
    state['saveFreq'] = 10

    # Sampling hook settings
    state['n_samples'] = 3
    state['n_examples'] = 3

    # Raise exception if nan
    state['on_nan'] = 'raise'

    # additional hidden layer for embbeding layer
    state['embd_layer'] = 0
    # Initialize embbeding layer from already trained model (just containing weights for embbeding)
    state['load_embd'] = False
    # exclude these parameters form update process during decoding
    state['frozen_params' = []
    

    return state

def prototype_phrase_state():
    """This prototype is the configuration used in the paper
    'Learning Phrase Representations using RNN Encoder-Decoder
    for  Statistical Machine Translation' """

    state = prototype_state()

    state['source'] = ["/SSD/siahbanim/NNPP/europarl/pp-dp-phrL2-pphrL3-w4-test-all/binarized_pp1.fr.shuf.h5"]
    state['target'] = ["/SSD/siahbanim/NNPP/europarl/pp-dp-phrL2-pphrL3-w4-test-all/binarized_pp2.fr.shuf.h5"]
    state['indx_word'] = "/SSD/siahbanim/NNPP/europarl/pp-dp-phrL2-pphrL3-w4-test-all/ivocab.pp1.pkl"
    state['indx_word_target'] = "/SSD/siahbanim/NNPP/europarl/pp-dp-phrL2-pphrL3-w4-test-all/ivocab.pp2.pkl"
    state['word_indx'] = "/SSD/siahbanim/NNPP/europarl/pp-dp-phrL2-pphrL3-w4-test-all/vocab.pp1.pkl"
    state['word_indx_trgt'] = "/SSD/siahbanim/NNPP/europarl/pp-dp-phrL2-pphrL3-w4-test-all/vocab.pp2.pkl"

    state['null_sym_source'] = 6177
    state['null_sym_target'] = 9826
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1
    state['timeStop'] = 24*60*5
    state['seqlen'] = 3

    return state

def prototype_encdec_state():
    """This prototype is the configuration used to train the RNNenc-30 model from the paper
    'Neural Machine Translation by Jointly Learning to Align and Translate' """

    state = prototype_state()

    state['source'] = ["/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/bitexts.selected/selected50/binarized_text.en.shuf.h5"]
    state['target'] = ["/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/bitexts.selected/selected50/binarized_text.fr.shuf.h5"]
    state['indx_word'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/bitexts.selected/selected50/ivocab.en.pkl"
    state['indx_word_target'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/bitexts.selected/selected50/ivocab.fr.pkl"
    state['word_indx'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/bitexts.selected/selected50/vocab.en.pkl"
    state['word_indx_trgt'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/bitexts.selected/selected50/vocab.fr.pkl"

    state['null_sym_source'] = 30000
    state['null_sym_target'] = 30000
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['n_sym_target'] = state['null_sym_target'] + 1

    state['seqlen'] = 50
    state['bs']  = 80

    state['dim'] = 1000
    state['rank_n_approx'] = 300

    state['prefix'] = 'encdec50_'

    return state

def prototype_search_state():
    """This prototype is the configuration used to train the RNNsearch-50 model from the paper
    'Neural Machine Translation by Jointly Learning to Align and Translate' """

    state = prototype_encdec_state()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['sort_k_batches'] = 20
    state['prefix'] = 'search50_reverse_'


    state['source'] = ["/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/reversedEN/vocab89k/binarized_text.en.shuf.h5"]
    state['target'] = ["/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/reversedEN/vocab89k/binarized_text.fr.shuf.h5"]
    state['indx_word'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/reversedEN/vocab89k/ivocab.en.pkl"
    state['indx_word_target'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/ivocab.fr.pkl"
    state['word_indx'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/reversedEN/vocab89k/vocab.en.pkl"
    state['word_indx_trgt'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/vocab.fr.pkl"
    state['null_sym_source'] = 78584
    state['n_sym_source'] = state['null_sym_source'] + 1
    state['load_embd'] = False
    #state['reload'] = False

    return state

def prototype_search_state_embd():
    """This prototype is the configuration used to train the RNNsearch-50 model from the paper
    'Neural Machine Translation by Jointly Learning to Align and Translate' """

    state = prototype_encdec_state()

    state['dec_rec_layer'] = 'RecurrentLayerWithSearch'
    state['search'] = True
    state['last_forward'] = False
    state['forward'] = True
    state['backward'] = True
    state['sort_k_batches'] = 20
    state['prefix'] = 'search50_reverse_embd_'


    state['source'] = ["/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/reversedEN/vocab89k/binarized_text.en.shuf.h5"]
    state['target'] = ["/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/reversedEN/vocab89k/binarized_text.fr.shuf.h5"]
    state['indx_word'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/reversedEN/vocab89k/ivocab.en.pkl"
    state['indx_word_target'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/ivocab.fr.pkl"
    state['word_indx'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/reversedEN/vocab89k/vocab.en.pkl"
    state['word_indx_trgt'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/vocab.fr.pkl"
    state['null_sym_source'] = 78584
    state['n_sym_source'] = state['null_sym_source'] + 1


    state['load_embd'] = True
    state['reload'] = False
    state['embd_layer'] = 1
    state['frozen_params'] = ['W_0_enc_approx_embdr', 'b_0_enc_approx_embdr']
    state['timeStop'] = 25*60


    return state

def prototype_phrase_lstm_state():
    #state = prototype_phrase_state()
    state = prototype_encdec_state()

    state['source'] = ["/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/binarized_text.en.shuf.h5"]
    state['target'] = ["/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/binarized_text.fr.shuf.h5"]
    state['indx_word'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/ivocab.en.pkl"
    state['indx_word_target'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/ivocab.fr.pkl"
    state['word_indx'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/vocab.en.pkl"
    state['word_indx_trgt'] = "/SSD/siahbanim/nnmt/data/pre-processed/nnmt-shared-task/fr-en/selected-1M/selected50/vocab.fr.pkl"

    state['enc_rec_layer'] = 'LSTMLayer'
    state['enc_rec_gating'] = False
    state['enc_rec_reseting'] = False
    state['dec_rec_layer'] = 'LSTMLayer'
    state['dec_rec_gating'] = False
    state['dec_rec_reseting'] = False
    state['dim_mult'] = 4
    state['sort_k_batches'] = 20
    state['prefix'] = 'encdec50_lstm_'

    return state
