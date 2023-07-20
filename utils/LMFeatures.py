# pulled from https://github.com/HuthLab/semantic-decoding/blob/main/decoding/StimulusModel.py

import numpy as np

class LMFeatures():
    """class for extracting contextualized features of stimulus words
    """
    def __init__(self, model, layer, context_words):
        self.model, self.layer, self.context_words = model, layer, context_words

    def extend(self, extensions, verbose = False):
        """outputs array of vectors corresponding to the last words of each extension
        """
        contexts = [extension[-(self.context_words+1):] for extension in extensions]
        if verbose: print(contexts)
        context_array = self.model.get_context_array(contexts)
        embs = self.model.get_hidden(context_array, layer = self.layer)
        return embs[:, len(contexts[0]) - 1]

    def make_stim(self, words):
        """outputs matrix of features corresponding to the stimulus words
        """
        context_array = self.model.get_story_array(words, self.context_words)
        embs = self.model.get_hidden(context_array, layer = self.layer)
        return np.vstack([embs[0, :self.context_words], 
            embs[:context_array.shape[0] - self.context_words, self.context_words]])