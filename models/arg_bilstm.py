"""BiLSTM CNN with embeddings model derived from ukplab BiLSTM model.

It has some more flexible functions, but the core of the model is the same."""

from sklearn import metrics
from ukplab_nets.neuralnets.BiLSTM import BiLSTM


class ArgBiLSTM(BiLSTM):
    """BiLSTM model tailored for argumentation mining tasks"""

    def computeScore(self, modelName, devMatrix, testMatrix):
        return self.computeF1Scores(modelName, devMatrix, testMatrix)

    def computeF1(self, modelName, sentences):
        """Returns a traditional f1 score.

        It does not check the consistency of BIO labels."""
        labelKey = self.labelKeys[modelName]
        model = self.models[modelName]
        idx2Label = self.idx2Labels[modelName]

        correctLabels = [sentences[idx][labelKey]
                         for idx in range(len(sentences))]
        predLabels = self.predictLabels(model, sentences)

        pre, rec, f1, _ = metrics.precision_recall_fscore_support(
            correctLabels, predLabels, average='weighted', warn_for=[])

        return pre, rec, f1
