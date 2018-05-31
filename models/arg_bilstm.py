"""BiLSTM CNN with embeddings model derived from ukplab BiLSTM model.

It has some more flexible functions, but the core of the model is the same."""

from ukplab_nets.neuralnets.BiLSTM import BiLSTM


class ArgBiLSTM(BiLSTM):
    """BiLSTM model tailored for argumentation mining tasks"""

    def computeScore(self, modelName, devMatrix, testMatrix):
        return self.computeF1Scores(modelName, devMatrix, testMatrix)