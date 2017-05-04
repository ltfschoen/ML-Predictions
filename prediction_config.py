class PredictionConfig:
    """ Machine Learning configuration """

    def __init__(self):
        # K-Nearest-Neighbors Machine Learning Model selection:
        #   - external - External (uses Skikit-Learn library) OR
        #   - manual - Manually configured
        self.ML_MODEL_KNN = "external"