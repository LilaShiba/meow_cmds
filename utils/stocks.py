from dataParser import DataParser


class NeuralNet(DataParser):
    def __init__(self, filepath: str) -> None:
        # Use super() correctly
        super().__init__(filepath)
