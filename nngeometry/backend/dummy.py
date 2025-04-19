class DummyGenerator:
    """
    This dummy generator is used for pickled objects
    """

    def __init__(self, device):
        self.device = device

    def get_device(self, layer_collection):
        return self.device
