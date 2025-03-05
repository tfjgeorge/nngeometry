class DummyGenerator:
    """
    This dummy generator is used for pickled objects
    """

    def __init__(self, layer_collection, device):
        self.layer_collection = layer_collection
        self.device = device

    def get_device(self):
        return self.device
