from .abstract_data import AbstractData


class ProcessingData(AbstractData):
    class Extra(object):
        signal_window_size = None
        orig_feats = None
        dim_reduction = None
        pass

    extra = Extra()

    def __init__(self):
        pass
