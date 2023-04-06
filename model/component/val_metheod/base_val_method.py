class BascValMetric:
    def __init__(self):
        self.model_name = None
        self.res_step_dict = {}
        self.res_end_dict = {}
        self.validation_step_outputs = []

    def set_model_name(self, new_name):
        self.model_name = new_name

    def validation_step(self, *args, **kwargs):
        raise NotImplemented

    def validation_end(self, *args, **kwargs):
        raise NotImplemented

    def reset(self):
        raise NotImplemented
