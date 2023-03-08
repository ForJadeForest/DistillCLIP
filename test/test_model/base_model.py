class BaseModel:
    def __init__(self, model_name, is_clip):
        self._model_name = model_name
        self._is_clip = is_clip

    def __call__(self, *args, **kwargs):
        return self.cal_score(*args, **kwargs)

    def cal_score(self, *args, **kwargs):
        raise NotImplemented('cal_score is not Implemented')

    @property
    def model_name(self):
        return self._model_name

    @property
    def is_clip(self):
        return self._is_clip
