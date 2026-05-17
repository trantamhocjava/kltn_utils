class MetricCalculator:
    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.c_pred = []
        self.c_true = []

        self.loss_dict = self.get_loss_dict()

    def get_loss_dict(self):
        pass


class CustomMetric(MetricCalculator):
    def get_loss_dict(self):
        return {
            "a": [],
            "b": [],
        }


a = CustomMetric()
a.reset()
