class BaseTrain(pl.LightningModule, ABC):
    def __init__(self, CustomMetric, cp_path):
        super().__init__()

        self.train_metric = CustomMetric()
        self.val_metric = CustomMetric()
        self.test_metric = CustomMetric()

        self.cp_path = cp_path

    @abstractmethod
    def setup_grad(self):
        pass


class Hello(BaseTrain):
    def setup_grad(self):
        pass
