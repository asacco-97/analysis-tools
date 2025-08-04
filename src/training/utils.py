from training.gbm_model_trainer import XGBModelTrainer
from typing import List
import logging 

class ListLogHandler(logging.Handler):
    def __init__(self, log_list: List[str]):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        msg = self.format(record)
        self.log_list.append(msg)

class XGBLogger:
    def __init__(self, trainer: XGBModelTrainer):
        self.trainer = trainer
        self.name = "xgb-trainer"

    def __enter__(self):
        self.trainer.logger = logging.getLogger(self.name)
        self.trainer.logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        log_path, _ = self.trainer._resolve_output_path(self.trainer.config.log_file)

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)
        self.trainer.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.trainer.logger.addHandler(stream_handler)

        memory_handler = ListLogHandler(self.trainer.log_lines)
        memory_handler.setFormatter(formatter)
        self.trainer.logger.addHandler(memory_handler)

        return self.trainer.logger

    def __exit__(self, exc_type, exc_value, traceback):
        logger = self.trainer.logger
        for handler in list(logger.handlers): # type: ignore
            handler.flush()
            handler.close()
            logger.removeHandler(handler) # type: ignore
        self.trainer.logger = None
