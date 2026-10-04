from src.Operators.OperatorBase import Operator
from abc import ABC
from src.DBHandler import DBHandler
from src.optimizer_paths import model_path

class Seeker(Operator, ABC):
    HAS_ML_COST_MODEL: bool = True

    def __init__(self, k: int) -> None:
        super().__init__(k)

        self._cached_predicted_runtime = None
        if self.DB.USE_ML_OPTIMIZER and self.HAS_ML_COST_MODEL:
            from xgboost import XGBRegressor
            mpath = model_path(self.DB.dataset_dir(), self.DB.optimizer_profile_str(),
                               self.__class__.__name__)
            if not mpath.is_file():
                raise FileNotFoundError(
                    f"USE_ML_OPTIMIZER on but {mpath} is missing. Train it with "
                    f"`python -m src.Optimizer.cli train --lake <name>`, "
                    f"or unset BLEND_USE_ML_OPTIMIZER.")
            self.model = XGBRegressor()
            self.model.load_model(mpath)
        else:
            self.model = None
            self._cached_predicted_runtime = 1

    def _predict_runtime(self, columns: list, db: DBHandler) -> float:
        if self._cached_predicted_runtime is not None:
            return self._cached_predicted_runtime
        from src.Optimizer.features import compute_features
        features = compute_features(columns, db)
        self._cached_predicted_runtime = self.model.predict([features])[0]
        return self._cached_predicted_runtime
