import pandas as pd
from typing import Tuple, Union, List, Optional
from sklearn.linear_model import LogisticRegression
class DelayModel:

    feature_names: List[str] = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    def __init__(
        self
    ):
        self._model: Optional[LogisticRegression]  = None # Model should be saved in this attribute.
        self._raw_data: Optional[pd.DataFrame] = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # One-hot encode the three categorical variables
        opera_dummies = pd.get_dummies(data["OPERA"], prefix="OPERA")
        tipo_dummies = pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO")
        mes_dummies = pd.get_dummies(data["MES"], prefix="MES")

        features = pd.concat(
            [opera_dummies, tipo_dummies, mes_dummies],
            axis=1,
        )

        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0

        # Keep only the 10 important features, in the correct order
        features = features[self.feature_names]

        # Serving Mode
        if target_column is None:
            self._raw_data = pd.read_csv(filepath_or_buffer="./data/data.csv")
            return features

        # Training mode
        if target_column is not None:
            # create target feature
            # min_diff = (Fecha-O - Fecha-I) in minutes, delayed if > 15
            fecha_o = pd.to_datetime(data["Fecha-O"], format="%Y-%m-%d %H:%M:%S")
            fecha_i = pd.to_datetime(data["Fecha-I"], format="%Y-%m-%d %H:%M:%S")
            min_diff = (fecha_o - fecha_i).dt.total_seconds() / 60.0
            data["delay"] = (min_diff > 15).astype(int)
            target = data[[target_column]]
            return features, target


    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Compute class weights as in the notebook
        n_samples = len(target)
        n_y0 = (target == 0).sum()
        n_y1 = (target == 1).sum()

        class_weight = {
            1: n_y0 / n_samples,
            0: n_y1 / n_samples,
        }

        self._model = LogisticRegression(class_weight=class_weight)
        self._model.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            # Nice to have, but not required for the moment.
            #if self._raw_data is None:
            #    raise ValueError("No training data is stored.")

            # Train using stored raw data and 'delay' as target
            print("Training Model")
            print(self._raw_data.columns)
            train_features, train_target = self.preprocess(
                data=self._raw_data,
                target_column="delay",
            )
            self.fit(train_features, train_target)

        preds = self._model.predict(features)
        return preds.astype(int).tolist()