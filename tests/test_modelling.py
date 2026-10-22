"""End-to-end check of the training and inference interface."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling.predicting import predict
from src.modelling.preprocessing import FEATURE_COLUMNS, read_training_data
from src.modelling.training import train_and_save
from src.modelling.utils import load_model


class ModellingTest(unittest.TestCase):
    def test_train_save_reload_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            count = 90
            rng = np.random.default_rng(42)
            data = pd.DataFrame(
                {
                    "Sex": np.resize(["M", "F", "I"], count),
                    "Length": rng.uniform(0.2, 0.8, count),
                    "Diameter": rng.uniform(0.1, 0.6, count),
                    "Height": rng.uniform(0.05, 0.25, count),
                    "Whole weight": rng.uniform(0.1, 2.0, count),
                    "Shucked weight": rng.uniform(0.05, 0.9, count),
                    "Viscera weight": rng.uniform(0.02, 0.5, count),
                    "Shell weight": rng.uniform(0.03, 0.6, count),
                    "Rings": rng.integers(5, 20, count),
                }
            )
            data.loc[0, "Height"] = 0.0
            csv_path = root / "abalone.csv"
            model_path = root / "abalone_model.pkl"
            data.to_csv(csv_path, index=False)

            result = train_and_save(
                csv_path,
                model_path,
                tracking_dir=root / "mlruns",
                results_path=root / "results.json",
            )
            model = load_model(model_path)
            predictions = predict(model, read_training_data(csv_path).loc[[0, 1], FEATURE_COLUMNS])

            self.assertEqual(len(predictions), 2)
            self.assertTrue(np.isfinite(predictions).all())
            self.assertEqual(sum(result["rows"].values()), count)
            self.assertIn(result["selected_model"], result["validation"])
            self.assertTrue((root / "results.json").is_file())
            with self.assertRaisesRegex(ValueError, "Missing input columns"):
                predict(model, data[["Sex"]].head(1))


if __name__ == "__main__":
    unittest.main()
