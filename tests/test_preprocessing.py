import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):

    # Test 1: select_features should fail if the user requests columns
    # that do not exist in the DataFrame. This prevents silent errors
    # where the clustering pipeline runs on the wrong data.
    def test_missing_feature_columns(self):
        df = pd.DataFrame({"x": [1,2,3], "y": [4,5,6]})
        with self.assertRaises(KeyError):
            select_features(df, ["x", "z"])  # "z" does not exist

    # Test 2: select_features must reject non-numeric data. Without this,
    # standardisation or clustering could crash later in the pipeline.
    def test_non_numeric_columns(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "label": ["a", "b", "c"],  # non-numeric
        })
        with self.assertRaises(TypeError):
            select_features(df, ["x", "label"])

    # Test 3: standardise_features must actually standardise the data so
    # that each feature has mean ~0 and std ~1. If not, clustering could
    # be dominated by features with large scales.
    def test_standardise_features_correctness(self):
        X = np.array([[1.0, 10.0],
                      [2.0, 20.0],
                      [3.0, 30.0]])
        X_scaled = standardise_features(X)

        # Means should be approximately zero
        means = X_scaled.mean(axis=0)
        self.assertTrue(np.allclose(means, 0.0, atol=1e-7))

        # Standard deviations should be approximately one
        stds = X_scaled.std(axis=0)
        self.assertTrue(np.allclose(stds, 1.0, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
