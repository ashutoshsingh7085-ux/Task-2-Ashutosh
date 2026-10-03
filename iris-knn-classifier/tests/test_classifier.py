import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classifier  # noqa: E402


class TestData(unittest.TestCase):
    def test_iris_shape(self):
        X, y, features, classes = classifier.load_data()
        self.assertEqual(X.shape, (150, 4))
        self.assertEqual(len(classes), 3)
        self.assertEqual(len(features), 4)

    def test_split_sizes_and_balance(self):
        X, y, _, _ = classifier.load_data()
        X_tr, X_te, y_tr, y_te = classifier.split_data(X, y)
        self.assertEqual((len(X_tr), len(X_te)), (120, 30))
        self.assertTrue(np.array_equal(np.bincount(y_te), [10, 10, 10]))

    def test_split_is_reproducible(self):
        X, y, _, _ = classifier.load_data()
        a = classifier.split_data(X, y)[1]
        b = classifier.split_data(X, y)[1]
        self.assertTrue(np.array_equal(a, b))


class TestModel(unittest.TestCase):
    def setUp(self):
        X, y, _, self.classes = classifier.load_data()
        self.X_tr, self.X_te, self.y_tr, self.y_te = classifier.split_data(X, y)

    def test_scaler_fitted_on_train_only(self):
        model = classifier.build_model(5).fit(self.X_tr, self.y_tr)
        scaler = model.named_steps["standardscaler"]
        self.assertTrue(np.allclose(scaler.mean_, self.X_tr.mean(axis=0)))

    def test_accuracy_reasonable(self):
        model = classifier.build_model(5).fit(self.X_tr, self.y_tr)
        res = classifier.evaluate(model, self.X_te, self.y_te, self.classes)
        self.assertGreaterEqual(res["accuracy"], 0.90)
        self.assertEqual(res["confusion_matrix"].shape, (3, 3))
        self.assertEqual(res["confusion_matrix"].sum(), 30)

    def test_best_k_tie_prefers_larger(self):
        self.assertEqual(classifier.best_k({1: 0.1, 3: 0.05, 5: 0.05, 7: 0.2}), 5)

    def test_sweep_returns_error_per_k(self):
        errs = classifier.sweep_k(self.X_tr, self.y_tr, k_values=[1, 5])
        self.assertEqual(set(errs), {1, 5})
        self.assertTrue(all(0 <= e <= 1 for e in errs.values()))


class TestEndToEnd(unittest.TestCase):
    def test_run_saves_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = classifier.run(out_dir=tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "k_selection.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "confusion_matrix.png")))
            self.assertGreaterEqual(res["accuracy"], 0.90)


if __name__ == "__main__":
    unittest.main()
