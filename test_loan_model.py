#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive test suite for the loan prediction model training and evaluation pipeline.
Tests cover model training, predictions, metrics, edge cases, and reproducibility.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class TestModelTraining(unittest.TestCase):
    """Test cases for model training with different data shapes."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple synthetic dataset
        np.random.seed(42)
        self.X_simple = np.random.randn(100, 5)
        self.y_simple = np.random.randint(0, 2, 100)
        
    def test_logistic_regression_training(self):
        """Test that Logistic Regression trains correctly."""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_simple, self.y_simple)
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))
        
    def test_decision_tree_training(self):
        """Test that Decision Tree trains correctly."""
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_simple, self.y_simple)
        self.assertTrue(hasattr(model, 'tree_'))
        
    def test_random_forest_training(self):
        """Test that Random Forest trains correctly."""
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(self.X_simple, self.y_simple)
        self.assertTrue(hasattr(model, 'estimators_'))
        self.assertEqual(len(model.estimators_), 10)
    
    def test_training_with_different_shapes(self):
        """Test model training with various data shapes."""
        test_shapes = [(50, 3), (100, 10), (200, 20)]
        
        for n_samples, n_features in test_shapes:
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 2, n_samples)
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, y)
            
            # Check that model has correct number of features
            self.assertEqual(model.coef_.shape[1], n_features)
    
    def test_training_with_scaled_data(self):
        """Test that models train correctly with scaled data."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X_simple)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, self.y_simple)
        
        # Verify model is trained
        self.assertTrue(hasattr(model, 'coef_'))


class TestModelPredictions(unittest.TestCase):
    """Test cases for prediction format and range validation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.randn(20, 5)
        
        # Train models
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.lr_model.fit(self.X_train, self.y_train)
        
        self.dt_model = DecisionTreeClassifier(random_state=42)
        self.dt_model.fit(self.X_train, self.y_train)
        
        self.rf_model = RandomForestClassifier(random_state=42, n_estimators=10)
        self.rf_model.fit(self.X_train, self.y_train)
    
    def test_prediction_shape(self):
        """Test that predictions have the correct shape."""
        for model in [self.lr_model, self.dt_model, self.rf_model]:
            predictions = model.predict(self.X_test)
            self.assertEqual(predictions.shape[0], self.X_test.shape[0])
            self.assertEqual(len(predictions.shape), 1)
    
    def test_prediction_range(self):
        """Test that predictions are in the valid range (0 or 1 for binary classification)."""
        for model in [self.lr_model, self.dt_model, self.rf_model]:
            predictions = model.predict(self.X_test)
            self.assertTrue(np.all(np.isin(predictions, [0, 1])))
    
    def test_prediction_type(self):
        """Test that predictions are of correct type."""
        for model in [self.lr_model, self.dt_model, self.rf_model]:
            predictions = model.predict(self.X_test)
            self.assertTrue(isinstance(predictions, np.ndarray))
    
    def test_predict_proba_output(self):
        """Test that predict_proba returns valid probabilities."""
        for model in [self.lr_model, self.rf_model]:  # Decision tree also has predict_proba
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(self.X_test)
                # Check shape
                self.assertEqual(probas.shape[0], self.X_test.shape[0])
                self.assertEqual(probas.shape[1], 2)  # Binary classification
                # Check probabilities sum to 1
                self.assertTrue(np.allclose(probas.sum(axis=1), 1.0))
                # Check probabilities are in [0, 1]
                self.assertTrue(np.all(probas >= 0) and np.all(probas <= 1))


class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for evaluation metrics calculation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        self.y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])
    
    def test_accuracy_score_calculation(self):
        """Test that accuracy score is calculated correctly."""
        accuracy = accuracy_score(self.y_true, self.y_pred)
        # Manual calculation: 7 correct out of 10
        expected_accuracy = 0.7
        self.assertAlmostEqual(accuracy, expected_accuracy, places=5)
    
    def test_confusion_matrix_shape(self):
        """Test that confusion matrix has correct shape."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(cm.shape, (2, 2))
    
    def test_confusion_matrix_values(self):
        """Test that confusion matrix values are correct."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        # All values should be non-negative
        self.assertTrue(np.all(cm >= 0))
        # Sum should equal total predictions
        self.assertEqual(cm.sum(), len(self.y_true))
    
    def test_classification_report_format(self):
        """Test that classification report returns expected format."""
        report = classification_report(self.y_true, self.y_pred)
        self.assertIsInstance(report, str)
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1-score', report)
    
    def test_precision_recall_f1_consistency(self):
        """Test that precision, recall, and F1 are consistent."""
        precision = precision_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)
        
        # F1 should be harmonic mean of precision and recall
        if precision + recall > 0:
            expected_f1 = 2 * (precision * recall) / (precision + recall)
            self.assertAlmostEqual(f1, expected_f1, places=5)


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and error handling."""
    
    def test_single_sample_prediction(self):
        """Test prediction with a single sample."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test_single = np.random.randn(1, 5)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        prediction = model.predict(X_test_single)
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], [0, 1])
    
    def test_single_class_training(self):
        """Test handling of single-class dataset (should raise warning or error)."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.ones(50)  # All same class
        
        # Logistic Regression may struggle with single class
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # This should either raise an error or handle gracefully
        try:
            model.fit(X, y)
            # If it doesn't raise an error, predictions should still be the single class
            predictions = model.predict(X)
            self.assertTrue(np.all(predictions == 1))
        except ValueError:
            # Expected behavior for single-class dataset
            pass
    
    def test_empty_dataset_handling(self):
        """Test that empty dataset raises appropriate error."""
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        with self.assertRaises(ValueError):
            model.fit(X_empty, y_empty)
    
    def test_mismatched_dimensions(self):
        """Test that mismatched dimensions raise appropriate error."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test_wrong = np.random.randn(20, 10)  # Wrong number of features
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        with self.assertRaises(ValueError):
            model.predict(X_test_wrong)
    
    def test_nan_handling(self):
        """Test that NaN values are handled (should raise error or warning)."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan  # Introduce NaN
        y = np.random.randint(0, 2, 100)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Should raise ValueError for NaN values
        with self.assertRaises(ValueError):
            model.fit(X, y)


class TestReproducibility(unittest.TestCase):
    """Test cases for reproducibility with fixed random seeds."""
    
    def test_logistic_regression_reproducibility(self):
        """Test that Logistic Regression produces consistent results with fixed seed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)
        
        # Train first model
        model1 = LogisticRegression(random_state=42, max_iter=1000)
        model1.fit(X, y)
        pred1 = model1.predict(X_test)
        
        # Train second model with same seed
        model2 = LogisticRegression(random_state=42, max_iter=1000)
        model2.fit(X, y)
        pred2 = model2.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_decision_tree_reproducibility(self):
        """Test that Decision Tree produces consistent results with fixed seed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)
        
        # Train first model
        model1 = DecisionTreeClassifier(random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X_test)
        
        # Train second model with same seed
        model2 = DecisionTreeClassifier(random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_random_forest_reproducibility(self):
        """Test that Random Forest produces consistent results with fixed seed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)
        
        # Train first model
        model1 = RandomForestClassifier(random_state=42, n_estimators=10)
        model1.fit(X, y)
        pred1 = model1.predict(X_test)
        
        # Train second model with same seed
        model2 = RandomForestClassifier(random_state=42, n_estimators=10)
        model2.fit(X, y)
        pred2 = model2.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_train_test_split_reproducibility(self):
        """Test that train_test_split produces consistent splits with fixed seed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # First split
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Second split with same seed
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Splits should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline_execution(self):
        """Test that the full pipeline executes without errors."""
        np.random.seed(42)
        
        # Generate data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate models
        models = [
            LogisticRegression(random_state=42, max_iter=1000),
            DecisionTreeClassifier(random_state=42),
            RandomForestClassifier(random_state=42, n_estimators=10)
        ]
        
        for model in models:
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Assertions
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)
            self.assertEqual(cm.shape, (2, 2))
            self.assertIsInstance(report, str)
    
    def test_pipeline_with_different_test_sizes(self):
        """Test pipeline with different train-test split ratios."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        test_sizes = [0.1, 0.2, 0.3, 0.4]
        
        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Check that we get valid results
            self.assertIsNotNone(accuracy)
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
