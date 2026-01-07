import pandas as pd
import numpy as np
import logging
import joblib
from typing import List, Dict, Optional, Tuple, Any, Callable
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .utils import DataLoadError

class ShuttlePredictorEngine:
    """
    Core engine for the Shuttle Predictor application.
    Handles data management, model training, and prediction logic.
    Implements the Observer pattern for UI updates.
    """
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.model: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.accuracy: Optional[float] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.y_pred: Optional[np.ndarray] = None
        self.model_name: str = "Logistic Regression"
        self._observers: List[Callable[[str, Any], None]] = []

    def add_observer(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for state updates."""
        self._observers.append(callback)

    def notify_observers(self, event_type: str, data: Any = None) -> None:
        """Notify all observers of a state change."""
        for callback in self._observers:
            callback(event_type, data)

    def load_data(self, file_path: Optional[str] = None) -> None:
        """Load data from CSV or generate mock data."""
        try:
            if file_path:
                logging.info(f"Loading data from {file_path}")
                self.data = pd.read_csv(file_path)
            else:
                logging.info("Generating mock data")
                self._generate_mock_data()
            self.notify_observers("data_loaded", len(self.data))
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise DataLoadError(f"Failed to load data: {e}")

    def _generate_mock_data(self) -> None:
        """Generate realistic mock data."""
        np.random.seed(42)
        n_samples = 1000
        
        routes = np.random.choice(
            ["Campus Loop", "Grove Village Route", "Al Hamra Link", "RAK City Express"],
            n_samples,
            p=[0.3, 0.3, 0.25, 0.15]
        )
        
        stop_distances = np.random.uniform(0, 25, n_samples)
        
        days = np.random.choice(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            n_samples,
            p=[0.2, 0.2, 0.2, 0.2, 0.15, 0.03, 0.02]
        )
        
        times = np.random.normal(12, 4, n_samples)
        times = np.clip(times, 0, 23)
        
        traffic_conditions = np.random.choice(
            ["Low", "Medium", "High"],
            n_samples,
            p=[0.3, 0.5, 0.2]
        )
        
        weather_conditions = np.random.choice(
            ["Clear", "Cloudy", "Dusty", "Foggy", "Rainy"],
            n_samples,
            p=[0.4, 0.25, 0.15, 0.1, 0.1]
        )
        
        special_events = np.random.choice(
            ["None", "Sports Match", "Career Fair", "University Exam", "Open Day"],
            n_samples,
            p=[0.7, 0.15, 0.08, 0.05, 0.02]
        )
        
        temperatures = np.random.normal(30, 8, n_samples)
        temperatures = np.clip(temperatures, 20, 45)
        
        delays = np.random.normal(3, 8, n_samples)
        delays += np.where(traffic_conditions == "High", 5, 0)
        delays += np.where(weather_conditions == "Dusty", 3, 0)
        delays += np.where(weather_conditions == "Foggy", 4, 0)
        delays += np.where(special_events != "None", 2, 0)
        delays += np.where((times >= 7) & (times <= 9), 2, 0)
        delays += np.where((times >= 17) & (times <= 19), 3, 0)
        
        arrival_status = np.where(delays <= 5, "On-Time", "Late")
        
        self.data = pd.DataFrame({
            'route': routes,
            'stop_distance_km': stop_distances,
            'day_of_week': days,
            'time_of_day': times,
            'traffic_condition': traffic_conditions,
            'weather': weather_conditions,
            'special_event': special_events,
            'temperature_celsius': temperatures,
            'delay_minutes': delays,
            'arrival_status': arrival_status
        })

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive new features from existing columns."""
        df = df.copy()
        
        # 1. Is_Rush_Hour
        # Check if time_of_day is numeric or categorical
        if pd.api.types.is_numeric_dtype(df['time_of_day']):
            # Numeric: 7-9 AM and 17-19 PM
            df['is_rush_hour'] = df['time_of_day'].apply(
                lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
            )
        else:
            # Categorical: Morning and Evening are rush hours
            df['is_rush_hour'] = df['time_of_day'].apply(
                lambda x: 1 if x in ['Morning', 'Evening'] else 0
            )
        
        # 2. Is_Weekend: Saturday or Sunday
        weekend_days = ['Saturday', 'Sunday']
        df['is_weekend'] = df['day_of_week'].apply(
            lambda x: 1 if x in weekend_days else 0
        )
        
        return df

    def train_model(self, model_name: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the model and return metrics."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        self.model_name = model_name
        
        # Apply feature engineering
        data_with_features = self._engineer_features(self.data)
        
        feature_columns = [
            'route', 'stop_distance_km', 'day_of_week', 'time_of_day',
            'traffic_condition', 'weather', 'special_event', 'temperature_celsius',
            'is_rush_hour', 'is_weekend'
        ]
        
        X = data_with_features[feature_columns].copy()
        y = data_with_features['arrival_status']
        X['special_event'] = X['special_event'].fillna('None')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Dynamically determine if time_of_day is numeric or categorical
        if pd.api.types.is_numeric_dtype(X_train['time_of_day']):
            # Numeric time_of_day
            categorical_features = ['route', 'day_of_week', 'traffic_condition', 'weather', 'special_event']
            numerical_features = ['stop_distance_km', 'time_of_day', 'temperature_celsius', 'is_rush_hour', 'is_weekend']
        else:
            # Categorical time_of_day
            categorical_features = ['route', 'day_of_week', 'time_of_day', 'traffic_condition', 'weather', 'special_event']
            numerical_features = ['stop_distance_km', 'temperature_celsius', 'is_rush_hour', 'is_weekend']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Define base model and param grid
        if model_name == "Logistic Regression":
            clf = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {'classifier__C': [0.1, 1.0, 10.0]}
        elif model_name == "Decision Tree Classifier":
            clf = DecisionTreeClassifier(random_state=42)
            param_grid = {'classifier__max_depth': [5, 10, 20, None]}
        elif model_name == "Random Forest Classifier":
            clf = RandomForestClassifier(random_state=42)
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5]
            }
        elif model_name == "Gradient Boosting Classifier":
            clf = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5]
            }
        else:
            # Fallback
            clf = LogisticRegression(random_state=42)
            param_grid = {}

        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', clf)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.accuracy = accuracy_score(y_test, y_pred)
        
        # Binarize for metrics: Late=1, On-Time=0
        pos_label = 'Late'
        
        precision = precision_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label, average='binary', zero_division=0)
        
        roc_auc = None
        if hasattr(self.model, "predict_proba"):
            try:
                # Find index of positive class
                classes = list(self.model.named_steps['classifier'].classes_)
                if pos_label in classes:
                    pos_idx = classes.index(pos_label)
                    y_prob = self.model.predict_proba(X_test)[:, pos_idx]
                    roc_auc = roc_auc_score((y_test == pos_label).astype(int), y_prob)
            except Exception:
                pass

        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        metrics = {
            "accuracy": self.accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "best_params": best_params
        }
        
        self.notify_observers("training_complete", metrics)
        return metrics

    def predict(self, input_record: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a prediction for a single record."""
        if self.model is None:
            raise ValueError("Model not trained")
            
        input_data = pd.DataFrame([input_record])
        input_data['special_event'] = input_data['special_event'].fillna('None')
        
        # Apply feature engineering
        input_data = self._engineer_features(input_data)
        
        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        confidence = float(max(probabilities) * 100)
        
        class_names = list(self.model.named_steps['classifier'].classes_)
        on_time_index = class_names.index('On-Time')
        on_time_prob = probabilities[on_time_index] * 100
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'on_time_prob': on_time_prob,
            'context': input_record
        }
    
    def get_feature_importances(self) -> List[Tuple[str, float]]:
        """Get feature importances."""
        if self.model is None:
            return []
        try:
            model_step = self.model.named_steps['classifier']
            feature_names = self.preprocessor.get_feature_names_out()
            if hasattr(model_step, 'coef_'):
                importances = np.abs(model_step.coef_[0])
            elif hasattr(model_step, 'feature_importances_'):
                importances = model_step.feature_importances_
            else:
                return []
            pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
            return [(name, float(value)) for name, value in pairs[:8]]
        except Exception:
            return []

    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        if self.model:
            joblib.dump(self.model, path)
            logging.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load the model from disk."""
        self.model = joblib.load(path)
        logging.info(f"Model loaded from {path}")
        self.notify_observers("model_loaded", path)
