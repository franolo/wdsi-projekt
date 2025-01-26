import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

EKSTRAKLASA_TEAMS = [
    'Legia Warszawa', 'Lech Poznań', 'Raków Częstochowa', 'Pogoń Szczecin',
    'Górnik Zabrze', 'Cracovia', 'Śląsk Wrocław', 'Warta Poznań',
    'Piast Gliwice', 'Jagiellonia Białystok', 'Widzew Łódź', 'Stal Mielec',
    'Korona Kielce', 'ŁKS Łódź', 'Radomiak Radom', 'Zagłębie Lubin'
]

# --------------------------------------------------
# ETAP 1: PRZYGOTOWANIE DANYCH
# --------------------------------------------------
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['Przeciwnik'].isin(EKSTRAKLASA_TEAMS)]
    
    df['Wieczorny_mecz'] = df['Godzina'].apply(lambda x: int(x.split(':')[0]) >= 18)
    df['Moc_przeciwnika'] = df.groupby('Przeciwnik')['Gole stracone'].transform('mean')
    df['Bilans'] = df['Gole strzelone'] - df['Gole stracone']
    
    df['Result'] = np.select(
        [df['Gole strzelone'] > df['Gole stracone'], 
         df['Gole strzelone'] == df['Gole stracone']],
        ['W', 'D'], default='L'
    )
    
    df[['Godzina', 'Minuta']] = df['Godzina'].str.split(':', expand=True).astype(int)
    df['Czas'] = df['Godzina'] * 60 + df['Minuta']
    
    return df[['Przeciwnik', 'Miejsce', 'Dzień_tygodnia', 'Czas', 'Waznosc', 
              'Wieczorny_mecz', 'Moc_przeciwnika', 'Bilans', 'Result', 
              'Gole strzelone', 'Gole stracone']]

# --------------------------------------------------
# ETAP 2: KODOWANIE PRZECIWNIKÓW
# --------------------------------------------------
def encode_opponents(X_train, y_train):
    temp_df = X_train[['Przeciwnik']].copy()
    temp_df['Result'] = y_train.values
    
    encoding = temp_df.groupby('Przeciwnik').agg({
        'Result': lambda x: (sum(x=='W') - sum(x=='L')) / (len(x)+3),
        'Przeciwnik': 'count'
    })
    encoding.columns = ['wynik_score', 'liczba_meczy']
    encoding['wspolczynnik'] = encoding['wynik_score'] * np.log1p(encoding['liczba_meczy'])
    
    encoded_values = X_train['Przeciwnik'].map(encoding['wspolczynnik']).fillna(0)
    return encoded_values, encoding['wspolczynnik'].to_dict()

# --------------------------------------------------
# ETAP 3: OPTYMALIZACJA
# --------------------------------------------------
def create_optimized_models():
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Miejsce', 'Dzień_tygodnia']),
        ('num', StandardScaler(), ['Czas', 'Waznosc', 'Moc_przeciwnika', 'Bilans']),
        ('interactions', PolynomialFeatures(degree=2, interaction_only=True), 
         ['Czas', 'Waznosc', 'Moc_przeciwnika'])
    ])

    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.7, 0.9]
    }

    classifier = GridSearchCV(
        Pipeline([
            ('preprocessor', preprocessor),
            ('model', CalibratedClassifierCV(
                base_estimator=GradientBoostingClassifier(random_state=42),
                cv=3,
                method='isotonic'
            ))
        ]),
        param_grid={'model__base_estimator__' + k: v for k, v in gb_params.items()},
        cv=5,
        scoring='accuracy'
    )

    goal_scored_model = GridSearchCV(
        Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(
                loss='poisson',
                random_state=42
            ))
        ]),
        param_grid=gb_params,
        cv=5,
        scoring='neg_mean_absolute_error'
    )

    return classifier, goal_scored_model, goal_scored_model

# --------------------------------------------------
# ETAP 4: TRENOWANIE MODELU
# --------------------------------------------------

def train_improved_models(df):
    required_cols = ['Przeciwnik', 'Miejsce', 'Dzień_tygodnia', 'Czas', 'Waznosc', 
                    'Gole strzelone', 'Gole stracone', 'Result']
    if not all(col in df.columns for col in required_cols):
        print("Błąd: Brak wymaganych kolumn w danych")
        return None, None, None, None

    try:
        X = df[['Przeciwnik', 'Miejsce', 'Dzień_tygodnia', 'Czas', 'Waznosc']]
        y_class = df['Result']
        y_scored = df['Gole strzelone']
        y_conceded = df['Gole stracone']
    except KeyError as e:
        print(f"Błąd przygotowania danych: {e}")
        return None, None, None, None

    class_dist = y_class.value_counts()
    print("\nRozkład klas przed przetwarzaniem:")
    print(class_dist)

    if len(class_dist) < 3 or class_dist.min() < 2:
        print("Błąd: Niewystarczająca liczba próbek w klasach")
        return None, None, None, None

    try:
        X_train, X_test, y_class_train, y_class_test, y_scored_train, y_scored_test, y_conceded_train, y_conceded_test = train_test_split(
            X, y_class, y_scored, y_conceded, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_class
        )
    except ValueError as e:
        print(f"Błąd podziału danych: {e}")
        return None, None, None, None

    try:
        X_train_encoded, encoding = encode_opponents(X_train, y_class_train)
        X_train = X_train.copy()
        X_train['Przeciwnik'] = X_train_encoded
    except Exception as e:
        print(f"Błąd kodowania przeciwników: {e}")
        return None, None, None, None

    categorical_features = ['Miejsce', 'Dzień_tygodnia']
    numerical_features = ['Czas', 'Waznosc']
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('przeciwnik', 'passthrough', ['Przeciwnik'])
    ])

    try:
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        feature_names = (
            preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist() +
            numerical_features +
            ['Przeciwnik']
        )
        X_train_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
        
        oversampler = SMOTE(sampling_strategy='auto', random_state=42)
        X_res, y_res = oversampler.fit_resample(X_train_df, y_class_train)
    except Exception as e:
        print(f"Błąd przetwarzania danych: {e}")
        return None, None, None, None

    try:
        classifier = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                random_state=42
            ))
        ])

        goal_scored_model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', TransformedTargetRegressor(
                regressor=GradientBoostingRegressor(
                    loss='squared_error',
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=42
                ),
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])

        goal_conceded_model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', TransformedTargetRegressor(
                regressor=GradientBoostingRegressor(
                    loss='squared_error',
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                ),
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])

        print("\nTrening klasyfikatora...")
        classifier.fit(X_train, y_class_train)

        print("\nTrening modelu goli strzelonych...")
        goal_scored_model.fit(X_train, y_scored_train)

        print("\nTrening modelu goli straconych...")
        goal_conceded_model.fit(X_train, y_conceded_train)

    except Exception as e:
        print(f"Błąd treningu modeli: {e}")
        return None, None, None, None

    try:
        X_test_processed = X_test.copy()
        X_test_processed['Przeciwnik'] = X_test['Przeciwnik'].map(encoding).fillna(0)
    except Exception as e:
        print(f"Błąd przetwarzania testowego: {e}")
        return None, None, None, None

    print("\n=== RAPORT KOŃCOWY ===")
    
    y_class_pred = classifier.predict(X_test_processed)
    print("\nRaport klasyfikacji:")
    print(classification_report(y_class_test, y_class_pred, target_names=['W', 'D', 'L']))
    
    y_scored_pred = goal_scored_model.predict(X_test_processed)
    y_conceded_pred = goal_conceded_model.predict(X_test_processed)
    
    print(f"\nMAE Gole strzelone: {mean_absolute_error(y_scored_test, y_scored_pred):.2f}")
    print(f"MAE Gole stracone: {mean_absolute_error(y_conceded_test, y_conceded_pred):.2f}")

    return classifier, goal_scored_model, goal_conceded_model, encoding
# --------------------------------------------------
# FUNKCJA PREDYKCYJNA Z FEEDBACK LOOP
# --------------------------------------------------
def predict_match(models, encoding, opponent, location, day, time, importance):
    classifier, goal_scored, goal_conceded = models
    
    try:
        if opponent not in encoding:
            return "Nieznany przeciwnik", ""
            
        hours, mins = map(int, time.split(':'))
        input_data = pd.DataFrame([{
            'Przeciwnik': encoding.get(opponent, 0),
            'Miejsce': location,
            'Dzień_tygodnia': day,
            'Czas': hours*60 + mins,
            'Waznosc': importance,
            'Wieczorny_mecz': int(hours >= 18),
            'Moc_przeciwnika': 0,  
            'Bilans': 0
        }])
        
        outcome = classifier.predict(input_data)[0]
        scored = max(0, int(round(goal_scored.predict(input_data)[0])))
        conceded = max(0, int(round(goal_conceded.predict(input_data)[0])))
        
        for _ in range(2):  
            if outcome == 'W' and scored <= conceded:
                input_data['przewidywana_przegrana'] = 1
                scored = int(round(goal_scored.predict(input_data)[0])) + 1
            elif outcome == 'L' and scored >= conceded:
                input_data['przewidywana_wygrana'] = 1
                conceded = int(round(goal_conceded.predict(input_data)[0])) + 1
            elif outcome == 'D' and scored != conceded:
                scored = conceded = (scored + conceded) // 2
        
        return outcome, f"{scored}:{conceded}"
    
    except Exception as e:
        print(f"Błąd: {e}")
        return "error", ""

# --------------------------------------------------
# URUCHOMIENIE SYSTEMU
# --------------------------------------------------
if __name__ == "__main__":
    df = prepare_data('dane.csv')
    
    if df is None:
        print("Błąd przygotowania danych")
        exit(1)
        
    models = train_improved_models(df)
    
    if None in models:
        print("\nNie udało się wytrenować modeli")
        exit(1)
        
    classifier, goal_scored, goal_conceded, encoding = models
    
    #przykładowa predykcja
    outcome, score = predict_match(
        (classifier, goal_scored, goal_conceded),
        encoding,
        opponent='Widzew Łódź',
        location='U siebie',
        day='Piątek',
        time='20:30',
        importance=0.6
    )
    
    print(f"\nPrzewidywany wynik: {outcome} {score}")