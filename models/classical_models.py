from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_compare_classical_models(X_train, y_train, X_test, y_test):
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "mse": mse,
            "r2": r2
        }

        print(f"[{name}] MSE: {mse:.4f} | R²: {r2:.4f}")

    return results