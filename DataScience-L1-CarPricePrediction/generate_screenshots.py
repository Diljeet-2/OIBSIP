import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car_data.csv"
OUT_DIR = BASE_DIR / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dataset_preview(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 2 + 0.25 * min(len(df), 10)))
    ax.axis('off')
    sample = df.head(10)
    table = ax.table(cellText=sample.values, colLabels=sample.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def correlation_heatmap(df: pd.DataFrame, out_path: Path):
    numeric = df.select_dtypes(include=['number']).copy()
    # add engineered Car_Age
    if 'Year' in numeric.columns:
        numeric['Car_Age'] = 2025 - numeric['Year']
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def feature_importance(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    df['Car_Age'] = 2025 - df['Year']
    df['Brand'] = df['Car_Name'].astype(str).apply(lambda x: x.split()[0])
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price', 'Car_Name'])
    X = pd.get_dummies(X, drop_first=True)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:20]

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind='barh', ax=ax)
    ax.invert_yaxis()
    ax.set_title('Feature Importances')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def actual_vs_predicted(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    df['Car_Age'] = 2025 - df['Year']
    df['Brand'] = df['Car_Name'].astype(str).apply(lambda x: x.split()[0])
    y = df['Selling_Price']
    X = df.drop(columns=['Selling_Price', 'Car_Name'])
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.7)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, '--r')
    ax.set_xlabel('Actual Selling Price')
    ax.set_ylabel('Predicted Selling Price')
    ax.set_title('Actual vs Predicted')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)

    dataset_preview(df, OUT_DIR / 'dataset_preview.png')
    correlation_heatmap(df, OUT_DIR / 'correlation_heatmap.png')
    feature_importance(df, OUT_DIR / 'feature_importance.png')
    actual_vs_predicted(df, OUT_DIR / 'actual_vs_predicted.png')


if __name__ == '__main__':
    main()
