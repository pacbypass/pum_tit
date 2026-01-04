import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(1337)

def main():
    df = pd.read_csv('titanic_raw.csv')
    print(f"Dane: {df.shape[0]} pasazerow, {df.shape[1]} cech")

    # 1. Eda przed czyszczeniem
    print("\n--- eksploracyjna analiza danych ---")

    df_viz = df.copy()
    # ? -> NaN dla wieku
    df_viz['age'] = pd.to_numeric(df_viz['age'], errors='coerce')

    # Podstawowe statystyki dla kolumn numerycznych (dane surowe z '?' jako stringi)
    print("\nPodstawowe statystyki (dane surowe - '?' traktowane jako stringi):")
    # Dla wieku trzeba obsluzyc stringi '?'
    age_counts_before = df['age'].apply(lambda x: isinstance(x, str) and x == '?').sum()
    print(f"  Wiek: {age_counts_before} wartosci '?' ({age_counts_before/len(df):.1%})")
    print("  Inne kolumny numeryczne (sibsp, parch, fare z '?'):")
    num_cols = ['fare', 'sibsp', 'parch']
    for col in num_cols:
        question_marks = df[col].apply(lambda x: isinstance(x, str) and x == '?').sum()
        if question_marks > 0:
            print(f"    {col}: {question_marks} wartosci '?'")

    # WIZUALIZACJA: Utworz wykres porownania rozkladu wieku
    print("\nTworzenie wizualizacji rozkladu wieku PRZED czyszczeniem...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lewy panel: Rozklad wieku PRZED czyszczeniem i imputacją
    axes[0].hist(df_viz['age'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_title('Rozklad wieku (PRZED czyszczeniem)', fontsize=14)
    axes[0].set_xlabel('Wiek', fontsize=12)
    axes[0].set_ylabel('Liczba', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    # Ustaw etykiety calkowite na osi X
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # oblicz statystyki dla danych surowych
    age_raw = df_viz['age'].dropna()
    if len(age_raw) > 0:
        axes[0].axvline(age_raw.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Srednia: {age_raw.mean():.1f}')
        axes[0].axvline(age_raw.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Mediana: {age_raw.median():.1f}')
        axes[0].legend()

    # prawy panel zostanie wypelniony PO czyszczeniu (tymczasowy placeholder)
    axes[1].text(0.5, 0.5, 'Rozklad wieku (PO czyszczeniu)\nZostanie pokazany po przetworzeniu danych',
                horizontalalignment='center', verticalalignment='center',
                transform=axes[1].transAxes, fontsize=12)
    axes[1].set_title('Rozklad wieku (PO czyszczeniu)', fontsize=14)
    axes[1].set_xlabel('Wiek', fontsize=12)
    axes[1].set_ylabel('Liczba', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # eda wykresy
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Rozklad wieku
    axes2[0].hist(df_viz['age'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes2[0].set_title('Rozklad wieku pasazerow', fontsize=14)
    axes2[0].set_xlabel('Wiek', fontsize=12)
    axes2[0].set_ylabel('Liczba pasazerow', fontsize=12)
    axes2[0].grid(True, alpha=0.3)
    axes2[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # panel 2: Wspolczynnik przezycia wedlug klasy
    survival_by_class = df_viz.groupby('pclass')['survived'].mean() * 100
    classes = ['1', '2', '3']
    survival_rates = [survival_by_class.get(1, 0), survival_by_class.get(2, 0), survival_by_class.get(3, 0)]
    bars = axes2[1].bar(classes, survival_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    axes2[1].set_title('Wspolczynnik przezycia wedlug klasy', fontsize=14)
    axes2[1].set_xlabel('Klasa', fontsize=12)
    axes2[1].set_ylabel('Przezycie (%)', fontsize=12)
    axes2[1].grid(True, alpha=0.3, axis='y')
    # etykiety
    for bar, rate in zip(bars, survival_rates):
        height = bar.get_height()
        axes2[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    # panel 3: Wspolczynnik przezycia wedlug plci
    survival_by_sex = df_viz.groupby('sex')['survived'].mean() * 100
    sexes = ['Mezczyzni', 'Kobiety']

    male_rate = survival_by_sex.get('male', 0) if 'male' in survival_by_sex.index else 0
    female_rate = survival_by_sex.get('female', 0) if 'female' in survival_by_sex.index else 0
    
    # Obsluz mozliwe NaN - just in case
    if pd.isna(male_rate): male_rate = 0
    if pd.isna(female_rate): female_rate = 0
    sex_rates = [male_rate, female_rate]
    bars2 = axes2[2].bar(sexes, sex_rates, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    axes2[2].set_title('Wspolczynnik przezycia wedlug plci', fontsize=14)
    axes2[2].set_xlabel('Plec', fontsize=12)
    axes2[2].set_ylabel('Przezycie (%)', fontsize=12)
    axes2[2].grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars2, sex_rates):
        height = bar.get_height()
        axes2[2].text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('titanic_eda_plots.png', dpi=150, bbox_inches='tight')
    print("Zapisano trojpanelowy wykres EDA: titanic_eda_plots.png")
    plt.close(fig2)

    # 2. czyszczenie danych - Zamiana '?' na NaN i utworzenie oczyszczonego dataframe
    print("\n--- czyszczenie danych ---")
    df_clean = df.copy()

    # Zamien '?' na NaN dla wszystkich kolumn, ktore moga je zawierac
    for col in df_clean.columns:
        if df_clean[col].apply(lambda x: isinstance(x, str) and x == '?').any():
            if col in ['age', 'fare', 'sibsp', 'parch']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            else:
                df_clean[col] = df_clean[col].replace('?', np.nan)

    # 3. tytuly ludzi tzn. doktor, pan, pani, etc.
    df_clean['title'] = df_clean['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Dr', 'Rev': 'Rev', 'Col': 'Military', 'Major': 'Military',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Mrs', 'Capt': 'Military',
        'Countess': 'Noble', 'Don': 'Noble', 'Dona': 'Noble',
        'Jonkheer': 'Noble', 'Lady': 'Noble', 'Sir': 'Noble'
    }
    df_clean['title'] = df_clean['title'].map(title_map).fillna('Other')

    # 4. uzupelnienie brakujacych wartosci
    title_age = df_clean.groupby('title')['age'].mean()
    df_clean['age'] = df_clean.apply(
        lambda row: title_age[row['title']] if pd.isna(row['age']) else row['age'], axis=1
    )

    # Dla pozostalych brakujacych wartosci wieku, uzyj sredniej wedlug plci - imputacja wieku
    if df_clean['age'].isnull().any():
        age_by_sex = df_clean.groupby('sex')['age'].mean()
        remaining_missing = df_clean['age'].isnull().sum()
        df_clean['age'] = df_clean.apply(
            lambda row: age_by_sex[row['sex']] if pd.isna(row['age']) else row['age'], axis=1
        )
        print(f"  Wiek: uzupelniono {remaining_missing} pozostalych wartosci wedlug plci")

    # embarked (moda - najczestsza wartosc)
    df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])

    # fare wedlug mediany klasy
    fare_by_class = df_clean.groupby('pclass')['fare'].median()
    df_clean['fare'] = df_clean.apply(
        lambda row: fare_by_class[row['pclass']] if pd.isna(row['fare']) else row['fare'], axis=1
    )

    # cabin jako zmienna binarna
    df_clean['has_cabin'] = df_clean['cabin'].notnull().astype(int)

    # 5. inzynieria cech
    df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
    df_clean['age_range'] = pd.cut(df_clean['age'],
                                   bins=[0, 6, 12, 18, 100],
                                   labels=['Bobas', 'Dzieciak', 'Nastolatek', 'Dorosły'],
                                   right=False)
    df_clean['mpc'] = df_clean['age'] * df_clean['pclass']
    df_clean['is_child'] = (df_clean['age'] < 18).astype(int)
    df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)
    df_clean['fare_per_person'] = df_clean['fare'] / df_clean['family_size']

    # 6. obsluga wartosci odstajajacych
    df_clean.loc[df_clean['age'] > 67, 'age'] = 67
    Q1, Q3 = df_clean['fare'].quantile(0.25), df_clean['fare'].quantile(0.75)
    fare_upper = Q3 + 1.5 * (Q3 - Q1)
    df_clean.loc[df_clean['fare'] > fare_upper, 'fare'] = fare_upper
    # aktualizacja pochodnych cech
    df_clean['mpc'] = df_clean['age'] * df_clean['pclass']
    df_clean['fare_per_person'] = df_clean['fare'] / df_clean['family_size']

    # aktualizacja wizualizacji: rozklad wieku PO czyszczeniu
    print("\nAktualizowanie wizualizacji rozkladu wieku PO czyszczeniu...")
    # wyczysc tekst zastepczy
    axes[1].clear()
    # narysuj histogram oczyszczonego wieku
    axes[1].hist(df_clean['age'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_title('Rozklad wieku (PO czyszczeniu)', fontsize=14)
    axes[1].set_xlabel('Wiek', fontsize=12)
    axes[1].set_ylabel('Liczba', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # oblicz statystyki dla oczyszczonych danych
    age_clean = df_clean['age'].dropna()
    if len(age_clean) > 0:
        axes[1].axvline(age_clean.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Srednia: {age_clean.mean():.1f}')
        axes[1].axvline(age_clean.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Mediana: {age_clean.median():.1f}')
        axes[1].legend()

    # zapisz zaktualizowany wykres
    plt.tight_layout()
    plt.savefig('titanic_age_comparison.png', dpi=150, bbox_inches='tight')
    print("Zapisano wykres porownania wieku: titanic_age_comparison.png")
    plt.close(fig)

    # 7. konwersja typow kategorycznych
    df_clean['pclass'] = df_clean['pclass'].astype('category')
    df_clean['sex'] = df_clean['sex'].astype('category')
    df_clean['embarked'] = df_clean['embarked'].astype('category')
    # age_range jest juz kategoryczny z pd.cut

    # 8. info kto przezyl i umarl
    features = ['sex', 'age', 'age_range', 'pclass', 'fare', 'family_size',
                'mpc', 'has_cabin', 'is_child', 'is_alone', 'embarked', 'title']
    X = df_clean[features].copy()
    y = df_clean['survived'].copy()

    print(f"{y.sum()} przezylo ({y.mean():.1%}), {len(y)-y.sum()} umarlo")

    # 9. podzial na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1337, stratify=y
    )
    print(f"Podzial: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # 10. kodowanie kategoryczne i normalizacja (tylko na danych treningowych)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # normalizacja cech numerycznych uzywajac MinMaxScaler (fit on train, transform on test)
    num_cols = ['age', 'fare', 'family_size', 'mpc']
    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 11. TRAIN MODELS
    models = {
        'Drzewo Decyzyjne': DecisionTreeClassifier(random_state=1337),
        'Regresja Logistyczna': LogisticRegression(random_state=1337, max_iter=1000),
        'K-NN': KNeighborsClassifier(),
        'SVM': SVC(random_state=1337, probability=True)
    }

    results = {}
    print("\n--- Wyniki modelowania ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results[name] = {'accuracy': acc, 'model': model, 'y_pred': y_pred}
        print(f"{name:20} Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

    best_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"Najlepszy model: {best_name}")

    # 14. dane wyjsciowe / debug / info / wszystko
    sorted_models = sorted([(n, r['accuracy']) for n, r in results.items()],
                          key=lambda x: x[1], reverse=True)

    print("\n" + "="*50)
    print("podsumowanie")
    print("="*50)
    for name, acc in sorted_models:
        status = "✓ PASS" if acc >= 0.80 else "✗ FAIL"
        print(f"{name:25} {acc:.4f}  {status}")

    best_acc = sorted_models[0][1]
    print(f"\nBest Accuracy: {best_acc:.2%}")
    if best_acc >= 0.80:
        print("cel osiagniety (>=80%)")
    else:
        print("cel nieosiagniety")

    # 15. zapisz oczyszczone dane
    fname = 'titanic_processed_short.csv'
    df_clean.to_csv(fname, index=False)
    print(f"\nZapisano dane: {fname}")

    return {
        'df': df_clean,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'results': results,
        'best_accuracy': best_acc
    }

if __name__ == "__main__":
    main()