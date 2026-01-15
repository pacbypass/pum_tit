import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

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

    # WIZUALIZACJA: Brakujace wartosci
    print("\nTworzenie wizualizacji brakujacych wartosci...")
    missing_counts = df.isnull().sum() + df.apply(lambda x: x == '?').sum()
    missing_percent = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({'Kolumna': missing_percent.index, 'Brakujace (%)': missing_percent.values})
    missing_df = missing_df[missing_df['Brakujace (%)'] > 0].sort_values('Brakujace (%)', ascending=False)

    fig_missing, ax_missing = plt.subplots(figsize=(10, 6))
    bars = ax_missing.barh(missing_df['Kolumna'], missing_df['Brakujace (%)'], color='skyblue')
    ax_missing.set_xlabel('Procent brakujacych wartosci (%)', fontsize=12)
    ax_missing.set_title('Brakujace wartosci w danych surowych', fontsize=14)
    ax_missing.grid(True, alpha=0.3, axis='x')
    for bar, value in zip(bars, missing_df['Brakujace (%)']):
        ax_missing.text(value + 0.5, bar.get_y() + bar.get_height()/2, f'{value:.1f}%',
                       va='center', ha='left', fontsize=10)
    plt.tight_layout()
    plt.savefig('titanic_missing_values.png', dpi=150, bbox_inches='tight')
    print("Zapisano wykres brakujacych wartosci: titanic_missing_values.png")
    plt.close(fig_missing)

    # WIZUALIZACJA: Analiza korelacji cech numerycznych
    print("\nTworzenie analizy korelacji cech numerycznych...")
    # Wybierz kolumny numeryczne do analizy korelacji
    numeric_cols_for_corr = ['age', 'fare', 'sibsp', 'parch', 'pclass', 'survived']
    # Konwersja '?' na NaN dla kolumn numerycznych
    df_corr = df.copy()
    for col in numeric_cols_for_corr:
        if col in df_corr.columns:
            df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')

    # Oblicz macierz korelacji
    correlation_matrix = df_corr[numeric_cols_for_corr].corr()

    # Wizualizacja macierzy korelacji
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    im = ax_corr.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Dodaj etykiety
    ax_corr.set_xticks(np.arange(len(numeric_cols_for_corr)))
    ax_corr.set_yticks(np.arange(len(numeric_cols_for_corr)))
    ax_corr.set_xticklabels(numeric_cols_for_corr, rotation=45, ha='right')
    ax_corr.set_yticklabels(numeric_cols_for_corr)

    # Dodaj wartosci korelacji w komorkach
    for i in range(len(numeric_cols_for_corr)):
        for j in range(len(numeric_cols_for_corr)):
            text = ax_corr.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center',
                               color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')

    ax_corr.set_title('Macierz korelacji cech numerycznych', fontsize=14)
    plt.colorbar(im, ax=ax_corr)
    plt.tight_layout()
    plt.savefig('titanic_correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("Zapisano macierz korelacji: titanic_correlation_matrix.png")
    plt.close(fig_corr)



    # WIZUALIZACJA: Rozklad cech numerycznych przed czyszczeniem
    print("\nTworzenie wizualizacji rozkladu cech numerycznych PRZED czyszczeniem...")
    # Konwersja kolumn numerycznych z '?' na NaN dla wizualizacji
    num_cols_viz = ['fare', 'sibsp', 'parch']
    for col in num_cols_viz:
        df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')

    fig_num, axes_num = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    titles = ['Oplata za bilet (fare)', 'Rodzenstwo/malzonkowie (sibsp)', 'Rodzice/dzieci (parch)']
    columns = ['fare', 'sibsp', 'parch']

    for _, (ax, col, color, title) in enumerate(zip(axes_num, columns, colors, titles)):
        # Usuniecie NaN dla histogramu
        data = df_viz[col].dropna()
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color=color)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Liczba pasazerow', fontsize=12)
        ax.grid(True, alpha=0.3)
        if col == 'fare':
            # Dla fare ustawiamy ograniczenie do 99 percentyla, aby uniknac skrajnych wartosci
            percentile_99 = data.quantile(0.99)
            ax.set_xlim([0, percentile_99])
        elif col in ['sibsp', 'parch']:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig('titanic_numeric_distributions_before.png', dpi=150, bbox_inches='tight')
    print("Zapisano wykres rozkladu cech numerycznych: titanic_numeric_distributions_before.png")
    plt.close(fig_num)

    # WIZUALIZACJA: Boxplot wieku według klasy i przeżycia
    print("\nTworzenie boxplotu wieku według klasy i przeżycia...")
    # Przygotuj dane - usuń NaN z wieku
    df_boxplot = df_viz.copy()
    df_boxplot['age'] = pd.to_numeric(df_boxplot['age'], errors='coerce')
    df_boxplot = df_boxplot.dropna(subset=['age', 'pclass', 'survived'])

    fig_box, axes_box = plt.subplots(1, 2, figsize=(14, 6))

    # Panel lewy: Boxplot wieku według klasy
    class_data = [df_boxplot[df_boxplot['pclass'] == i]['age'] for i in [1, 2, 3]]
    bp1 = axes_box[0].boxplot(class_data, labels=['Klasa 1', 'Klasa 2', 'Klasa 3'],
                              patch_artist=True, medianprops={'color': 'red', 'linewidth': 2})
    # Kolorowanie pudełek
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axes_box[0].set_title('Rozkład wieku według klasy', fontsize=14)
    axes_box[0].set_xlabel('Klasa', fontsize=12)
    axes_box[0].set_ylabel('Wiek', fontsize=12)
    axes_box[0].grid(True, alpha=0.3, axis='y')

    # Panel prawy: Boxplot wieku według przeżycia
    survived_data = [df_boxplot[df_boxplot['survived'] == i]['age'] for i in [0, 1]]
    bp2 = axes_box[1].boxplot(survived_data, labels=['Nie przeżył', 'Przeżył'],
                              patch_artist=True, medianprops={'color': 'red', 'linewidth': 2})
    colors2 = ['lightgray', 'lightyellow']
    for patch, color in zip(bp2['boxes'], colors2):
        patch.set_facecolor(color)
    axes_box[1].set_title('Rozkład wieku według przeżycia', fontsize=14)
    axes_box[1].set_xlabel('Przeżycie', fontsize=12)
    axes_box[1].set_ylabel('Wiek', fontsize=12)
    axes_box[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('titanic_age_boxplots.png', dpi=150, bbox_inches='tight')
    print("Zapisano boxplot wieku: titanic_age_boxplots.png")
    plt.close(fig_box)

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
    fare_before = df_clean['fare'].copy()  # zapisz przed przycieciem
    df_clean.loc[df_clean['age'] > 67, 'age'] = 67
    Q1, Q3 = df_clean['fare'].quantile(0.25), df_clean['fare'].quantile(0.75)
    fare_upper = Q3 + 1.5 * (Q3 - Q1)
    df_clean.loc[df_clean['fare'] > fare_upper, 'fare'] = fare_upper
    # aktualizacja pochodnych cech
    df_clean['mpc'] = df_clean['age'] * df_clean['pclass']
    df_clean['fare_per_person'] = df_clean['fare'] / df_clean['family_size']

    # WIZUALIZACJA: Porownanie rozkladu oplate przed i po przycieciu wartosci odstajacych
    print("\nTworzenie wizualizacji porownania rozkladu oplate przed i po przycieciu...")
    fig_fare, axes_fare = plt.subplots(1, 2, figsize=(14, 5))

    # Panel lewy: rozklad przed przycieciem
    axes_fare[0].hist(fare_before, bins=30, edgecolor='black', alpha=0.7, color='lightblue')
    axes_fare[0].axvline(fare_before.mean(), color='red', linestyle='--', linewidth=2, label=f'Srednia: {fare_before.mean():.1f}')
    axes_fare[0].axvline(fare_before.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {fare_before.median():.1f}')
    axes_fare[0].set_title('Rozklad oplate przed przycieciem', fontsize=14)
    axes_fare[0].set_xlabel('Oplata', fontsize=12)
    axes_fare[0].set_ylabel('Liczba', fontsize=12)
    axes_fare[0].grid(True, alpha=0.3)
    axes_fare[0].legend()

    # Panel prawy: rozklad po przycieciu
    axes_fare[1].hist(df_clean['fare'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    axes_fare[1].axvline(df_clean['fare'].mean(), color='red', linestyle='--', linewidth=2, label=f'Srednia: {df_clean["fare"].mean():.1f}')
    axes_fare[1].axvline(df_clean['fare'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df_clean["fare"].median():.1f}')
    axes_fare[1].set_title('Rozklad oplate po przycieciu', fontsize=14)
    axes_fare[1].set_xlabel('Oplata', fontsize=12)
    axes_fare[1].set_ylabel('Liczba', fontsize=12)
    axes_fare[1].grid(True, alpha=0.3)
    axes_fare[1].legend()

    plt.tight_layout()
    plt.savefig('titanic_fare_distribution_comparison.png', dpi=150, bbox_inches='tight')
    print("Zapisano wykres porownania rozkladu oplate: titanic_fare_distribution_comparison.png")
    plt.close(fig_fare)

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

    # WIZUALIZACJA: Dodatkowe analizy EDA po czyszczeniu danych
    print("\nTworzenie dodatkowych wizualizacji EDA po czyszczeniu danych...")

    # Przygotuj dane do wizualizacji
    # 1. Przeżycie według grup wiekowych (age_range)
    survival_by_age_range = df_clean.groupby('age_range')['survived'].mean() * 100

    # 2. Przeżycie według rozmiaru rodziny (family_size) - grupujemy większe rodziny
    df_clean['family_size_group'] = df_clean['family_size'].apply(
        lambda x: str(x) if x <= 4 else '5+'
    )
    survival_by_family_size = df_clean.groupby('family_size_group')['survived'].mean() * 100

    # 3. Przeżycie według portu zaokrętowania (embarked)
    survival_by_embarked = df_clean.groupby('embarked')['survived'].mean() * 100
    embarked_labels = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}

    # 4. Przeżycie według tytułu (title)
    survival_by_title = df_clean.groupby('title')['survived'].mean() * 100

    # Stwórz wykres 2x2
    fig_eda2, axes_eda2 = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Przeżycie według grup wiekowych
    age_ranges = survival_by_age_range.index.tolist()
    age_survival_rates = survival_by_age_range.values.tolist()
    bars1 = axes_eda2[0, 0].bar(age_ranges, age_survival_rates, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'], alpha=0.7)
    axes_eda2[0, 0].set_title('Współczynnik przeżycia według grup wiekowych', fontsize=14)
    axes_eda2[0, 0].set_xlabel('Grupa wiekowa', fontsize=12)
    axes_eda2[0, 0].set_ylabel('Przeżycie (%)', fontsize=12)
    axes_eda2[0, 0].grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars1, age_survival_rates):
        height = bar.get_height()
        axes_eda2[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    # Panel 2: Przeżycie według rozmiaru rodziny
    family_sizes = sorted(survival_by_family_size.index.tolist(), key=lambda x: int(x) if x != '5+' else 5)
    family_survival_rates = [survival_by_family_size[fs] for fs in family_sizes]
    bars2 = axes_eda2[0, 1].bar(family_sizes, family_survival_rates, color='#66b3ff', alpha=0.7)
    axes_eda2[0, 1].set_title('Współczynnik przeżycia według rozmiaru rodziny', fontsize=14)
    axes_eda2[0, 1].set_xlabel('Rozmiar rodziny', fontsize=12)
    axes_eda2[0, 1].set_ylabel('Przeżycie (%)', fontsize=12)
    axes_eda2[0, 1].grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars2, family_survival_rates):
        height = bar.get_height()
        axes_eda2[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    # Panel 3: Przeżycie według portu zaokrętowania
    embarked_codes = sorted(survival_by_embarked.index.tolist())
    embarked_survival_rates = [survival_by_embarked[code] for code in embarked_codes]
    embarked_names = [embarked_labels.get(code, code) for code in embarked_codes]
    bars3 = axes_eda2[1, 0].bar(embarked_names, embarked_survival_rates, color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.7)
    axes_eda2[1, 0].set_title('Współczynnik przeżycia według portu zaokrętowania', fontsize=14)
    axes_eda2[1, 0].set_xlabel('Port zaokrętowania', fontsize=12)
    axes_eda2[1, 0].set_ylabel('Przeżycie (%)', fontsize=12)
    axes_eda2[1, 0].grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars3, embarked_survival_rates):
        height = bar.get_height()
        axes_eda2[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    # Panel 4: Przeżycie według tytułu
    titles = survival_by_title.index.tolist()
    title_survival_rates = survival_by_title.values.tolist()
    bars4 = axes_eda2[1, 1].bar(titles, title_survival_rates, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6'], alpha=0.7)
    axes_eda2[1, 1].set_title('Współczynnik przeżycia według tytułu', fontsize=14)
    axes_eda2[1, 1].set_xlabel('Tytuł', fontsize=12)
    axes_eda2[1, 1].set_ylabel('Przeżycie (%)', fontsize=12)
    axes_eda2[1, 1].grid(True, alpha=0.3, axis='y')
    axes_eda2[1, 1].tick_params(axis='x', rotation=45)
    for bar, rate in zip(bars4, title_survival_rates):
        height = bar.get_height()
        axes_eda2[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('titanic_additional_eda.png', dpi=150, bbox_inches='tight')
    print("Zapisano dodatkowe wizualizacje EDA: titanic_additional_eda.png")
    plt.close(fig_eda2)

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

    # 11. Dostrajanie hiperparametrow (Grid Search)
    print("\n--- Dostrajanie hiperparametrow (Grid Search) ---")

    # Definicja siatki parametrow dla kazdego modelu
    param_grids = {
        'Drzewo Decyzyjne': {
            'max_depth': [3, None],
            'min_samples_split': [2]
        },
        'Regresja Logistyczna': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs']
        },
        'K-NN': {
            'n_neighbors': [5, 7],
            'weights': ['uniform']
        },
        'SVM': {
            'C': [1, 10],
            'kernel': ['linear']
        }
    }

    # Bazowe modele
    base_models = {
        'Drzewo Decyzyjne': DecisionTreeClassifier(random_state=1337),
        'Regresja Logistyczna': LogisticRegression(random_state=1337, max_iter=1000),
        'K-NN': KNeighborsClassifier(),
        'SVM': SVC(random_state=1337, probability=True)
    }

    tuned_models = {}
    print("Przeprowadzanie Grid Search dla kazdego modelu...")
    for name, base_model in base_models.items():
        print(f"  Dostrajanie {name}...")
        try:
            # Uzyj 5-krotnej walidacji krzyzowej
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grids[name],
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)

            tuned_models[name] = {
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }

            print(f"    Najlepsze parametry: {grid_search.best_params_}")
            print(f"    Najlepszy wynik CV: {grid_search.best_score_:.4f}")

        except Exception as e:
            print(f"    Blad podczas dostrajania {name}: {e}")
            # W przypadku bledu, uzyj modelu bazowego
            tuned_models[name] = {
                'best_model': base_model,
                'best_params': {},
                'best_score': 0.0
            }

    print("\n--- Podsumowanie dostrojonych modeli ---")
    for name, tuned_info in tuned_models.items():
        print(f"{name:20} CV Accuracy: {tuned_info['best_score']:.4f}")

    # 12. TREnowANIE DOSTROJONYCH MODELOW NA PELNYM ZBIORZE TRENINGOWYM
    print("\n--- Trenowanie dostrojonych modeli na pełnym zbiorze treningowym ---")
    models = {}
    for name, tuned_info in tuned_models.items():
        # Uzyj dostrojonego modelu z najlepszymi parametrami
        models[name] = tuned_info['best_model']
        # Jesli model nie byl dostrojony (blad), wytrenuj go od nowa
        if name not in tuned_info['best_params']:
            models[name] = base_models[name]

    # 13. TREnowANIE I EWALUACJA MODELI
    results = {}
    print("\n--- Wyniki modelowania ---")
    print("Trenowanie modeli...")
    for name, model in models.items():
        print(f"  Trenowanie {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Podstawowe metryki
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Macierz pomylek
        cm = confusion_matrix(y_test, y_pred)

        # AUC-ROC (jesli model ma predict_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'auc': auc_score
        }
        print(f"{name:20} Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}", end="")
        if auc_score is not None:
            print(f"  AUC: {auc_score:.4f}")
        else:
            print()

    best_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"Najlepszy model: {best_name}")

    # 12. Wizualizacja ewaluacji modeli
    print("\n--- Wizualizacja ewaluacji modeli ---")

    # 12a. Wykresy krzywych ROC
    print("Tworzenie wykresow krzywych ROC...")
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    for idx, (name, result) in enumerate(results.items()):
        if result['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=colors[idx % len(colors)],
                       label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

    # Losowy klasyfikator (linia referencyjna)
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Losowy klasyfikator (AUC = 0.5)', linewidth=1)

    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('Krzywe ROC dla modeli', fontsize=14)
    ax_roc.legend(loc='lower right', fontsize=10)
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('titanic_roc_curves.png', dpi=150, bbox_inches='tight')
    print("Zapisano wykres krzywych ROC: titanic_roc_curves.png")
    plt.close(fig_roc)

    # 12b. Macierze pomylek
    print("Tworzenie wizualizacji macierzy pomylek...")
    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
    axes_cm = axes_cm.flatten()

    for idx, (name, result) in enumerate(results.items()):
        if idx < 4:  # Mamy 4 modele, 4 osie
            cm = result['confusion_matrix']
            ax = axes_cm[idx]

            # Wizualizacja macierzy pomylek
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f'{name}', fontsize=12)

            # Etykiety
            classes = ['Nie przeżył', 'Przeżył']
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(classes, rotation=45)
            ax.set_yticklabels(classes)

            # Wartosci w komorkach
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

            ax.set_ylabel('Rzeczywista klasa', fontsize=10)
            ax.set_xlabel('Przewidziana klasa', fontsize=10)

    plt.tight_layout()
    plt.savefig('titanic_confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("Zapisano wizualizacje macierzy pomylek: titanic_confusion_matrices.png")
    plt.close(fig_cm)

    # 12c. Porownanie metryk modeli
    print("Tworzenie wykresu porownania metryk modeli...")
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Dokładność', 'Precyzja', 'Czułość', 'F1-score']
    model_names = list(results.keys())

    # Przygotuj dane
    metric_data = []
    for metric in metrics:
        metric_values = [results[name][metric] for name in model_names]
        metric_data.append(metric_values)

    fig_metrics, ax_metrics = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.2

    for i, (metric_vals, metric_label) in enumerate(zip(metric_data, metric_names)):
        ax_metrics.bar(x + i*width - width*1.5, metric_vals, width, label=metric_label)

    ax_metrics.set_xlabel('Model', fontsize=12)
    ax_metrics.set_ylabel('Wartość', fontsize=12)
    ax_metrics.set_title('Porównanie metryk ewaluacji modeli', fontsize=14)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(model_names, rotation=15)
    ax_metrics.legend(loc='upper right')
    ax_metrics.grid(True, alpha=0.3, axis='y')

    # Dodaj wartosci na slupkach
    for i, metric_vals in enumerate(metric_data):
        for j, val in enumerate(metric_vals):
            ax_metrics.text(x[j] + i*width - width*1.5, val + 0.01, f'{val:.3f}',
                           ha='center', va='bottom', fontsize=9, rotation=90)

    plt.tight_layout()
    plt.savefig('titanic_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("Zapisano wykres porownania metryk: titanic_metrics_comparison.png")
    plt.close(fig_metrics)

    # 13. Szczegolowe raporty klasyfikacji
    print("\n--- Szczegolowe raporty klasyfikacji ---")
    for name, result in results.items():
        print(f"\n{name}:")
        print("Macierz pomylek:")
        print(result['confusion_matrix'])
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, result['y_pred'], target_names=['Nie przeżył', 'Przeżył']))
        if result['auc'] is not None:
            print(f"AUC-ROC: {result['auc']:.4f}")

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