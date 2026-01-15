# Titanic

Dominik Chyliński (s27359)

## Opis problemu
Problem polega na przewidywaniu przeżycia pasażerów Titanica na podstawie danych demograficznych i informacji o podróży. Katastrofa Titanica z 1912 roku jest dobrze udokumentowanym wydarzeniem historycznym, a dostępne dane pozwalają na analizę czynników wpływających na szanse przeżycia.

Model może być wykorzystany przez:
- Historyków i badaczy do analizy wzorców przeżycia w katastrofach morskich
- Projektantów systemów bezpieczeństwa na statkach do identyfikacji grup wymagających specjalnej ochrony
- Edukatorów do ilustracji zasad ewakuacji i priorytetów ratunkowych ("kobiety i dzieci pierwsze")

Problem jest interesujący ze względu na:
- Historyczny kontekst i emocjonalne znaczenie wydarzenia
- Możliwość analizy decyzji ludzkich w sytuacjach kryzysowych
- Wyzwania związane z niekompletnymi danymi historycznymi
- Możliwość zastosowania zaawansowanych technik uczenia maszynowego do danych z początku XX wieku

## Dane
**Źródło danych:** Oficjalny zbiór danych Titanica z OpenML (https://www.openml.org/data/get_csv/16826755)

**Ocena wiarygodności:** Dane pochodzą z dokumentacji katastrofy i są uważane za wiarygodne, choć zawierają niekompletne rekordy typowe dla danych historycznych.

**Krótka analiza opisowa danych:**
- Zbiór zawiera 1309 pasażerów z 14 cechami
- 38.2% pasażerów przeżyło (500 osób), 61.8% zginęło (809 osób)
- Wiek: 263 brakujących wartości (20.1% danych)
- Klasa: wyraźny gradient przeżycia (1 klasa: 61.9%, 2 klasa: 43.0%, 3 klasa: 25.5%)
- Płeć: kobiety przeżywały znacznie częściej (72.7%) niż mężczyźni (19.1%)
- Opłata za bilet (fare): 1 brakująca wartość
- Kabina (cabin): 1014 brakujących wartości (77.5% danych)
- Wizualizacje rozkładów potwierdzają silne zależności między klasą, płcią a szansami przeżycia:

![](titanic_eda_plots.png)

*Rysunek 1: Trzy panele analizy danych: (a) Rozkład wieku pasażerów, (b) Współczynnik przeżycia według klasy, (c) Współczynnik przeżycia według płci*

**Analiza brakujących wartości:** Rozkład brakujących wartości w danych surowych pokazuje, które cechy wymagają szczególnej uwagi podczas przetwarzania.

![](titanic_missing_values.png)

*Rysunek 2: Rozkład brakujących wartości w danych surowych - wiek (20.1%), kabina (77.5%), oraz pojedyncze braki w innych cechach*

**Macierz korelacji:** Analiza korelacji między cechami numerycznymi ujawnia silne zależności między klasą, opłatą i wiekiem.

![](titanic_correlation_matrix.png)

*Rysunek 3: Macierz korelacji cech numerycznych - silna ujemna korelacja między klasą a opłatą (-0.55), umiarkowana korelacja między klasą a wiekiem (-0.37)*


**Rozkład cech numerycznych:** Histogramy przedstawiają rozkład opłaty, liczby rodzeństwa/małżonków oraz liczby rodziców/dzieci przed czyszczeniem danych.

![](titanic_numeric_distributions_before.png)

*Rysunek 4: Rozkład cech numerycznych przed czyszczeniem - opłata (silnie prawostronnie skośna), większość pasażerów podróżuje samotnie*

**Porównanie rozkładu wieku:** Wizualizacja ilustruje wpływ procesu czyszczenia danych na rozkład wieku pasażerów.

![](titanic_age_comparison.png)

*Rysunek 5: Porównanie rozkładu wieku przed (lewy panel) i po (prawy panel) czyszczeniu danych - imputacja brakujących wartości i przycięcie wartości odstających*

**Dodatkowe analizy:** Po procesie czyszczenia danych przeprowadzono dodatkowe analizy czynników wpływających na przeżycie, w tym analizę grup wiekowych, rozmiaru rodziny, portu zaokrętowania oraz tytułów pasażerów.

![](titanic_additional_eda.png)

*Rysunek 6: Dodatkowe analizy po czyszczeniu danych - czteropanelowy wykres przedstawiający: (a) Współczynnik przeżycia według grup wiekowych, (b) Współczynnik przeżycia według rozmiaru rodziny, (c) Współczynnik przeżycia według portu zaokrętowania, (d) Współczynnik przeżycia według tytułu*

**Analiza rozkładu wieku:** Boxplot przedstawia rozkład wieku pasażerów według klasy oraz według przeżycia, co pozwala na wizualizację mediany, kwartyli oraz wartości odstających.

![](titanic_age_boxplots.png)

*Rysunek 7: Boxplot wieku pasażerów - lewy panel: rozkład wieku według klasy (1, 2, 3), prawy panel: rozkład wieku według przeżycia (nie przeżył, przeżył)*

**Uzasadnienie:** Dane zawierają kluczowe informacje demograficzne i podróżnicze, które mogą korelować z szansami przeżycia:
- Klasa kabiny odzwierciedla status społeczno-ekonomiczny i lokalizację na statku
- Wiek i płeć są związane z zasadą "kobiety i dzieci pierwsze"
- Rozmiar rodziny może wpływać na decyzje ewakuacyjne
- Port zaokrętowania może korelować z narodowością i statusem

## Sposób rozwiązania problemu
**Wybrany model:** Zastosowano zestaw 4 modeli klasyfikacyjnych w celu porównania ich skuteczności:
1. Drzewo decyzyjne
2. Regresja logistyczna 
3. K-najbliższych sąsiadów (K-NN)
4. SVM

**Uzasadnienie wyboru:** Problem jest klasyfikacją binarną (przeżył/nie przeżył). Zestaw modeli obejmuje różne podejścia do klasyfikacji, co pozwala na znalezienie najlepszego rozwiązania.

**Etapy realizacji projektu:**
1. Eksploracyjna analiza danych - analiza rozkładów, brakujących wartości, korelacji
2. Czyszczenie danych - konwersja '?' na NaN, typy danych
3. Inżynieria cech - ekstrakcja tytułów, tworzenie nowych zmiennych
4. Imputacja brakujących wartości - wiek per tytuł i płeć, opłata per klasa
5. Przycięcie wartości odstających - wiek >67, opłata (metoda IQR)
6. Konwersja typów kategorycznych
7. Podział danych - 80% treningowe, 20% testowe (z zachowaniem proporcji)
8. Kodowanie kategoryczne i normalizacja (tylko na danych treningowych)
9. Trenowanie modeli
10. Ewaluacja i analiza wyników

**Ilustracja przygotowania danych:** Poniższy wykres przedstawia porównanie rozkładu opłat przed i po przycięciu wartości odstających metodą IQR, co stanowi kluczowy etap przetwarzania danych.

![](titanic_fare_distribution_comparison.png)

*Rysunek 8: Porównanie rozkładu opłat przed (lewy panel) i po (prawy panel) przycięciu wartości odstających metodą IQR - redukcja skośności rozkładu poprawia stabilność modeli*

**Dostrajanie hiperparametrów (Grid Search):**
W celu optymalizacji wydajności modeli zastosowano technikę Grid Search z 3-krotną walidacją krzyżową. Dla każdego modelu zdefiniowano siatkę parametrów, a następnie wybrano najlepszą kombinację na podstawie dokładności (accuracy) na zbiorze walidacyjnym.

**Wyniki Grid Search - najlepsze parametry i wyniki walidacji krzyżowej:**

| Model | Najlepsze parametry | CV Accuracy (3-fold) |
|-------|---------------------|----------------------|
| Drzewo Decyzyjne | `max_depth`: 3, `min_samples_split`: 2 | 79.08% |
| Regresja Logistyczna | `C`: 10, `solver`: 'lbfgs' | 79.85% |
| K-NN | `n_neighbors`: 7, `weights`: 'uniform' | 77.65% |
| SVM | `C`: 10, `kernel`: 'linear' | 78.80% |

*Tabela 1: Wyniki dostrajania hiperparametrów metodą Grid Search z 3-krotną walidacją krzyżową*

**Interpretacja wyników:** Regresja Logistyczna osiągnęła najwyższy wynik walidacji krzyżowej (79.85%), co sugeruje najlepszą generalizację na nowych danych. Wszystkie modele osiągnęły wyniki powyżej 77%, co potwierdza odpowiedni dobór siatek parametrów.

Po wyborze najlepszych parametrów, modele zostały wytrenowane na pełnym zbiorze treningowym (1047 próbek) i przetestowane na zbiorze testowym (262 próbek).

**Kompleksowa ewaluacja modeli:**
Oprócz podstawowych metryk (accuracy, precision, recall, F1-score), przeprowadzono kompleksową analizę modeli obejmującą:
- Krzywe ROC i wartość AUC-ROC dla modeli z funkcją predict_proba
- Macierze pomyłek dla wszystkich modeli
- Porównanie metryk na wykresie słupkowym

Wizualizacje ewaluacji dostępne są w załącznikach.

**Miary ewaluacji:**
- Dokładność (Accuracy) - główna metryka, cel ≥80%
- Precyzja (Precision) - stosunek prawdziwie pozytywnych do wszystkich pozytywnych
- Czułość (Recall) - zdolność do wykrywania pozytywnych przypadków
- F1-score - średnia harmoniczna precyzji i czułości

## Dyskusja wyników i ewaluacja modelu
**Wyniki modelowania:**

| Model                | Accuracy | Precision | Recall | F1-score | Status |
| -------------------  | -------- | --------- | ------ | -------- | ------ |
| Regresja logistyczna | 0.8168   | 0.7708    | 0.7400 | 0.7551   | PASS |
| K-NN                 | 0.8130   | 0.7576    | 0.7500 | 0.7538   | PASS |
| SVM                  | 0.8130   | 0.7576    | 0.7500 | 0.7538   | PASS |
| Drzewo decyzyjne     | 0.7519   | 0.6667    | 0.7000 | 0.6829   | FAIL |

**Analiza AUC-ROC:** Dla modeli z funkcją predict_proba obliczono obszar pod krzywą ROC (AUC-ROC). Regresja logistyczna osiągnęła AUC = 0.857, K-NN = 0.853, SVM = 0.853, co potwierdza dobrą zdolność dyskryminacyjną modeli. Krzywe ROC przedstawiono na Rysunku 9.

**Wizualizacje ewaluacji modeli:** Poniżej przedstawiono kompleksową wizualizację wyników ewaluacji wszystkich modeli.

**Krzywe ROC (Rysunek 9):** Wykres przedstawia krzywe ROC dla wszystkich modeli wraz z wartościami AUC-ROC. Linia przerywana reprezentuje klasyfikator losowy (AUC = 0.5). Im wyższa krzywa nad linią referencyjną, tym lepsza zdolność dyskryminacyjna modelu.

![](titanic_roc_curves.png)

*Rysunek 9: Krzywe ROC dla modeli - Regresja Logistyczna (AUC = 0.857), K-NN (AUC = 0.853), SVM (AUC = 0.853)*

**Macierze pomyłek (Rysunek 10):** Wizualizacja macierzy pomyłek dla każdego modelu pozwala na analizę rodzajów błędów klasyfikacji (fałszywie pozytywne i fałszywie negatywne).

![](titanic_confusion_matrices.png)

*Rysunek 10: Macierze pomyłek dla modeli - wartości w komórkach przedstawiają liczbę sklasyfikowanych przypadków*

**Porównanie metryk (Rysunek 11):** Wykres słupkowy umożliwia bezpośrednie porównanie czterech kluczowych metryk ewaluacji dla wszystkich modeli.

![](titanic_metrics_comparison.png)

*Rysunek 11: Porównanie metryk accuracy, precision, recall, F1-score - Regresja Logistyczna osiąga najlepsze wyniki we wszystkich metrykach*

**Najważniejsze wnioski:**
- **Regresja Logistyczna** osiągnęła najlepszy wynik: **81.68% accuracy**
- Wszystkie modele oprócz drzewa decyzyjnego przekroczyły próg 80%
- **Cel projektu (≥80% accuracy) został osiągnięty**
- Modele regresji logistycznej i metody oparte na odległości (KNN, SVM) wykazały się najwyższą skutecznością


**Ewaluacja modelu:** Regresja logistyczna została wybrana jako najlepszy model ze względu na najwyższą dokładność. Model został przetestowany na zbiorze testowym (262 pasażerów), co zapewnia wiarygodność wyników. Uniknięto przeuczenia dzięki prawidłowemu podziałowi danych i uniknięciu wycieku danych.

## Podsumowanie
**Co się udało?**
- Osiągnięto cel projektu z accuracy 81.68%
- Zidentyfikowano kluczowe czynniki wpływające na przeżycie (płeć, wiek, klasa, opłata)
- Przeprowadzono kompleksowe przetwarzanie danych zgodnie z wymaganiami
- Porównano skuteczność 4 różnych algorytmów uczenia maszynowego
- Uniknięto typowych błędów metodologicznych (wyciek danych)
- Przeprowadzono dostrajanie hiperparametrów za pomocą Grid Search oraz kompleksową ewaluację z wykorzystaniem krzywych ROC, macierzy pomyłek i porównaniem metryk

**Jakie były problemy? Jak je rozwiązaliśmy?**
1. **Brakujące wartości wieku** (263 rekordy) - rozwiązano przez imputację średnią według tytułu, a następnie średnią według płci
2. **Wyciek danych przy normalizacji** - rozwiązano przez przeniesienie normalizacji i kodowania kategorycznego po podziale na dane treningowe i testowe
3. **Wartości odstające** - przycięto wiek (>67) i opłatę (metodą IQR)
4. **Niekompletne dane kabin** - stworzono zmienną binarną has_cabin

**W jaki sposób może być to wykorzystane/rozwinięte w przyszłości?**
- **Zastosowania praktyczne:** Systemy wczesnego ostrzegania na statkach, projektowanie procedur ewakuacyjnych
- **Rozwój modelu:** Dodanie większej liczby cech kontekstowych (np. narodowość, zawód)
- **Ulepszenia techniczne:** Strojenie hiperparametrów, zastosowanie głębokiego uczenia
- **Analiza porównawcza:** Porównanie z innymi katastrofami morskimi
- **Edukacja:** Materiał dydaktyczny do nauczania analizy danych i uczenia maszynowego

## Załączniki
### Załącznik 1. Eksploracyjna analiza danych - eda_titanic_s27359.pdf

### Załącznik 2. Wizualizacje ewaluacji modeli
Pliki PNG zawierające wizualizacje ewaluacji modeli:
1. `titanic_roc_curves.png` - Krzywe ROC dla wszystkich modeli (Rysunek 9)
2. `titanic_confusion_matrices.png` - Macierze pomyłek dla każdego modelu (Rysunek 10)
3. `titanic_metrics_comparison.png` - Porównanie metryk accuracy, precision, recall, F1-score (Rysunek 11)