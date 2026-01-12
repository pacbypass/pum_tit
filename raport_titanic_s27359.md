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

*Rysunek 1: Trzy panele analizy EDA: (a) Rozkład wieku pasażerów, (b) Współczynnik przeżycia według klasy, (c) Współczynnik przeżycia według płci*

- Dodatkowo, w załączonej analizie EDA przedstawiono porównanie rozkładu wieku przed i po czyszczeniu danych, co ilustruje wpływ procesu imputacji brakujących wartości.

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
1. Eksploracyjna analiza danych (EDA) - analiza rozkładów, brakujących wartości, korelacji
2. Czyszczenie danych - konwersja '?' na NaN, typy danych
3. Inżynieria cech - ekstrakcja tytułów, tworzenie nowych zmiennych
4. Imputacja brakujących wartości - wiek per tytuł i płeć, opłata per klasa
5. Przycięcie wartości odstających - wiek >67, opłata (metoda IQR)
6. Konwersja typów kategorycznych
7. Podział danych - 80% treningowe, 20% testowe (z zachowaniem proporcji)
8. Kodowanie kategoryczne i normalizacja (tylko na danych treningowych)
9. Trenowanie modeli
10. Ewaluacja i analiza wyników

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