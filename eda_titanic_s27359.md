# Eksploracyjna analiza danych (EDA) - Titanic

## 1. Przegląd zbioru danych
Zbiór danych Titanica zawiera informacje o 1309 pasażerach statku RMS Titanic, który zatonął 15 kwietnia 1912 roku po kolizji z górą lodową.

**Podstawowe informacje:**
- Liczba pasażerów: 1309
- Liczba cech: 14
- Liczba przeżyłych: 500 (38.2%)
- Liczba zgonów: 809 (61.8%)

**Struktura danych:**
```
pclass     - klasa kabiny (1 = pierwsza, 2 = druga, 3 = trzecia)
survived   - przeżycie (0 = nie, 1 = tak)
name       - imię i nazwisko
sex        - płeć
age        - wiek
sibsp      - liczba małżonków/rodzeństwa na pokładzie
parch      - liczba rodziców/dzieci na pokładzie
ticket     - numer biletu
fare       - opłata za bilet
cabin      - numer kabiny
embarked   - port zaokrętowania (C, Q, S)
boat       - numer łodzi ratunkowej
body       - numer znalezionego ciała
home.dest  - miejsce pochodzenia/przeznaczenia
```

## 2. Analiza brakujących wartości
Brakujące wartości stanowią istotne wyzwanie w tym zbiorze danych:

| Kolumna | Brakujące wartości | Procent |
|---------|-------------------|---------|
| age | 263 | 20.1% |
| fare | 1 | 0.08% |
| cabin | 1014 | 77.5% |
| embarked | 2 | 0.15% |
| boat | 823 | 62.9% |
| body | 1188 | 90.8% |
| home.dest | 564 | 43.1% |

**Wnioski:**
- Wiek ma znaczną liczbę braków (20.1%), co wymaga imputacji
- Dane kabin są bardzo niekompletne (77.5% braków)
- Zmienne `body` i `boat` są związane z faktem przeżycia i nie powinny być używane jako predyktory (wyciek danych)
- `home.dest` ma 43.1% braków, co ogranicza jej użyteczność

## 3. Statystyki opisowe zmiennych numerycznych

### Wiek (age)
```
count    1046.000000
mean       29.881138
std        14.413500
min         0.166700
25%        21.000000
50%        28.000000
75%        39.000000
max        80.000000
```

**Obserwacje:**
- Średni wiek: 29.9 lat
- Mediana: 28 lat (rozkład prawostronnie skośny)
- Najmłodszy pasażer: 0.17 roku (~2 miesiące)
- Najstarszy pasażer: 80 lat
- 25% pasażerów ma ≤21 lat, 75% ma ≤39 lat

### Opłata za bilet (fare)
```
count    1308.000000
mean       33.295479
std        51.758668
min         0.000000
25%         7.895800
50%        14.454200
75%        31.275000
max       512.329200
```

**Obserwacje:**
- Średnia opłata: 33.30
- Duże odchylenie standardowe (51.76) wskazuje na duże zróżnicowanie
- Rozkład silnie prawostronnie skośny (mediana 14.45 vs średnia 33.30)
- Istnieją bilety darmowe (fare = 0)
- Maksymalna opłata: 512.33 (ekstremalna wartość)

### Liczba rodzeństwa/małżonków (sibsp)
```
count    1309.000000
mean        0.498854
std         1.041658
min         0.000000
25%         0.000000
50%         0.000000
75%         1.000000
max         8.000000
```

### Liczba rodziców/dzieci (parch)
```
count    1309.000000
mean        0.385027
std         0.865560
min         0.000000
25%         0.000000
50%         0.000000
75%         0.000000
max         9.000000
```

**Obserwacje:**
- Większość pasażerów podróżuje samotnie (sibsp=0, parch=0)
- Maksymalna liczba rodzeństwa/małżonków: 8
- Maksymalna liczba rodziców/dzieci: 9

## 4. Analiza zmiennych kategorycznych

### Klasa (pclass)
```
1    323 (24.7%)
2    277 (21.2%)
3    709 (54.1%)
```

**Współczynniki przeżycia:**
- Klasa 1: 61.9%
- Klasa 2: 43.0%
- Klasa 3: 25.5%

**Wnioski:** Silna korelacja między klasą a szansami przeżycia. Pasażerowie pierwszej klasy mieli 2.4 razy większe szanse przeżycia niż pasażerowie trzeciej klasy.

### Płeć (sex)
```
male      843 (64.4%)
female    466 (35.6%)
```

**Współczynniki przeżycia:**
- Kobiety: 72.7%
- Mężczyźni: 19.1%

**Wnioski:** Kobiety miały 3.8 razy większe szanse przeżycia niż mężczyźni, co potwierdza zasadę "kobiety i dzieci pierwsze".

### Port zaokrętowania (embarked)
```
S    914 (69.8%)
C    270 (20.6%)
Q    123 (9.4%)
NaN    2 (0.2%)
```

**Współczynniki przeżycia:**
- Cherbourg (C): 55.4%
- Queenstown (Q): 38.8%
- Southampton (S): 33.7%

**Wnioski:** Pasażerowie z Cherbourg mieli najwyższe szanse przeżycia, co może korelować z wyższą klasą społeczno-ekonomiczną.

## 5. Wizualizacje

![](titanic_eda_plots.png)

*Rysunek 1: Trzy panele analizy EDA Titanica*

### 5.4 Analiza brakujących wartości

![](titanic_missing_values.png)

*Rysunek 2: Rozkład brakujących wartości w danych surowych*

**Obserwacje:**
- Wiek ma 20.1% brakujących wartości (263 rekordy)
- Kabina (cabin) ma 77.5% braków, co uzasadnia utworzenie zmiennej binarnej `has_cabin`
- Pozostałe zmienne mają niewielką liczbę braków (<1%)

### 5.5 Macierz korelacji cech numerycznych

![](titanic_correlation_matrix.png)

*Rysunek 3: Macierz korelacji między cechami numerycznymi*

**Kluczowe korelacje:**
- Silna dodatnia korelacja między klasą (pclass) a opłatą (fare): -0.55 (wyższa klasa = wyższa opłata)
- Ujemna korelacja między klasą a wiekiem: -0.37 (wyższa klasa = starsi pasażerowie)
- Słaba korelacja między przeżyciem a płcią (zakodowaną numerycznie): 0.54
- Brak silnych korelacji między pozostałymi zmiennymi

### 5.6 Rozkład cech numerycznych przed czyszczeniem

![](titanic_numeric_distributions_before.png)

*Rysunek 4: Rozkład cech numerycznych przed czyszczeniem danych*

**Obserwacje:**
- **Opłata (fare):** Rozkład silnie prawostronnie skośny z ekstremalnymi wartościami (>500)
- **Rodzeństwo/małżonkowie (sibsp):** 68% pasażerów podróżuje bez rodzeństwa/małżonka
- **Rodzice/dzieci (parch):** 76% pasażerów podróżuje bez rodziców/dzieci

### 5.7 Porównanie rozkładu opłat przed i po przycięciu wartości odstających

![](titanic_fare_distribution_comparison.png)

*Rysunek 5: Porównanie rozkładu opłat przed (lewy panel) i po (prawy panel) przycięciu wartości odstających metodą IQR*

**Wpływ przycięcia:**
- **Przed przycięciem:** Rozkład wykazuje ekstremalne wartości (>500), które zniekształcają statystyki
- **Po przycięciu:** Wartości powyżej górnego kwartylu + 1.5×IQR zostały przycięte do wartości granicznej
- **Efekt:** Redukcja skośności rozkładu, co poprawia stabilność modeli

### 5.8 Rozkład wieku (lewy panel)
- Rozkład prawostronnie skośny z większą liczbą młodych osób
- Większość pasażerów w wieku 20-40 lat
- Widoczna grupa dzieci (<10 lat)
- Brak wyraźnych grup wiekowych powyżej 60 lat

### 5.9 Przeżycie według klasy (środkowy panel)
- Wyraźny gradient: klasa 1 > klasa 2 > klasa 3
- Różnica między klasą 1 a 3 wynosi 36.4 punktów procentowych
- Wskazuje na silny wpływ statusu społeczno-ekonomicznego na szanse przeżycia

### 5.10 Przeżycie według płci (prawy panel)
- Dramatyczna różnica: kobiety 72.7% vs mężczyźni 19.1%
- Potwierdzenie historycznie udokumentowanej zasady priorytetowej ewakuacji kobiet i dzieci

### 5.11 Porównanie rozkładu wieku przed i po czyszczeniu
Aby zilustrować wpływ procesu czyszczenia danych, porównano rozkład wieku przed i po przetworzeniu:

![](titanic_age_comparison.png)

*Rysunek 6: Porównanie rozkładu wieku przed czyszczeniem (lewy panel) i po czyszczeniu (prawy panel)*

**Kluczowe obserwacje:**
1. **Przed czyszczeniem:** Wykazano 263 brakujących wartości wieku (20.1% danych), reprezentowanych jako '?' w surowych danych. Rozkład oparty jest na 1046 dostępnych rekordach.
2. **Po czyszczeniu:** Wszystkie brakujące wartości zostały imputowane za pomocą średnich według tytułu, a następnie średnich według płci. Dodatkowo przycięto wartości odstające powyżej 67 lat.
3. **Zmiany statystyczne:**
   - Średnia wieku: 29.9 → 29.6 lat (nieznaczny spadek)
   - Mediana wieku: 28.0 → 28.0 lat (bez zmian)
   - Rozkład zachowuje prawostronną skośność, ale staje się bardziej kompletny
4. **Wpływ imputacji:** Uzupełnienie brakujących wartości pozwoliło na zachowanie wszystkich 1309 rekordów bez utraty danych, co jest kluczowe dla jakości modelu.


### 5.12 Boxplot wieku według klasy i przeżycia
Boxplot przedstawia rozkład wieku pasażerów według klasy oraz według przeżycia, co pozwala na wizualizację mediany, kwartyli oraz wartości odstających. Wykres składa się z dwóch panelów: lewy panel pokazuje rozkład wieku według klasy (1, 2, 3), a prawy panel pokazuje rozkład wieku według przeżycia (nie przeżył, przeżył).

![](titanic_age_boxplots.png)

*Rysunek 7: Boxplot wieku pasażerów - lewy panel: rozkład wieku według klasy (1, 2, 3), prawy panel: rozkład wieku według przeżycia (nie przeżył, przeżył)*

**Kluczowe obserwacje z boxplotu wieku:**
- **Rozkład według klasy:** Pasażerowie klasy 1 mają najwyższą medianę wieku (~39 lat), podczas gdy pasażerowie klasy 3 mają najniższą medianę wieku (~24 lat). Potwierdza to obserwację, że starsi pasażerowie mają wyższy status ekonomiczny.
- **Rozkład według przeżycia:** Mediana wieku osób, które przeżyły, jest niższa (~28 lat) niż mediana wieku osób, które nie przeżyły (~30 lat). Wskazuje to na wyższe szanse przeżycia młodszych pasażerów.
- **Wartości odstające:** W obu panelach widoczne są wartości odstające powyżej 60-70 lat, co zostało uwzględnione w procesie czyszczenia danych.
- **Rozrzut danych:** Rozkład wieku w klasie 3 ma największy rozrzut (najszersze pudełko), co sugeruje większą różnorodność wieku wśród pasażerów klasy trzeciej.

## 6. Analiza korelacji i zależności

### Zależność wieku od klasy
- **Klasa 1:** średni wiek = 38.2, mediana = 39
- **Klasa 2:** średni wiek = 29.9, mediana = 29
- **Klasa 3:** średni wiek = 25.1, mediana = 24

**Wnioski:** Pasażerowie wyższych klas są średnio starsi, co odzwierciedla ich wyższy status ekonomiczny.

### Zależność opłaty od klasy
- **Klasa 1:** średnia opłata = 87.5, mediana = 60.3
- **Klasa 2:** średnia opłata = 21.2, mediana = 15.0
- **Klasa 3:** średnia opłata = 13.7, mediana = 8.1

**Wnioski:** Wyraźna korelacja między klasą a opłatą, ale duże zróżnicowanie w obrębie klas (szczególnie klasy 1).

### Rozkład przeżycia w zależności od wieku i płci
- Kobiety przeżywają we wszystkich grupach wiekowych
- Mężczyźni mają niskie szanse przeżycia we wszystkich grupach wiekowych
- Dzieci (<12 lat) mają wyższe szanse przeżycia niezależnie od płci

## 7. Wnioski i rekomendacje dla przetwarzania danych

1. **Imputacja wieku:** Uzupełnić braki średnią według tytułu, a następnie średnią według płci
2. **Zmienna cabin:** Ze względu na 77.5% braków, utworzyć zmienną binarną `has_cabin`
3. **Usunięcie zmiennych:** `boat` i `body` usunąć (wyciek danych), `home.dest` usunąć (43.1% braków)
4. **Tworzenie nowych cech:**
   - `family_size` = sibsp + parch + 1
   - `age_range` kategoryzacja: Bobas (0-6), Dzieciak (6-12), Nastolatek (12-18), Dorosły (18+)
   - `mpc` = age × pclass (uwypukla szanse dzieci w wyższych klasach)
5. **Przycięcie wartości odstających:** Wiek >67, opłata (metoda IQR)
6. **Normalizacja:** Wszystkie zmienne numeryczne znormalizować metodą MinMax

## 8. Dodatkowe analizy EDA po czyszczeniu danych
Po procesie czyszczenia danych i inżynierii cech, przeprowadzono dodatkowe analizy eksploracyjne, które dostarczają głębszego wglądu w czynniki wpływające na przeżycie pasażerów Titanica.

![](titanic_additional_eda.png)

*Rysunek 8: Dodatkowe analizy EDA po czyszczeniu danych - czteropanelowy wykres przedstawiający: (a) Współczynnik przeżycia według grup wiekowych, (b) Współczynnik przeżycia według rozmiaru rodziny, (c) Współczynnik przeżycia według portu zaokrętowania, (d) Współczynnik przeżycia według tytułu*

### 8.1 Współczynnik przeżycia według grup wiekowych
Analiza potwierdza zasadę priorytetowej ewakuacji dzieci:
- **Bobasy (0-6 lat):** 65.2% przeżycia - najwyższy wskaźnik wśród wszystkich grup wiekowych
- **Dzieciaki (6-12 lat):** 46.7% przeżycia
- **Nastolatki (12-18 lat):** 40.0% przeżycia
- **Dorośli (18+ lat):** 36.8% przeżycia

Gradient przeżycia wyraźnie wskazuje, że młodsze osoby miały wyższe szanse przeżycia, co potwierdza historycznie udokumentowaną zasadę "kobiety i dzieci pierwsze".

### 8.2 Współczynnik przeżycia według rozmiaru rodziny
Analiza rozmiaru rodziny ujawnia interesujące zależności:
- **Osoby samotne (rozmiar 1):** 30.4% przeżycia - najniższy wskaźnik
- **Rodziny 2-osobowe:** 55.0% przeżycia
- **Rodziny 3-osobowe:** 60.0% przeżycia
- **Rodziny 4-osobowe:** 72.7% przeżycia
- **Rodziny 5+ osobowe:** 25.0% przeżycia

**Wnioski:** Małe rodziny (2-4 osoby) miały najwyższe szanse przeżycia, podczas gdy osoby podróżujące samotnie oraz bardzo duże rodziny miały niższe szanse. Może to wynikać z faktu, że małe rodziny mogły skuteczniej współpracować podczas ewakuacji, podczas gdy duże rodziny miały trudności z koordynacją.

### 8.3 Współczynnik przeżycia według portu zaokrętowania
Port zaokrętowania wykazuje wyraźne różnice w szansach przeżycia:
- **Cherbourg (C):** 55.4% przeżycia
- **Queenstown (Q):** 38.8% przeżycia
- **Southampton (S):** 33.7% przeżycia

**Interpretacja:** Pasażerowie z Cherbourg (Francja) mieli najwyższe szanse przeżycia, co może korelować z wyższym statusem społeczno-ekonomicznym (więcej pasażerów pierwszej klasy) lub narodowością.

### 8.4 Współczynnik przeżycia według tytułu
Analiza tytułów wyekstrahowanych z imion i nazwisk ujawnia dramatyczne różnice:
- **Miss (panna):** 69.7% przeżycia
- **Mrs (mężatka):** 79.4% przeżycia
- **Master (młody mężczyzna):** 57.6% przeżycia
- **Mr (mężczyzna):** 15.7% przeżycia
- **Military (wojskowy):** 41.7% przeżycia
- **Noble (szlachta):** 50.0% przeżycia

**Kluczowe obserwacje:** Kobiety (Miss, Mrs) miały bardzo wysokie szanse przeżycia, podczas że mężczyźni (Mr) mieli bardzo niskie. Młodzi mężczyźni (Master) mieli znacząco wyższe szanse niż dorośli mężczyźni (Mr), co ponownie potwierdza priorytet dzieci. Osoby z tytułami wojskowymi i szlacheckimi miały umiarkowane szanse przeżycia.

## 9. Podsumowanie EDA
Zbiór danych Titanica zawiera cenne informacje demograficzne i podróżnicze, które silnie korelują z szansami przeżycia. Kluczowe czynniki to płeć, klasa i wiek. Znaczna liczba brakujących wartości wymaga ostrożnego przetwarzania, a niektóre zmienne (`boat`, `body`) muszą zostać usunięte ze względu na wyciek danych. Analiza potwierdza historyczne doniesienia o priorytecie ewakuacji kobiet, dzieci i pasażerów wyższych klas.