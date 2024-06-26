---
mermaid: true
mathjax: true
---

# Typy danych w Uczeniu Maszynowym

<div class="mermaid">
mindmap
    id(("Typy danych w uczeniu maszynowym"))
        id("Ilościonwe")
            id("Dane dyskretne")
                id("Liczba goli w meczu: 0 goli, 2 gole, 3 gole")
                id("Liczba sprzedanych biletów: 100 biletów, 250 biletów, 500 biletów")
            id("Dane ciągłe")
                id("Wzrost: 180.5 cm, 184.8 cm, 190.2 cm")
                id("Waga: 60.7 kg, 50.3 kg, 55.0 kg")
                id("Czas: 12.34 sekundy, 1.23 godziny, 56 minut")
        id("Jakościowe")
            id("Dane nominalne")
                id("Płeć: Kobieta, Mężczyzna")
                id("Kolor karoseri: Czerwony, Czzarny, ..")
                id("Typ krwi: A, 0, B, AB")
            id("Dane porzadkowe")
                id("Oceny satysfakcji: Bardzo niezadowolony, Niezadowolony, Neutralny, Zadowolony, Bardzo zadowolony")
                id("Poziom wykształcenia: Podstawowe, Średnie, Wyższe")
                id("Oceny w szkole: Dopuszczający, Dostateczny, Dobry, Bardzo dobry, Celujący")
</div>

<div class="mermaid">
classDiagram
    classA <|-- Quantative
    classA <|-- Qualitative
    Quantative <|-- Discrete
    Quantative <|-- Continious
    Qualitative <|-- Nominal
    Qualitative <|-- Ordinal
    
</div>