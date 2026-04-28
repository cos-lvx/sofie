# NS Corpus — Manifest reasoning chains

> Vygenerováno automaticky z `dataset/reasoning_chains/law/`. Needitovat ručně.
> Aktualizovat skriptem `dataset/reasoning_chains/law/scripts/manifest.py`.

**Stav:** 43 distillates, 6-sekční šablona u 43/43, párování se zdroji 43/43.

## Souhrn

- Distillates: **43**
- Slov v distillates: **38 394** (průměr 892 / chain)
- Slov v plných zněních: **210 786** (komprese ~5.5×)
- Délka chain: min 770 — max 1047 slov
- Distillate bez citace §§: **1** (`NS-25Cdo2422-2019`)
- Tokenů (Falcon-H1 tokenizer): **118 642** (min 2324 — max 3091  průměr 2759)

## Distribuce podle patternů

| Pattern | Téma | Počet | Distillates |
|---|---|---|---|
| P1 | lex specialis / kvalifikace | 4 | 20Cdo1897-98, 31Cdo4356-2008, 31Cdo4781-2009, 33Cdo1507-2022 |
| P2 | teleologický výklad | 5 | 21Cdo4521-2011, 23Cdo1001-2021, 23Cdo2885-2022, 25Cdo3925-2013, 28Cdo1214-2023 |
| P3 | eurokonformní výklad | 4 | 21Cdo3976-2013, 25Cdo2202-2018, 30Cdo1982-2012, Cpjn206-2010 |
| P4 | analogie / mezery | 3 | 22Cdo3277-2014, 22Cdo3732-2014, 23Cdo672-2021 |
| P5 | rebus sic stantibus | 3 | 23Cdo2486-2020, 26Cdo1670-2018, 33Odo343-2005 |
| P6 | ratio vs obiter / sjednocování | 3 | 31Cdo1468-2023, 31Cdo2805-2011, Plsn1-2015 |
| P7 | překonání vlastní judikatury | 3 | 31Cdo1622-2021, 31Cdo3263-2024, 31Cdo353-2016 |
| P8 | balancing principů | 6 | 21Cdo1276-2001, 22Cdo1172-2022, 22Cdo3192-2015, 25Cdo2422-2019, 31Cdo2273-2022, 33Cdo42-2021 |
| P9 | důkazní břemeno | 3 | 21Cdo246-2008, 22Cdo1843-2000, 26Cdo732-98 |
| P10 | procesně-hmotná hranice | 3 | 31Cdo1038-2009, 31Cdo365-2009, 31Cdo4616-2010 |
| P11 | kauzalita / škoda | 4 | 25Cdo1417-2006, 29Cdo3321-2020, 31Cdo1778-2014, 31Cdo2376-2021 |
| P12 | výklad vůle | 2 | 23Cdo2856-2022, 29Cdo61-2017 |

## Plný seznam

| Distillate | Pattern | Slov | Tokenů | Komprese | §§ |
|---|---|---:|---:|---:|---:|
| `NS-20Cdo1897-98` | P1 | 938 | 2889 | 2.4× | 2 |
| `NS-31Cdo4356-2008` | P1 | 867 | 2531 | 6.3× | 7 |
| `NS-31Cdo4781-2009` | P1 | 882 | 2683 | 4.8× | 9 |
| `NS-33Cdo1507-2022` | P1 | 838 | 2673 | 3.8× | 9 |
| `NS-21Cdo4521-2011` | P2 | 871 | 2629 | 4.3× | 8 |
| `NS-23Cdo1001-2021` | P2 | 906 | 2761 | 8.3× | 11 |
| `NS-23Cdo2885-2022` | P2 | 876 | 2635 | 6.0× | 8 |
| `NS-25Cdo3925-2013` | P2 | 818 | 2585 | 5.5× | 3 |
| `NS-28Cdo1214-2023` | P2 | 895 | 2744 | 4.1× | 11 |
| `NS-21Cdo3976-2013` | P3 | 859 | 2692 | 2.7× | 13 |
| `NS-25Cdo2202-2018` | P3 | 854 | 2712 | 4.5× | 14 |
| `NS-30Cdo1982-2012` | P3 | 932 | 2825 | 4.2× | 9 |
| `NS-Cpjn206-2010` | P3 | 956 | 3091 | 14.5× | 11 |
| `NS-22Cdo3277-2014` | P4 | 895 | 2826 | 6.4× | 8 |
| `NS-22Cdo3732-2014` | P4 | 851 | 2755 | 3.3× | 4 |
| `NS-23Cdo672-2021` | P4 | 795 | 2450 | 6.9× | 6 |
| `NS-23Cdo2486-2020` | P5 | 770 | 2324 | 1.0× | 2 |
| `NS-26Cdo1670-2018` | P5 | 1047 | 3003 | 3.3× | 12 |
| `NS-33Odo343-2005` | P5 | 1000 | 2877 | 2.6× | 9 |
| `NS-31Cdo1468-2023` | P6 | 864 | 2612 | 2.7× | 7 |
| `NS-31Cdo2805-2011` | P6 | 915 | 2724 | 4.3× | 8 |
| `NS-Plsn1-2015` | P6 | 939 | 2965 | 16.1× | 12 |
| `NS-31Cdo1622-2021` | P7 | 919 | 2760 | 3.3× | 12 |
| `NS-31Cdo3263-2024` | P7 | 974 | 3023 | 10.7× | 13 |
| `NS-31Cdo353-2016` | P7 | 921 | 2802 | 3.5× | 2 |
| `NS-21Cdo1276-2001` | P8 | 837 | 2726 | 3.0× | 3 |
| `NS-22Cdo1172-2022` | P8 | 1023 | 3053 | 5.8× | 10 |
| `NS-22Cdo3192-2015` | P8 | 837 | 2625 | 4.3× | 1 |
| `NS-25Cdo2422-2019` | P8 | 903 | 2939 | 9.8× | 0 |
| `NS-31Cdo2273-2022` | P8 | 836 | 2668 | 13.9× | 4 |
| `NS-33Cdo42-2021` | P8 | 1011 | 3059 | 2.7× | 8 |
| `NS-21Cdo246-2008` | P9 | 889 | 2914 | 5.3× | 5 |
| `NS-22Cdo1843-2000` | P9 | 903 | 2706 | 2.4× | 4 |
| `NS-26Cdo732-98` | P9 | 817 | 2525 | 1.4× | 11 |
| `NS-31Cdo1038-2009` | P10 | 917 | 2925 | 5.0× | 10 |
| `NS-31Cdo365-2009` | P10 | 833 | 2614 | 2.6× | 4 |
| `NS-31Cdo4616-2010` | P10 | 895 | 2802 | 6.6× | 13 |
| `NS-25Cdo1417-2006` | P11 | 821 | 2657 | 2.2× | 15 |
| `NS-29Cdo3321-2020` | P11 | 959 | 2906 | 14.6× | 10 |
| `NS-31Cdo1778-2014` | P11 | 875 | 2699 | 3.4× | 13 |
| `NS-31Cdo2376-2021` | P11 | 953 | 3077 | 7.6× | 12 |
| `NS-23Cdo2856-2022` | P12 | 859 | 2612 | 3.9× | 14 |
| `NS-29Cdo61-2017` | P12 | 844 | 2564 | 3.5× | 16 |

## Review kandidáti pro Ondru

Filtry vybírají chains s mírným odchýlením od mediánu. Smyslem je dát review
prioritu, ne vytknout chybu — chains mohou být legitimně delší/kratší podle
složitosti judikátu.

### Bez citace §§

- `NS-25Cdo2422-2019` (P8 balancing principů) — 0 citací §§.

### Nejnižší komprese (distillate ≈ zdroj — možná příliš mnoho faktů)

- `NS-23Cdo2486-2020` (P5, 1.0×) — zdroj má jen 774 slov; zvážit, zda chain destiluje *metodu*, nebo jen převypráví.
- `NS-26Cdo732-98` (P9, 1.4×) — zdroj má jen 1184 slov; zvážit, zda chain destiluje *metodu*, nebo jen převypráví.
- `NS-25Cdo1417-2006` (P11, 2.2×) — zdroj má jen 1769 slov; zvážit, zda chain destiluje *metodu*, nebo jen převypráví.

### Nejkratší 5 (template suggestoval 400–600, prakticky min 770)

- `NS-23Cdo2486-2020` (P5, 770 slov) — Reasoning chain — odklad vykonatelnosti při hrozbě nevratné újmy (P5)
- `NS-23Cdo672-2021` (P4, 795 slov) — Reasoning chain — analogie přes srovnání s pozitivními případy (P4)
- `NS-26Cdo732-98` (P9, 817 slov) — Reasoning chain — povinnost tvrzení a poučovací povinnost soudu (P9)
- `NS-25Cdo3925-2013` (P2, 818 slov) — Reasoning chain — široký teleologický výklad pojmu "provoz" (P2)
- `NS-25Cdo1417-2006` (P11, 821 slov) — Reasoning chain — protiprávnost smluvního porušení i vůči třetím osobám (P11)

### Nejdelší 5

- `NS-26Cdo1670-2018` (P5, 1047 slov) — Reasoning chain — rebus sic stantibus u nájmu (P5)
- `NS-22Cdo1172-2022` (P8, 1023 slov) — Reasoning chain — teleologická redukce textu zákona u valorizace vnosu (P8)
- `NS-33Cdo42-2021` (P8, 1011 slov) — Reasoning chain — neúměrné zkrácení, balancing s kvantitativní hranicí (P8)
- `NS-33Odo343-2005` (P5, 1000 slov) — Reasoning chain — časové ohraničení výjimky rebus sic stantibus (P5)
- `NS-31Cdo3263-2024` (P7, 974 slov) — Reasoning chain — překonání vlastní judikatury kvůli pomíjení autonomie stran (P7)

### Kandidáti na reklasifikaci patternu (z verifikace 2026-04-19)

- `NS-23Cdo2486-2020` (aktuálně P5) — procesní usnesení o odkladu vykonatelnosti,
  ne hmotněprávní rozhodnutí o rebus sic stantibus. Zvážit reklasifikaci nebo
  vyřazení. Distillate má poznámku.
- `NS-23Cdo672-2021` (aktuálně P4) — vymezení rozsahu úřední odpovědnosti přes
  srovnání pozitivních/negativních precedentů; spíš P1 (kvalifikace).
