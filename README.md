# Predmetni projekat - Soft kompjuting 2018/2019.
Marko Vuković, RA 200/2015

Level 3 - Predefinisani projekat za 60 bodova 

## Pokretanje projekta
Potrebno je klonirati repozitorijum:
```
git clone https://github.com/vukovic-marko/projekat-soft
```
Potom se pozicionirati u preuzet folder u komandnoj liniji, i pokrenuti projekat sledećom komandom:
```
python main.py
```

## Provera tačnosti
Provera tačnosti projekta vrši se pozivanjem date skripte za testiranje rešenja:
```
python test.py
```

## Napomene o implementaciji
* kako bi izvršavanje bilo uspešno, potrebno je video zapise nad kojima se vrši analiza smestiti u folder dataset, koji se nalazi u root folderu kloniranog repozitorijuma
* u folderu mreza, nalazi se fajl model.h5, u kom je sacuvan model neuronske mreže koja vrši prepoznavanje cifara. Ukoliko je potrebno ponovo izvršiti treniranje, dovoljno je samo obrisati fajl model.h5
