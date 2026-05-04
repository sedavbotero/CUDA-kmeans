This project was part of a university course on CUDA. Hence the use of Polish and references to university things.

# CUDA - projekt 2 - Algorytm k-means

## Struktura projektu

W katalogu `src/` znajduje się kod źródłowy.
W katalogu `inlude/` znajdziemy pliki nagłówkowe.
W katalogu `tests/` umieściłem prosty skrypt w bashu,
który sprawdza poprawność wyników wykorzystując pliki od prowadzącego

```
.
├── README.md
├── Makefile
├── include
│   ├── cuda_utils.cuh
│   ├── kmeans.h
│   └── utils.h
├── src
│   ├── kmeans_cpu.cpp
│   ├── kmeans_custom_kernel.cu
│   ├── kmeans_thrust.cu
│   └── main.c
└── tests
    └── test.bash
```

## Kompilacja i kompatybilność

- `make` - kompilacja projektu
- `make check` - kompilacja i uruchomienie skryptu testującego
- `make clean` - wyczyszczenie projektu z dotychczas zbudowanych plików

Starałem się zadbać o to, by kod był możliwie przenośny. Jest potrzebne użycie systemu Linux. karta graficzna musi wspierać compute capability ≥ 6.0 (`atomicAdd(double)`).

Sprawdziłem działanie programu na swoim komputerze, który ma system Linux Mint (wewnętrznie prawie identyczne do Ubuntu) i na komputerze z sali 304 z systemem Arch Linux.

Skrypt testujący tworzy plik `tests/times.txt`, w którym są umieszczane czasy wykonania w testach.

## Zaimplementowane warianty działania

- GPU1 - Własne kernele oparte o `atomicAdd`
- GPU2 - działanie oparte o thrust
- CPU - implementacja napisana pod działanie na cpu. Starałem się zadbać o to,
  by kompilator wykorzystywał instrukcje zwektoryzowane

## Dodatkowe punkty

We wszystkich wariantach wykorzystałem rekurencyjne template'y do określenia liczby wymiarów.

Za to były przewidziany dodatkowe 5 pkt.
