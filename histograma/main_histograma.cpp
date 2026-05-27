/**
 * Proyecto Final - Programación Paralela y Concurrente
 * Versión: 4 - Histograma de Colores + False Sharing
 *
 * Compara tres implementaciones para calcular el histograma de colores
 * de la imagen del Mandelbrot:
 *
 *   1. atomic    — exclusión mutua con #pragma omp atomic
 *   2. reduction — variables locales por hilo + reducción final (correcto)
 *   3. falsesharing — arreglo compartido sin padding (demostración del problema)
 *
 * El histograma cuenta cuántos píxeles hay de cada valor (0-255)
 * por canal (R, G, B) → 3 arreglos de 256 enteros.
 *
 * Compilar:
 *   g++ -O2 -std=c++17 -fopenmp -o mandelbrot_hist main_histogram.cpp
 * Ejecutar:
 *   OMP_NUM_THREADS=4 ./mandelbrot_hist
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <omp.h>

// ─── Parámetros de imagen ────────────────────────────────────────────────────
constexpr int    WIDTH      = 7680;
constexpr int    HEIGHT     = 4320;
constexpr int    MAX_ITER   = 1000;
constexpr int    HIST_SIZE  = 256;

// ─── Parámetros del plano complejo ───────────────────────────────────────────
constexpr double X_MIN      = -2.5;
constexpr double X_MAX      =  1.0;
constexpr double Y_MIN      = -1.25;
constexpr double Y_MAX      =  1.25;

// ─── Padding para evitar false sharing (línea de caché = 64 bytes) ───────────
constexpr int CACHE_LINE    = 64;
constexpr int INT_PER_LINE  = CACHE_LINE / sizeof(int);  // = 16

// ─── Estructura de pixel RGB ─────────────────────────────────────────────────
struct Pixel { unsigned char r, g, b; };

// ─────────────────────────────────────────────────────────────────────────────
//  Generar imagen Mandelbrot (con dynamic chunk8 — scheduler óptimo)
// ─────────────────────────────────────────────────────────────────────────────
void generateMandelbrot(std::vector<Pixel>& image) {
    const double dx = (X_MAX - X_MIN) / WIDTH;
    const double dy = (Y_MAX - Y_MIN) / HEIGHT;

    #pragma omp parallel for schedule(dynamic, 8) default(none) shared(image, dx, dy)
    for (int row = 0; row < HEIGHT; ++row) {
        double cy = Y_MIN + row * dy;
        for (int col = 0; col < WIDTH; ++col) {
            double cx = X_MIN + col * dx;
            double zx = 0, zy = 0; int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2*zx*zy + cy; zx = tmp; ++iter;
            }
            double t = static_cast<double>(iter) / MAX_ITER;
            if (iter == MAX_ITER) { image[row*WIDTH+col] = {0,0,0}; continue; }
            image[row*WIDTH+col] = {
                static_cast<unsigned char>(9   * (1-t)*t*t*t       * 255),
                static_cast<unsigned char>(15  * (1-t)*(1-t)*t*t   * 255),
                static_cast<unsigned char>(8.5 * (1-t)*(1-t)*(1-t)*t * 255)
            };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  MÉTODO 1: Histograma con #pragma omp atomic
//  Problema: alta contención — muchos hilos actualizan las mismas posiciones
// ─────────────────────────────────────────────────────────────────────────────
double histogramAtomic(const std::vector<Pixel>& image,
                       long long histR[HIST_SIZE],
                       long long histG[HIST_SIZE],
                       long long histB[HIST_SIZE]) {
    memset(histR, 0, HIST_SIZE * sizeof(long long));
    memset(histG, 0, HIST_SIZE * sizeof(long long));
    memset(histB, 0, HIST_SIZE * sizeof(long long));

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 8) default(none) \
        shared(image, histR, histG, histB)
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        #pragma omp atomic
        histR[image[i].r]++;
        #pragma omp atomic
        histG[image[i].g]++;
        #pragma omp atomic
        histB[image[i].b]++;
    }

    return omp_get_wtime() - t0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  MÉTODO 2: Histograma con variables locales por hilo + reducción manual
//  Correcto y eficiente: cada hilo trabaja en su propia copia, sin contención
// ─────────────────────────────────────────────────────────────────────────────
double histogramReduction(const std::vector<Pixel>& image,
                          long long histR[HIST_SIZE],
                          long long histG[HIST_SIZE],
                          long long histB[HIST_SIZE]) {
    memset(histR, 0, HIST_SIZE * sizeof(long long));
    memset(histG, 0, HIST_SIZE * sizeof(long long));
    memset(histB, 0, HIST_SIZE * sizeof(long long));

    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(image, histR, histG, histB)
    {
        // Cada hilo tiene su histograma local — sin contención
        long long localR[HIST_SIZE] = {};
        long long localG[HIST_SIZE] = {};
        long long localB[HIST_SIZE] = {};

        #pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            localR[image[i].r]++;
            localG[image[i].g]++;
            localB[image[i].b]++;
        }

        // Reducción: sumar histogramas locales al global (una sola sección crítica)
        #pragma omp critical
        {
            for (int v = 0; v < HIST_SIZE; ++v) {
                histR[v] += localR[v];
                histG[v] += localG[v];
                histB[v] += localB[v];
            }
        }
    }

    return omp_get_wtime() - t0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  MÉTODO 3: Demostración de FALSE SHARING
//  Cada hilo escribe en posiciones contiguas de un arreglo compartido
//  → las posiciones caen en la misma línea de caché → invalidaciones constantes
// ─────────────────────────────────────────────────────────────────────────────
double histogramFalseSharing(const std::vector<Pixel>& image,
                             long long histR[HIST_SIZE],
                             long long histG[HIST_SIZE],
                             long long histB[HIST_SIZE]) {
    int nthreads = omp_get_max_threads();

    // Arreglo SIN padding: thread i escribe en [i], thread i+1 en [i+1]
    // → misma línea de caché → false sharing
    std::vector<long long> partialR(nthreads * HIST_SIZE, 0);
    std::vector<long long> partialG(nthreads * HIST_SIZE, 0);
    std::vector<long long> partialB(nthreads * HIST_SIZE, 0);

    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) \
        shared(image, partialR, partialG, partialB, nthreads)
    {
        int tid = omp_get_thread_num();
        int base = tid * HIST_SIZE;  // offset sin padding → false sharing

        #pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            partialR[base + image[i].r]++;
            partialG[base + image[i].g]++;
            partialB[base + image[i].b]++;
        }
    }

    // Reducción final
    memset(histR, 0, HIST_SIZE * sizeof(long long));
    memset(histG, 0, HIST_SIZE * sizeof(long long));
    memset(histB, 0, HIST_SIZE * sizeof(long long));
    for (int t = 0; t < nthreads; ++t) {
        int base = t * HIST_SIZE;
        for (int v = 0; v < HIST_SIZE; ++v) {
            histR[v] += partialR[base + v];
            histG[v] += partialG[base + v];
            histB[v] += partialB[base + v];
        }
    }

    return omp_get_wtime() - t0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Guardar histograma en CSV
// ─────────────────────────────────────────────────────────────────────────────
void saveHistogramCSV(const std::string& fname,
                      long long histR[HIST_SIZE],
                      long long histG[HIST_SIZE],
                      long long histB[HIST_SIZE]) {
    std::ofstream f(fname);
    f << "value,R,G,B\n";
    for (int i = 0; i < HIST_SIZE; ++i)
        f << i << "," << histR[i] << "," << histG[i] << "," << histB[i] << "\n";
    std::cout << "  CSV guardado: " << fname << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Verificar que los dos histogramas son iguales
// ─────────────────────────────────────────────────────────────────────────────
bool histogramsEqual(long long a[HIST_SIZE], long long b[HIST_SIZE]) {
    for (int i = 0; i < HIST_SIZE; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    int nthreads = omp_get_max_threads();
    std::cout << "=== Proyecto Final — Histograma + False Sharing ===\n";
    std::cout << "Hilos: " << nthreads
              << "  Resolución: " << WIDTH << "x" << HEIGHT << "\n\n";

    // Generar imagen
    std::cout << "[Generando imagen Mandelbrot...]\n";
    std::vector<Pixel> image(WIDTH * HEIGHT);
    double tGen = 0;
    {
        double t0 = omp_get_wtime();
        generateMandelbrot(image);
        tGen = omp_get_wtime() - t0;
    }
    std::cout << "  Generación: " << std::fixed << std::setprecision(4) << tGen << " s\n\n";

    long long histR1[HIST_SIZE], histG1[HIST_SIZE], histB1[HIST_SIZE];
    long long histR2[HIST_SIZE], histG2[HIST_SIZE], histB2[HIST_SIZE];
    long long histR3[HIST_SIZE], histG3[HIST_SIZE], histB3[HIST_SIZE];

    // ── Método 1: atomic ──────────────────────────────────────────────────────
    std::cout << "[Método 1] atomic...\n";
    double tAtomic = histogramAtomic(image, histR1, histG1, histB1);
    std::cout << "  Tiempo: " << std::fixed << std::setprecision(4) << tAtomic << " s\n\n";

    // ── Método 2: reduction ───────────────────────────────────────────────────
    std::cout << "[Método 2] reduction (variables locales)...\n";
    double tReduction = histogramReduction(image, histR2, histG2, histB2);
    std::cout << "  Tiempo: " << std::fixed << std::setprecision(4) << tReduction << " s\n\n";

    // ── Método 3: false sharing ───────────────────────────────────────────────
    std::cout << "[Método 3] false sharing (sin padding)...\n";
    double tFalseSharing = histogramFalseSharing(image, histR3, histG3, histB3);
    std::cout << "  Tiempo: " << std::fixed << std::setprecision(4) << tFalseSharing << " s\n\n";

    // ── Verificación de correctitud ───────────────────────────────────────────
    bool ok12 = histogramsEqual(histR1, histR2) &&
                histogramsEqual(histG1, histG2) &&
                histogramsEqual(histB1, histB2);
    bool ok13 = histogramsEqual(histR1, histR3) &&
                histogramsEqual(histG1, histG3) &&
                histogramsEqual(histB1, histB3);
    std::cout << "Verificación atomic vs reduction:    " << (ok12 ? "OK" : "DIFERENTE") << "\n";
    std::cout << "Verificación atomic vs falsesharing: " << (ok13 ? "OK" : "DIFERENTE") << "\n\n";

    // ── Tabla comparativa ─────────────────────────────────────────────────────
    std::cout << std::left
              << std::setw(30) << "Método"
              << std::setw(14) << "Tiempo (s)"
              << std::setw(14) << "Speedup vs atomic"
              << "\n";
    std::cout << std::string(58, '-') << "\n";
    std::cout << std::left << std::setw(30) << "atomic"
              << std::setw(14) << std::fixed << std::setprecision(4) << tAtomic
              << std::setw(14) << "1.000" << "\n";
    std::cout << std::left << std::setw(30) << "reduction (local vars)"
              << std::setw(14) << std::fixed << std::setprecision(4) << tReduction
              << std::setw(14) << std::fixed << std::setprecision(3) << (tAtomic / tReduction) << "\n";
    std::cout << std::left << std::setw(30) << "false sharing (sin padding)"
              << std::setw(14) << std::fixed << std::setprecision(4) << tFalseSharing
              << std::setw(14) << std::fixed << std::setprecision(3) << (tAtomic / tFalseSharing) << "\n";

    // ── Análisis de false sharing ─────────────────────────────────────────────
    std::cout << "\n=== ANÁLISIS DE FALSE SHARING ===\n";
    std::cout << "Línea de caché: " << CACHE_LINE << " bytes = "
              << INT_PER_LINE << " ints por línea\n";
    std::cout << "Sin padding: hilos adyacentes comparten líneas de caché\n";
    std::cout << "             → invalidaciones constantes entre núcleos\n";
    if (tFalseSharing > tReduction)
        std::cout << "Resultado: false sharing fue "
                  << std::fixed << std::setprecision(2)
                  << (tFalseSharing / tReduction) << "x MÁS LENTO que reduction\n";
    else
        std::cout << "Resultado: en este hardware el efecto fue mínimo\n"
                  << "           (posible: pocos núcleos o caché muy grande)\n";

    // ── Guardar CSV ───────────────────────────────────────────────────────────
    saveHistogramCSV("histogram.csv", histR2, histG2, histB2);

    return 0;
}