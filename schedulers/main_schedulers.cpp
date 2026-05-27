/**
 * Proyecto Final - Programación Paralela y Concurrente
 * Versión: 3 - Experimentos de Schedulers (Tarea A)
 *
 * Evalúa empíricamente los planificadores de OpenMP para la generación
 * del fractal de Mandelbrot (carga de trabajo irregular por fila):
 *   - static  (chunk sizes: default, 1, 8, 32)
 *   - dynamic (chunk sizes: 1, 8, 32)
 *   - guided  (chunk sizes: 1, 8)
 *
 * El Mandelbrot es un caso ideal para este análisis porque las filas
 * del centro del plano complejo requieren muchas más iteraciones que
 * las filas de los bordes → carga muy desbalanceada con static.
 *
 * Compilar:
 *   g++ -O2 -std=c++17 -fopenmp -o mandelbrot_sched main_schedulers.cpp
 * Ejecutar:
 *   ./mandelbrot_sched
 * Controlar hilos:
 *   OMP_NUM_THREADS=4 ./mandelbrot_sched
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <omp.h>
#include <algorithm>

// ─── Parámetros de imagen ────────────────────────────────────────────────────
constexpr int    WIDTH      = 7680;
constexpr int    HEIGHT     = 4320;
constexpr int    MAX_ITER   = 1000;

// ─── Parámetros del plano complejo ───────────────────────────────────────────
constexpr double X_MIN      = -2.5;
constexpr double X_MAX      =  1.0;
constexpr double Y_MIN      = -1.25;
constexpr double Y_MAX      =  1.25;

// ─── Estructura de pixel RGB ─────────────────────────────────────────────────
struct Pixel { unsigned char r, g, b; };

// ─── Resultado de un experimento ─────────────────────────────────────────────
struct Result {
    std::string label;
    double      time;
    double      speedup;  // respecto al mejor static
};

// ─────────────────────────────────────────────────────────────────────────────
//  Mapeo de iteraciones a color
// ─────────────────────────────────────────────────────────────────────────────
inline Pixel iterToColor(int iter, int maxIter) {
    if (iter == maxIter) return {0, 0, 0};
    double t = static_cast<double>(iter) / maxIter;
    return {
        static_cast<unsigned char>(9   * (1-t) * t*t*t       * 255),
        static_cast<unsigned char>(15  * (1-t)*(1-t) * t*t   * 255),
        static_cast<unsigned char>(8.5 * (1-t)*(1-t)*(1-t)*t * 255)
    };
}

// ─────────────────────────────────────────────────────────────────────────────
//  Kernels de Mandelbrot con cada scheduler
//  (se definen por separado para que el pragma quede fijo en compilación)
// ─────────────────────────────────────────────────────────────────────────────

// STATIC — chunk por defecto (HEIGHT / nthreads)
double runStatic_default(std::vector<Pixel>& img) {
    const double dx = (X_MAX - X_MIN) / WIDTH;
    const double dy = (Y_MAX - Y_MIN) / HEIGHT;
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static) default(none) shared(img, dx, dy)
    for (int row = 0; row < HEIGHT; ++row) {
        double cy = Y_MIN + row * dy;
        for (int col = 0; col < WIDTH; ++col) {
            double cx = X_MIN + col * dx;
            double zx = 0, zy = 0; int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2*zx*zy + cy; zx = tmp; ++iter;
            }
            img[row * WIDTH + col] = iterToColor(iter, MAX_ITER);
        }
    }
    return omp_get_wtime() - t0;
}

// STATIC — chunk fijo
double runStatic_chunk(std::vector<Pixel>& img, int chunk) {
    const double dx = (X_MAX - X_MIN) / WIDTH;
    const double dy = (Y_MAX - Y_MIN) / HEIGHT;
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static, 1) default(none) shared(img, dx, dy, chunk)
    for (int row = 0; row < HEIGHT; ++row) {
        double cy = Y_MIN + row * dy;
        for (int col = 0; col < WIDTH; ++col) {
            double cx = X_MIN + col * dx;
            double zx = 0, zy = 0; int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2*zx*zy + cy; zx = tmp; ++iter;
            }
            img[row * WIDTH + col] = iterToColor(iter, MAX_ITER);
        }
    }
    return omp_get_wtime() - t0;
}

// DYNAMIC — chunk 1
double runDynamic_1(std::vector<Pixel>& img) {
    const double dx = (X_MAX - X_MIN) / WIDTH;
    const double dy = (Y_MAX - Y_MIN) / HEIGHT;
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 1) default(none) shared(img, dx, dy)
    for (int row = 0; row < HEIGHT; ++row) {
        double cy = Y_MIN + row * dy;
        for (int col = 0; col < WIDTH; ++col) {
            double cx = X_MIN + col * dx;
            double zx = 0, zy = 0; int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2*zx*zy + cy; zx = tmp; ++iter;
            }
            img[row * WIDTH + col] = iterToColor(iter, MAX_ITER);
        }
    }
    return omp_get_wtime() - t0;
}

// DYNAMIC — chunk 8
double runDynamic_8(std::vector<Pixel>& img) {
    const double dx = (X_MAX - X_MIN) / WIDTH;
    const double dy = (Y_MAX - Y_MIN) / HEIGHT;
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 8) default(none) shared(img, dx, dy)
    for (int row = 0; row < HEIGHT; ++row) {
        double cy = Y_MIN + row * dy;
        for (int col = 0; col < WIDTH; ++col) {
            double cx = X_MIN + col * dx;
            double zx = 0, zy = 0; int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2*zx*zy + cy; zx = tmp; ++iter;
            }
            img[row * WIDTH + col] = iterToColor(iter, MAX_ITER);
        }
    }
    return omp_get_wtime() - t0;
}

// DYNAMIC — chunk 32
double runDynamic_32(std::vector<Pixel>& img) {
    const double dx = (X_MAX - X_MIN) / WIDTH;
    const double dy = (Y_MAX - Y_MIN) / HEIGHT;
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 32) default(none) shared(img, dx, dy)
    for (int row = 0; row < HEIGHT; ++row) {
        double cy = Y_MIN + row * dy;
        for (int col = 0; col < WIDTH; ++col) {
            double cx = X_MIN + col * dx;
            double zx = 0, zy = 0; int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2*zx*zy + cy; zx = tmp; ++iter;
            }
            img[row * WIDTH + col] = iterToColor(iter, MAX_ITER);
        }
    }
    return omp_get_wtime() - t0;
}

// GUIDED — chunk 1
double runGuided_1(std::vector<Pixel>& img) {
    const double dx = (X_MAX - X_MIN) / WIDTH;
    const double dy = (Y_MAX - Y_MIN) / HEIGHT;
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(guided, 1) default(none) shared(img, dx, dy)
    for (int row = 0; row < HEIGHT; ++row) {
        double cy = Y_MIN + row * dy;
        for (int col = 0; col < WIDTH; ++col) {
            double cx = X_MIN + col * dx;
            double zx = 0, zy = 0; int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2*zx*zy + cy; zx = tmp; ++iter;
            }
            img[row * WIDTH + col] = iterToColor(iter, MAX_ITER);
        }
    }
    return omp_get_wtime() - t0;
}

// GUIDED — chunk 8
double runGuided_8(std::vector<Pixel>& img) {
    const double dx = (X_MAX - X_MIN) / WIDTH;
    const double dy = (Y_MAX - Y_MIN) / HEIGHT;
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(guided, 8) default(none) shared(img, dx, dy)
    for (int row = 0; row < HEIGHT; ++row) {
        double cy = Y_MIN + row * dy;
        for (int col = 0; col < WIDTH; ++col) {
            double cx = X_MIN + col * dx;
            double zx = 0, zy = 0; int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2*zx*zy + cy; zx = tmp; ++iter;
            }
            img[row * WIDTH + col] = iterToColor(iter, MAX_ITER);
        }
    }
    return omp_get_wtime() - t0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Imprimir tabla de resultados
// ─────────────────────────────────────────────────────────────────────────────
void printTable(const std::vector<Result>& results) {
    std::cout << "\n";
    std::cout << std::left
              << std::setw(28) << "Scheduler"
              << std::setw(14) << "Tiempo (s)"
              << std::setw(12) << "Speedup vs static_default"
              << "\n";
    std::cout << std::string(54, '-') << "\n";

    double baseTime = results[0].time;
    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(28) << r.label
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.time
                  << std::setw(12) << std::fixed << std::setprecision(3) << (baseTime / r.time)
                  << "\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Guardar resultados en CSV para graficar en el reporte
// ─────────────────────────────────────────────────────────────────────────────
void saveCSV(const std::vector<Result>& results, int nthreads) {
    std::string fname = "schedulers_results_" + std::to_string(nthreads) + "t.csv";
    std::ofstream f(fname);
    f << "scheduler,chunk_size,time_s,speedup\n";
    double baseTime = results[0].time;
    for (const auto& r : results) {
        f << r.label << "," << r.time << ","
          << std::fixed << std::setprecision(4) << (baseTime / r.time) << "\n";
    }
    std::cout << "  CSV guardado: " << fname << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    int nthreads = omp_get_max_threads();

    std::cout << "=== Proyecto Final — Experimentos de Schedulers ===\n";
    std::cout << "Hilos: " << nthreads
              << "  Resolución: " << WIDTH << "x" << HEIGHT
              << "  MaxIter: " << MAX_ITER << "\n";
    std::cout << "Nota: el Mandelbrot tiene carga IRREGULAR por fila\n"
              << "      (filas del centro = muchas más iteraciones)\n\n";

    std::vector<Pixel> img(WIDTH * HEIGHT);
    std::vector<Result> results;

    auto run = [&](const std::string& label, double t) {
        std::cout << "  [OK] " << label << " -> " << std::fixed
                  << std::setprecision(4) << t << " s\n";
        results.push_back({label, t, 0.0});
    };

    std::cout << "[Corriendo experimentos...]\n";
    run("static_default",  runStatic_default(img));
    run("static_chunk1",   runStatic_chunk(img, 1));
    run("dynamic_chunk1",  runDynamic_1(img));
    run("dynamic_chunk8",  runDynamic_8(img));
    run("dynamic_chunk32", runDynamic_32(img));
    run("guided_chunk1",   runGuided_1(img));
    run("guided_chunk8",   runGuided_8(img));

    printTable(results);

    // Encontrar el mejor
    auto best = std::min_element(results.begin(), results.end(),
        [](const Result& a, const Result& b){ return a.time < b.time; });
    std::cout << ">>> Scheduler óptimo: " << best->label
              << " (" << std::fixed << std::setprecision(4) << best->time << " s)\n\n";

    saveCSV(results, nthreads);

    return 0;
}