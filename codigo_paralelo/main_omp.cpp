/**
 * Proyecto Final - Programación Paralela y Concurrente
 * Versión: 2 - Paralelización Base con OpenMP (generada por IA)
 *
 * Cambios respecto a la versión secuencial:
 *   - Tarea A: #pragma omp parallel for sobre filas del fractal
 *   - Tarea B: #pragma omp parallel for sobre filas del filtro
 *   - Medición de tiempo con omp_get_wtime() para mayor precisión
 *   - Reporte del número de hilos utilizados
 *
 * Compilar:
 *   g++ -O2 -std=c++17 -fopenmp -o mandelbrot_omp main_omp.cpp
 * Ejecutar:
 *   ./mandelbrot_omp
 * Controlar hilos:
 *   OMP_NUM_THREADS=4 ./mandelbrot_omp
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <omp.h>

// ─── Parámetros de imagen ────────────────────────────────────────────────────
constexpr int    WIDTH      = 7680;
constexpr int    HEIGHT     = 4320;
constexpr int    MAX_ITER   = 1000;

// ─── Parámetros del plano complejo ───────────────────────────────────────────
constexpr double X_MIN      = -2.5;
constexpr double X_MAX      =  1.0;
constexpr double Y_MIN      = -1.25;
constexpr double Y_MAX      =  1.25;

// ─── Parámetros del filtro Gaussiano ─────────────────────────────────────────
constexpr int    GAUSS_RADIUS = 7;

// ─── Estructura de pixel RGB ─────────────────────────────────────────────────
struct Pixel { unsigned char r, g, b; };

// ─────────────────────────────────────────────────────────────────────────────
//  Mapeo de iteraciones a color
// ─────────────────────────────────────────────────────────────────────────────
Pixel iterToColor(int iter, int maxIter) {
    if (iter == maxIter) return {0, 0, 0};

    double t = static_cast<double>(iter) / maxIter;
    unsigned char r = static_cast<unsigned char>(9   * (1-t) * t*t*t       * 255);
    unsigned char g = static_cast<unsigned char>(15  * (1-t)*(1-t) * t*t   * 255);
    unsigned char b = static_cast<unsigned char>(8.5 * (1-t)*(1-t)*(1-t)*t * 255);
    return {r, g, b};
}

// ─────────────────────────────────────────────────────────────────────────────
//  TAREA A: Generar Mandelbrot — paralelizado con OpenMP
// ─────────────────────────────────────────────────────────────────────────────
void generateMandelbrot(std::vector<Pixel>& image) {
    const double dx = (X_MAX - X_MIN) / WIDTH;
    const double dy = (Y_MAX - Y_MIN) / HEIGHT;

    #pragma omp parallel for schedule(static) default(none) \
        shared(image, dx, dy)
    for (int row = 0; row < HEIGHT; ++row) {
        double cy = Y_MIN + row * dy;
        for (int col = 0; col < WIDTH; ++col) {
            double cx = X_MIN + col * dx;

            double zx = 0.0, zy = 0.0;
            int iter = 0;
            while (zx*zx + zy*zy <= 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = tmp;
                ++iter;
            }
            image[row * WIDTH + col] = iterToColor(iter, MAX_ITER);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Construcción del kernel Gaussiano 2D
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> buildGaussianKernel(int radius, double sigma) {
    int size = 2 * radius + 1;
    std::vector<double> kernel(size * size);
    double sum = 0.0;
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            double val = std::exp(-(kx*kx + ky*ky) / (2.0 * sigma * sigma));
            kernel[(ky + radius) * size + (kx + radius)] = val;
            sum += val;
        }
    }
    for (auto& v : kernel) v /= sum;
    return kernel;
}

// ─────────────────────────────────────────────────────────────────────────────
//  TAREA B: Gaussian Blur — paralelizado con OpenMP
// ─────────────────────────────────────────────────────────────────────────────
void applyGaussianBlur(const std::vector<Pixel>& src,
                       std::vector<Pixel>&       dst,
                       int radius) {
    const double sigma  = radius / 2.0;
    const auto   kernel = buildGaussianKernel(radius, sigma);
    const int    ksize  = 2 * radius + 1;

    #pragma omp parallel for schedule(static) default(none) \
        shared(src, dst, kernel, radius, ksize)
    for (int row = 0; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            double accR = 0, accG = 0, accB = 0;

            for (int ky = -radius; ky <= radius; ++ky) {
                int ny = row + ky;
                if (ny < 0) ny = 0;
                if (ny >= HEIGHT) ny = HEIGHT - 1;

                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = col + kx;
                    if (nx < 0) nx = 0;
                    if (nx >= WIDTH) nx = WIDTH - 1;

                    double w = kernel[(ky + radius) * ksize + (kx + radius)];
                    const Pixel& p = src[ny * WIDTH + nx];
                    accR += w * p.r;
                    accG += w * p.g;
                    accB += w * p.b;
                }
            }
            dst[row * WIDTH + col] = {
                static_cast<unsigned char>(accR),
                static_cast<unsigned char>(accG),
                static_cast<unsigned char>(accB)
            };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Guardar imagen PPM
// ─────────────────────────────────────────────────────────────────────────────
void savePPM(const std::string& filename, const std::vector<Pixel>& image) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()),
               image.size() * sizeof(Pixel));
    std::cout << "  Imagen guardada: " << filename << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== Proyecto Final — Versión OpenMP (Base IA) ===\n";
    std::cout << "Hilos disponibles: " << omp_get_max_threads() << "\n";
    std::cout << "Resolución: " << WIDTH << "x" << HEIGHT
              << "  MaxIter: " << MAX_ITER
              << "  Gauss radius: " << GAUSS_RADIUS << "\n\n";

    std::vector<Pixel> imageA(WIDTH * HEIGHT);
    std::vector<Pixel> imageB(WIDTH * HEIGHT);

    // ── Tarea A ───────────────────────────────────────────────────────────────
    std::cout << "[Tarea A] Generando Mandelbrot con " 
              << omp_get_max_threads() << " hilos...\n";
    double t0 = omp_get_wtime();
    generateMandelbrot(imageA);
    double t1 = omp_get_wtime();
    double timeA = t1 - t0;
    std::cout << "  Tiempo Tarea A: " << timeA << " s\n\n";

    savePPM("mandelbrot_omp_raw.ppm", imageA);

    // ── Tarea B ───────────────────────────────────────────────────────────────
    std::cout << "[Tarea B] Gaussian Blur con "
              << omp_get_max_threads() << " hilos...\n";
    double t2 = omp_get_wtime();
    applyGaussianBlur(imageA, imageB, GAUSS_RADIUS);
    double t3 = omp_get_wtime();
    double timeB = t3 - t2;
    std::cout << "  Tiempo Tarea B: " << timeB << " s\n\n";

    savePPM("mandelbrot_omp_blur.ppm", imageB);

    // ── Resumen ───────────────────────────────────────────────────────────────
    std::cout << "=== RESUMEN ===\n";
    std::cout << "  Hilos usados:             " << omp_get_max_threads() << "\n";
    std::cout << "  Tarea A (Mandelbrot):     " << timeA << " s\n";
    std::cout << "  Tarea B (Gaussian Blur):  " << timeB << " s\n";
    std::cout << "  TOTAL:                    " << (timeA + timeB) << " s\n";

    return 0;
}