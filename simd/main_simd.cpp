/**
 * Proyecto Final - Programación Paralela y Concurrente
 * Versión: 5 - SPMD + Vectorización (Tarea B)
 *
 * Modifica el filtro de convolución Gaussiana para forzar vectorización
 * usando directivas OpenMP SIMD (estructura SPMD).
 *
 * Comparaciones:
 *   1. blur_baseline  — paralelo con dynamic chunk8 (sin vectorización explícita)
 *   2. blur_simd      — paralelo + #pragma omp simd en el bucle interno
 *   3. blur_simd_aligned — datos alineados a 64 bytes + simd aligned
 *
 * Verificar vectorización con:
 *   g++ -O2 -std=c++17 -fopenmp -fopt-info-vec -o mandelbrot_simd main_simd.cpp
 *   (busca líneas que digan "vectorized" en la salida del compilador)
 *
 * Compilar normal:
 *   g++ -O2 -std=c++17 -fopenmp -o mandelbrot_simd main_simd.cpp
 * Ejecutar:
 *   OMP_NUM_THREADS=4 ./mandelbrot_simd
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>
#include <omp.h>

// ─── Parámetros de imagen ────────────────────────────────────────────────────
constexpr int    WIDTH        = 7680;
constexpr int    HEIGHT       = 4320;
constexpr int    MAX_ITER     = 1000;
constexpr int    GAUSS_RADIUS = 7;

// ─── Parámetros del plano complejo ───────────────────────────────────────────
constexpr double X_MIN        = -2.5;
constexpr double X_MAX        =  1.0;
constexpr double Y_MIN        = -1.25;
constexpr double Y_MAX        =  1.25;

// ─── Estructura de pixel RGB ─────────────────────────────────────────────────
struct Pixel { unsigned char r, g, b; };

// ─── Canales separados (float) para facilitar vectorización ──────────────────
// El compilador vectoriza mejor arreglos de float que structs de uchar
struct ImageFloat {
    std::vector<float> R, G, B;
    ImageFloat(int size) : R(size), G(size), B(size) {}
};

// ─────────────────────────────────────────────────────────────────────────────
//  Generar Mandelbrot con scheduler óptimo
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
                static_cast<unsigned char>(9   * (1-t)*t*t*t         * 255),
                static_cast<unsigned char>(15  * (1-t)*(1-t)*t*t     * 255),
                static_cast<unsigned char>(8.5 * (1-t)*(1-t)*(1-t)*t * 255)
            };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Convertir Pixel[] → canales float separados (AoS → SoA)
//  Structure of Arrays es fundamental para que el compilador vectorice
// ─────────────────────────────────────────────────────────────────────────────
void pixelsToFloat(const std::vector<Pixel>& src, ImageFloat& dst) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        dst.R[i] = static_cast<float>(src[i].r);
        dst.G[i] = static_cast<float>(src[i].g);
        dst.B[i] = static_cast<float>(src[i].b);
    }
}

void floatToPixels(const ImageFloat& src, std::vector<Pixel>& dst) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        dst[i] = {
            static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, src.R[i]))),
            static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, src.G[i]))),
            static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, src.B[i])))
        };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Kernel Gaussiano
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> buildGaussianKernel(int radius) {
    int size = 2 * radius + 1;
    std::vector<float> kernel(size * size);
    float sigma = radius / 2.0f;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky)
        for (int kx = -radius; kx <= radius; ++kx) {
            float val = std::exp(-(kx*kx + ky*ky) / (2.0f * sigma * sigma));
            kernel[(ky+radius)*(2*radius+1) + (kx+radius)] = val;
            sum += val;
        }
    for (auto& v : kernel) v /= sum;
    return kernel;
}

// ─────────────────────────────────────────────────────────────────────────────
//  MÉTODO 1: Blur baseline (paralelo, sin SIMD explícito)
// ─────────────────────────────────────────────────────────────────────────────
double blurBaseline(const ImageFloat& src, ImageFloat& dst, int radius,
                    const std::vector<float>& kernel) {
    const int ksize = 2 * radius + 1;
    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 8) default(none) \
        shared(src, dst, kernel, radius, ksize)
    for (int row = 0; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            float accR = 0, accG = 0, accB = 0;
            for (int ky = -radius; ky <= radius; ++ky) {
                int ny = std::max(0, std::min(HEIGHT-1, row + ky));
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = std::max(0, std::min(WIDTH-1, col + kx));
                    float w = kernel[(ky+radius)*ksize + (kx+radius)];
                    accR += w * src.R[ny*WIDTH + nx];
                    accG += w * src.G[ny*WIDTH + nx];
                    accB += w * src.B[ny*WIDTH + nx];
                }
            }
            dst.R[row*WIDTH+col] = accR;
            dst.G[row*WIDTH+col] = accG;
            dst.B[row*WIDTH+col] = accB;
        }
    }
    return omp_get_wtime() - t0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  MÉTODO 2: Blur con SPMD — #pragma omp simd en el bucle más interno
//  El bucle sobre kx (columnas del kernel) se vectoriza explícitamente
// ─────────────────────────────────────────────────────────────────────────────
double blurSIMD(const ImageFloat& src, ImageFloat& dst, int radius,
                const std::vector<float>& kernel) {
    const int ksize = 2 * radius + 1;

    // Punteros raw para que el compilador no dude en vectorizar
    const float* __restrict__ srcR = src.R.data();
    const float* __restrict__ srcG = src.G.data();
    const float* __restrict__ srcB = src.B.data();
    float* __restrict__ dstR = dst.R.data();
    float* __restrict__ dstG = dst.G.data();
    float* __restrict__ dstB = dst.B.data();
    const float* __restrict__ kern = kernel.data();

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 8) default(none)        \
        shared(srcR, srcG, srcB, dstR, dstG, dstB, kern, radius, ksize)
    for (int row = 0; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            float accR = 0.0f, accG = 0.0f, accB = 0.0f;

            for (int ky = -radius; ky <= radius; ++ky) {
                int ny = std::max(0, std::min(HEIGHT-1, row + ky));
                int base_src = ny * WIDTH;
                int base_k   = (ky + radius) * ksize;

                // ── SPMD: vectorizar el bucle más interno (sobre kx) ─────────
                #pragma omp simd reduction(+:accR, accG, accB)
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = col + kx;
                    // Clamp manual (sin branch para no romper vectorización)
                    nx = nx < 0 ? 0 : (nx >= WIDTH ? WIDTH-1 : nx);
                    float w = kern[base_k + kx + radius];
                    accR += w * srcR[base_src + nx];
                    accG += w * srcG[base_src + nx];
                    accB += w * srcB[base_src + nx];
                }
            }
            dstR[row*WIDTH+col] = accR;
            dstG[row*WIDTH+col] = accG;
            dstB[row*WIDTH+col] = accB;
        }
    }
    return omp_get_wtime() - t0;
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
    int nthreads = omp_get_max_threads();
    std::cout << "=== Proyecto Final — SPMD + Vectorización (Tarea B) ===\n";
    std::cout << "Hilos: " << nthreads
              << "  Resolución: " << WIDTH << "x" << HEIGHT
              << "  Gauss radius: " << GAUSS_RADIUS << "\n\n";

    // Generar imagen
    std::cout << "[Generando imagen Mandelbrot...]\n";
    std::vector<Pixel> imageRaw(WIDTH * HEIGHT);
    double t0 = omp_get_wtime();
    generateMandelbrot(imageRaw);
    std::cout << "  Generación: " << std::fixed << std::setprecision(4)
              << (omp_get_wtime() - t0) << " s\n\n";

    // Convertir a SoA float
    ImageFloat srcFloat(WIDTH * HEIGHT);
    ImageFloat dstFloat1(WIDTH * HEIGHT);
    ImageFloat dstFloat2(WIDTH * HEIGHT);
    pixelsToFloat(imageRaw, srcFloat);

    auto kernel = buildGaussianKernel(GAUSS_RADIUS);

    // ── Método 1: baseline ────────────────────────────────────────────────────
    std::cout << "[Método 1] Blur baseline (sin SIMD explícito)...\n";
    double tBaseline = blurBaseline(srcFloat, dstFloat1, GAUSS_RADIUS, kernel);
    std::cout << "  Tiempo: " << std::fixed << std::setprecision(4) << tBaseline << " s\n\n";

    // ── Método 2: SIMD ────────────────────────────────────────────────────────
    std::cout << "[Método 2] Blur SPMD con #pragma omp simd...\n";
    double tSIMD = blurSIMD(srcFloat, dstFloat2, GAUSS_RADIUS, kernel);
    std::cout << "  Tiempo: " << std::fixed << std::setprecision(4) << tSIMD << " s\n\n";

    // ── Tabla comparativa ─────────────────────────────────────────────────────
    std::cout << std::left
              << std::setw(32) << "Método"
              << std::setw(14) << "Tiempo (s)"
              << std::setw(14) << "Speedup"
              << "\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::left << std::setw(32) << "baseline (paralelo)"
              << std::setw(14) << std::fixed << std::setprecision(4) << tBaseline
              << std::setw(14) << "1.000" << "\n";
    std::cout << std::left << std::setw(32) << "SPMD omp simd"
              << std::setw(14) << std::fixed << std::setprecision(4) << tSIMD
              << std::setw(14) << std::fixed << std::setprecision(3)
              << (tBaseline / tSIMD) << "\n";

    // ── Guardar imagen resultado ──────────────────────────────────────────────
    std::vector<Pixel> result(WIDTH * HEIGHT);
    floatToPixels(dstFloat2, result);
    savePPM("mandelbrot_simd_blur.ppm", result);

    // ── Instrucciones para verificar vectorización ────────────────────────────
    std::cout << "\n=== VERIFICAR VECTORIZACIÓN ===\n";
    std::cout << "Recompila con:\n";
    std::cout << "  g++ -O2 -std=c++17 -fopenmp -fopt-info-vec \\\n";
    std::cout << "      -o mandelbrot_simd main_simd.cpp 2>&1 | grep vectorized\n";
    std::cout << "Busca líneas como:\n";
    std::cout << "  main_simd.cpp:NNN:N: optimized: loop vectorized\n";

    return 0;
}