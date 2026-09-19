/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of the TRT-VLBM experiment reproduction code.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program (see LICENSE). If not, see
 * <https://www.gnu.org/licenses/>.
 */

#include <stddef.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

/*
 * Double-precision zero-force D3N7 collision and periodic transport.
 * The population layout is [Nx, Ny, Nz, 7, 4] in C order, exactly matching
 * d3n7_taylorgreen.  Non-periodic links are corrected afterwards by the
 * solver's existing Python boundary-condition method.
 */
API void d3n7_collide_stream(
    const double *f,
    const double *w,
    double *fstar,
    double *nextf,
    int nx,
    int ny,
    int nz,
    double h,
    double a,
    double alpha,
    double relax1,
    double relax2
) {
    const int ex[7] = {1, 0, 0, -1, 0, 0, 0};
    const int ey[7] = {0, 1, 0, 0, -1, 0, 0};
    const int ez[7] = {0, 0, 1, 0, 0, -1, 0};
    const int opp[7] = {3, 4, 5, 0, 1, 2, 6};
    const double half_alpha = 0.5 * alpha;
    const double rest_weight = 1.0 - 6.0 * a;

    #ifdef _OPENMP

    #pragma omp parallel for collapse(3) schedule(static)

    #endif
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                const size_t cell = ((size_t)x * (size_t)ny + (size_t)y)
                                  * (size_t)nz + (size_t)z;
                const size_t wb = cell * 4u;
                const size_t fb = cell * 28u;
                const double rho = w[wb];
                const double q1 = w[wb + 1u];
                const double q2 = w[wb + 2u];
                const double q3 = w[wb + 3u];
                const double pressure_h2 = rho - 1.0;
                const double inv_rho = 1.0 / rho;
                double flux[3][4];
                double equilibrium[28];
                double post[28];

                flux[0][0] = q1;
                flux[0][1] = q1*q1*inv_rho + pressure_h2;
                flux[0][2] = q1*q2*inv_rho;
                flux[0][3] = q1*q3*inv_rho;
                flux[1][0] = q2;
                flux[1][1] = flux[0][2];
                flux[1][2] = q2*q2*inv_rho + pressure_h2;
                flux[1][3] = q2*q3*inv_rho;
                flux[2][0] = q3;
                flux[2][1] = flux[0][3];
                flux[2][2] = flux[1][3];
                flux[2][3] = q3*q3*inv_rho + pressure_h2;

                for (int axis = 0; axis < 3; ++axis) {
                    const int positive = axis;
                    const int negative = axis + 3;
                    for (int component = 0; component < 4; ++component) {
                        const double base = a * w[wb + (size_t)component];
                        const double correction = half_alpha * flux[axis][component];
                        equilibrium[positive*4 + component] = base + correction;
                        equilibrium[negative*4 + component] = base - correction;
                    }
                }
                for (int component = 0; component < 4; ++component) {
                    equilibrium[24 + component] =
                        rest_weight * w[wb + (size_t)component];
                }

                for (int direction = 0; direction < 7; ++direction) {
                    const int opposite = opp[direction];
                    for (int component = 0; component < 4; ++component) {
                        const int local = direction*4 + component;
                        const int opposite_local = opposite*4 + component;
                        const double nonequilibrium = equilibrium[local] - f[fb + (size_t)local];
                        const double opposite_nonequilibrium =
                            equilibrium[opposite_local] - f[fb + (size_t)opposite_local];
                        post[local] = f[fb + (size_t)local]
                            + relax1 * nonequilibrium
                            + relax2 * opposite_nonequilibrium;
                        fstar[fb + (size_t)local] = post[local];
                    }
                }

                for (int direction = 0; direction < 7; ++direction) {
                    int destination_x = x + ex[direction];
                    int destination_y = y + ey[direction];
                    int destination_z = z + ez[direction];
                    if (destination_x == nx) destination_x = 0;
                    if (destination_x < 0) destination_x = nx - 1;
                    if (destination_y == ny) destination_y = 0;
                    if (destination_y < 0) destination_y = ny - 1;
                    if (destination_z == nz) destination_z = 0;
                    if (destination_z < 0) destination_z = nz - 1;
                    const size_t destination =
                        (((size_t)destination_x * (size_t)ny
                          + (size_t)destination_y) * (size_t)nz
                          + (size_t)destination_z) * 28u
                        + (size_t)direction * 4u;
                    for (int component = 0; component < 4; ++component) {
                        nextf[destination + (size_t)component] =
                            post[direction*4 + component];
                    }
                }
            }
        }
    }
    (void)h;
}

API void d3n7_sum_populations(
    const double *f,
    double *w,
    size_t cells
) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t cell = 0; cell < cells; ++cell) {
        const size_t fb = cell * 28u;
        const size_t wb = cell * 4u;
        for (int component = 0; component < 4; ++component) {
            double value = 0.0;
            for (int direction = 0; direction < 7; ++direction) {
                value += f[fb + (size_t)direction*4u + (size_t)component];
            }
            w[wb + (size_t)component] = value;
        }
    }
}
