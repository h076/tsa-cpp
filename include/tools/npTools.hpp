#ifndef NPTOOLS_H_
#define NPTOOLS_H_

#include <stdexcept>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <numeric>

namespace tools {

    namespace np {

        inline xt::xarray<double> vander(const xt::xarray<double>& x, std::size_t N=0) {
            if (x.dimension() != 1)
                throw std::invalid_argument("tools::np::vander : x must be a one-dimensional array or sequence.");
            if (N == 0)
                N = x.size();

            const auto size = x.size();
            xt::xarray<double> V = xt::ones<double>({size, N + 1});

            xt::xarray<double> power = xt::ones<double>(x.shape());

            for (std::size_t d = 0; d <= N; d++) {
                //xt::view(V, xt::all(), N - d) = xt::pow(x, static_cast<int>(d));
                xt::view(V, xt::all(), N - d) = power;
                power *= x;
            }

            return V;
        }

        inline xt::xarray<double> polyfit(const xt::xarray<double>& x, const xt::xarray<double>& y, int deg) {
            // Create vandermonde matrix
            xt::xarray<double> X = vander(x, deg);

            // Compute coefs using least squares
            xt::xarray<double> Xt = xt::transpose(X);
            xt::xarray<double> XtX = xt::linalg::dot(Xt, X);
            xt::xarray<double> Xty = xt::linalg::dot(Xt, y);
            return xt::linalg::solve(XtX, Xty);
        }

    };
};

#endif // NPTOOLS_H_
