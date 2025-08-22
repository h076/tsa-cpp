#ifndef HURST_H_
#define HURST_H_

#include "../tools/npTools.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xadapt.hpp>

namespace tests {

    /*
     *
     * The goal of the hurst exponent is to provide us with a scalar value
     * that will help identify wether a series is mean reverting, random walk, or trending.
     *
     * We use the variance of a log price series to assess the rate of diffusion
     * behaviour. A time series can be characterised in the following manner ...
     *
     *  - H < 0.5 : Mean reverting
     *  - H == 0.5 : Geometric Brownian Motion
     *  - H > 0.5 : Trending
     *
     */
    inline double hurst(const xt::xarray<double>& ts) {
        /*
         * ts : xarray<double>
         *     - Time series upon which the Hurst Exponent will be calculated
         *
         * Returns ...
         *
         * 'double'
         *     - The Hurst Exponent from the poly fit output
         *
         */

        xt::xarray<int> lags = xt::arange(2, 100, 1);

        std::vector<double> tau_vec;
        for (int lag : lags) {
            auto diff = xt::view(ts, xt::range(lag, xt::all())) - xt::view(ts, xt::range(0, ts.size() - lag));
             tau_vec.push_back(std::sqrt(xt::stddev(diff)()));
        }

        xt::xarray<double> tau_x = xt::adapt(tau_vec);

        xt::xarray<double> poly = tools::np::polyfit(xt::log(lags), xt::log(tau_x), 1);

        // Having issues with hurst being negative ....
        return std::max(0.0, poly[0]*2.0);
    }
};

#endif // HURST_H_
