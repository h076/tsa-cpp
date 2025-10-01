#ifndef AUTOREG_H_
#define AUTOREG_H_

#include "coreTools.hpp"
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

namespace tools {

    inline double AROneHalfLife(xt::xtensor<double, 1> exog) {
        // centre th series
        xt::xtensor<double, 1> x = exog - xt::mean(exog);

        // get lagged and current values
        xt::xtensor<double, 1> x_lag = xt::view(x, xt::range(0, x.size()-1));
        auto x_cur = xt::view(x, xt::range(1, xt::placeholders::_));

        // add constant trend to lag values
        xt::xtensor<double, 2> x_lag_c = tools::addTrend(x_lag, "c", true);

        // Use OLS model to get phi
        linModels::OLSModel ols(x_lag_c, x_cur);
        ols.fit();
        double phi = ols.getParams()(1);

        // calculate half life and return
        double h = -(std::log(2.) / std::log(abs(phi)));
        return h;
    }

}

#endif // AUTOREG_H_
