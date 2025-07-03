#ifndef REGRESSIONMODEL_H_
#define REGRESSIONMODEL_H_

#include <xtensor/containers/xarray.hpp>

class RegressionModel {

    protected:
        xt::xarray<double> X;
        xt::xarray<double> y;

};

#endif // REGRESSIONMODEL_H_
