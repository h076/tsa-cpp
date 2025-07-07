#ifndef REGRESSIONMODEL_H_
#define REGRESSIONMODEL_H_

#include <xtensor/containers/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>

namespace linModels {

    struct RegressionResult {
        xt::xarray<double> params;
        xt::xarray<double> fittedValues;
        xt::xarray<double> residuals;
        xt::xarray<double> tValues;

        double aic = 0.0;
        double bic = 0.0;

        int lag = -1;
    };

    class RegressionModel {

        protected:
            xt::xarray<double> X;
            xt::xarray<double> y;
            xt::xarray<double> params; // model coefficients
            xt::xarray<double> fittedValues; // The predicted values
            xt::xarray<double> residuals; // residual error between true y and predicted
            xt::xarray<double> tValues; // how significantly different coeffiecients are from zero

            double aic; // Akaike information criterion
            double bic; // Bayesian information criterion

            int lag; // Lag length used

        public:
            RegressionModel(const xt::xarray<double>& x, const xt::xarray<double>& y)
                : X(x), y(y) {};

            virtual ~RegressionModel() = default;

            virtual RegressionResult fit() = 0;

            xt::xarray<double> getParams() const {return params;}
            xt::xarray<double> getFitted() const {return fittedValues;}
            xt::xarray<double> getResiduals() const {return residuals;}
    };

}

#endif // REGRESSIONMODEL_H_
