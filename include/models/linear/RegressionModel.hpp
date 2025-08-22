#ifndef REGRESSIONMODEL_H_
#define REGRESSIONMODEL_H_

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>

namespace linModels {

    struct RegressionResult {
        xt::xtensor<double, 1> params;
        xt::xtensor<double, 1> fittedValues;
        xt::xtensor<double, 1> residuals;
        xt::xtensor<double, 1> tValues;

        double aic = 0.0;
        double bic = 0.0;

        int lag = -1;
    };

    class RegressionModel {

        protected:
            xt::xtensor<double, 2> X;
            xt::xtensor<double, 1> y;
            xt::xtensor<double, 1> params; // model coefficients
            xt::xtensor<double, 1> fittedValues; // The predicted values
            xt::xtensor<double, 1> residuals; // residual error between true y and predicted
            xt::xtensor<double, 1> tValues; // how significantly different coeffiecients are from zero

            double aic; // Akaike information criterion
            double bic; // Bayesian information criterion

            int lag; // Lag length used

        public:
            // x passed from xt::view which gives expression
            template <typename EXPR>
            RegressionModel(const EXPR& x, const xt::xtensor<double, 1>& y)
                : X(x), y(y) {};

            virtual ~RegressionModel() = default;

            virtual RegressionResult fit() = 0;

            xt::xtensor<double, 1> getParams() const {return params;}
            xt::xtensor<double, 1> getFitted() const {return fittedValues;}
            xt::xtensor<double, 1> getResiduals() const {return residuals;}
    };

}

#endif // REGRESSIONMODEL_H_
