#ifndef REGRESSIONMODEL_H_
#define REGRESSIONMODEL_H_

#include <xtensor/containers/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>

class RegressionModel {

    protected:
        xt::xarray<double> X;
        xt::xarray<double> y;
        xt::xarray<double> params; // model coefficients
        xt::xarray<double> fittedValues; // The predicted values
        xt::xarray<double> residuals; // residual error between true y and predicted

    public:
        RegressionModel(const xt::xarray<double>& x, const xt::xarray<double>& y)
            : X(x), y(y) {};

        virtual void fit() = 0;

        xt::xarray<double> predict(const xt::xarray<double>& x) const {
            return xt::linalg::dot(x, params);
        }

        xt::xarray<double> getParams() const {return params;}
        xt::xarray<double> getFitted() const {return fittedValues;}
        xt::xarray<double> getResiduals() const {return residuals;}
};

#endif // REGRESSIONMODEL_H_
