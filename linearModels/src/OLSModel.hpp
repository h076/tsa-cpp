#ifndef OLS_H_
#define OLS_H_

#include "RegressionModel.hpp"

namespace linModels {

    // Ordinaray Least Squares model

    class OLSModel : public RegressionModel {

        using RegressionModel::RegressionModel;

        // Closed form solution working on X (n, k)
        // n observations
        // p features
        RegressionResult fit() override {
            // Transpose X (n, k) to Xt (k, n)
            auto Xt = xt::transpose(X);
            // Compute X^{t}X with dot
            auto XtX = xt::linalg::dot(Xt, X);
            // Invert
            auto XtXi = xt::linalg::inv(XtX);
            // Compute X^{t}y with dot
            auto Xty = xt::linalg::dot(Xt, y);
            // Calculate coefficients as dot(XtXi, Xty)
            params = xt::linalg::dot(XtXi, Xty);
            // Make predictions
            fittedValues = xt::linalg::dot(X, params);
            // Calculate residuals
            residuals = y - fittedValues;

            // Error metrics
            double rss = xt::sum(xt::square(residuals))();
            size_t n = X.shape(0);
            size_t k = X.shape(1); // can be used for lag used
            double sigma2 = rss / (n - k);

            // Variance-covariance matrix
            auto varBeta = sigma2 * XtXi;

            // T-values
            xt::xarray<double> tValues = params / xt::sqrt(xt::diag(varBeta));

            // AIC and BIC
            double aic = n * std::log(rss / n) + 2 * k;
            double bic = n * std::log(rss / n) + k * std::log(n);

            return {params, fittedValues, residuals, tValues, aic, bic, static_cast<int>(k)};
        }
    };

}

#endif // OLS_H_
