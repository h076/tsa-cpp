#ifndef OLS_H_
#define OLS_H_

#include "RegressionModel.hpp"

#include <xtensor/core/xnoalias.hpp>
#include <xtensor/containers/xadapt.hpp>


namespace linModels {

    // Ordinaray Least Squares model

    class OLSModel : public RegressionModel {

        public:

            using RegressionModel::RegressionModel;

            // Closed form solution working on X (n, k)
            // n observations
            // p features
            inline RegressionResult fit() override {
                // Transpose X (n, k) to Xt (k, n)
                xt::xtensor<double, 2> Xt = xt::transpose(X);
                // Compute X^{t}X with dot
                xt::xtensor<double, 2> XtX = xt::linalg::dot(Xt, X);

                // Before inversion must check that XtX is not singular
                // This is done using the condition number
                auto svd = xt::linalg::svd(XtX);
                auto S = std::get<1>(svd);
                double cond = S(0) / S(S.shape(0) - 1); // largest / smallest singular value
                xt::xtensor<double, 2> XtXi;
                if (cond > 1e12) {
                    // matrix is near singular so must use Moore-Rose pseudo inverse
                    xt::noalias(XtXi) = xt::linalg::pinv(XtX);
                } else {
                    xt::noalias(XtXi) = xt::linalg::inv(XtX);
                }

                // Compute X^{t}y with dot
                xt::xtensor<double, 1> Xty = xt::linalg::dot(Xt, y);
                // Calculate coefficients as dot(XtXi, Xty)
                xt::noalias(params) = xt::linalg::dot(XtXi, Xty);
                // Make predictions
                xt::noalias(fittedValues) = xt::linalg::dot(X, params);
                // Calculate residuals
                xt::noalias(residuals) = y - fittedValues;

                // Error metrics
                double rss = xt::sum(xt::square(residuals))();
                size_t n = X.shape(0);
                size_t k = X.shape(1); // can be used for lag used
                double sigma2 = rss / (n - k);

                // Variance-covariance matrix
                auto varBeta = sigma2 * XtXi;

                // T-values
                xt::noalias(tValues) = params / xt::sqrt(xt::diagonal(varBeta));

                // AIC and BIC
                aic = n * std::log(rss / n) + 2 * k;
                bic = n * std::log(rss / n) + k * std::log(n);

                lag = static_cast<int>(k);

                return {params, fittedValues, residuals, tValues, aic, bic, lag};
            }
    };

}

#endif // OLS_H_
