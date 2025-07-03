#ifndef OLS_H_
#define OLS_H_

#include "RegressionModel.hpp"

namespace models {

    // Ordinaray Least Squares model

    class OLS : public RegressionModel {

        using RegressionModel::RegressionModel;

        // Closed form solution working on X (n, p)
        // n observations
        // p features
        void fit() override {
            // Transpose X (n, p) to Xt (p, n)
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
        }
    };

}

#endif // OLS_H_
