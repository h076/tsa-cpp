#ifndef MODELHELPERS_H_
#define MODELHELPERS_H_

#include "RegressionModel.hpp"
#include "OLSModel.hpp"

namespace linModels {

    enum modelType {
        OLS
    };

    RegressionModel * getModelOfType(modelType t, const xt::xarray<double>& X, const xt::xarray<double>& y) {
        switch(t) {
            case OLS:
                return new OLSModel(X, y);
                break;
            default:
                throw std::invalid_argument("modelHelpers::getModelOfType : Invalid model type.");
        }
    }
}

#endif // MODELHELPERS_H_
