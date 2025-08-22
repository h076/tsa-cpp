#ifndef MODELHELPERS_H_
#define MODELHELPERS_H_

#include "RegressionModel.hpp"
#include "OLSModel.hpp"

namespace linModels {

    enum modelType {
        OLS
    };

    inline std::unique_ptr<RegressionModel> getModelOfType(modelType t, const xt::xtensor<double, 2>& X, const xt::xtensor<double, 1>& y) {
        switch(t) {
            case OLS:
                return std::make_unique<OLSModel>(X, y);
                break;
            default:
                throw std::invalid_argument("modelHelpers::getModelOfType : Invalid model type.");
        }
    }
}

#endif // MODELHELPERS_H_
