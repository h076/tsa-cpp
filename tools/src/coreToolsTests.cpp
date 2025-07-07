#include "coreTools.hpp"

void autolagTest();

int main(int argc, char* argv[]) {

    std::cout << "testing all tools" << std::endl;

    autolagTest();

    return 0;
}

void autolagTest() {
    std::cout << "Testing autolag ... " << std::endl;

    xt::xarray<double> X = {{1.0, 0.9, 0.8},
                        {1.0, 0.7, 0.6},
                        {1.0, 0.5, 0.4},
                        {1.0, 0.3, 0.2},
                        {1.0, 0.1, 0.0}};

    xt::xarray<double> y = {1.0, 0.8, 0.6, 0.4, 0.2};

    int startLag = 1;
    int maxLag = 2;

    // Test with AIC method
    auto resultAIC = tools::autoLag(linModels::modelType::OLS, X, y, startLag, maxLag, "aic");
    std::cout << "[AIC] Best IC: " << resultAIC.icbest << ", Best Lag: " << resultAIC.bestLag << std::endl;

    // Test with BIC method
    auto resultBIC = tools::autoLag(linModels::modelType::OLS, X, y, startLag, maxLag, "bic");
    std::cout << "[BIC] Best IC: " << resultBIC.icbest << ", Best Lag: " << resultBIC.bestLag << std::endl;

    // Test with T-Stat method
    auto resultT = tools::autoLag(linModels::modelType::OLS, X, y, startLag, maxLag, "t-stat");
    std::cout << "[T-STAT] T-Value: " << resultT.icbest << ", Best Lag: " << resultT.bestLag << std::endl;
}
