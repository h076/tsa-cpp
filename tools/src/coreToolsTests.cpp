#include "coreTools.hpp"

void autolagTest();
void addTrendTest();
void lagmatTest();

int main(int argc, char* argv[]) {

    std::cout << "testing all tools" << std::endl;

    autolagTest();
    //addTrendTest();
    //lagmatTest();

    return 0;
}

void autolagTest() {
    std::cout << "Testing autolag ... " << std::endl;

    xt::xtensor<double, 2> X = {{1.0, 0.9, 0.8},
                        {1.0, 0.7, 0.6},
                        {1.0, 0.5, 0.4},
                        {1.0, 0.3, 0.2},
                        {1.0, 0.1, 0.0}};

    xt::xtensor<double, 1> y = {1.0, 0.8, 0.6, 0.4, 0.2};

    int startLag = 1;
    int maxLag = 3;

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

void addTrendTest() {
    std::cout << "Testing addTrend ..." << std::endl;

    xt::xtensor<double, 2> x = xt::expand_dims(xt::arange(0., 6.), 1);

    std::cout << "X : " << x << std::endl;

    xt::xtensor<double, 2> xt = tools::addTrend(x, "ctt", true);

    std::cout << "Xt : " << xt << std::endl;
}

void lagmatTest() {
    std::cout << "Testing lagmat ... " << std::endl;

    xt::xtensor<double, 1> x = xt::arange(1., 7.);

    std::cout << "X : " << x << std::endl;

    xt::xtensor<double, 2> xl = tools::lagmat(x, 2, "both", "in");

    std::cout << "Xl : " << xl << std::endl;
}
