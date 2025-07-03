#include <iostream>

#include "coreTools.hpp"

int main(int argc, char *argv[]) {

    xt::xarray<double> x = xt::arange<double>(1, 6).reshape({5, 1});

    std::cout << x << std::endl;

    //xt::xarray<double> y = tools::lagmat(x, 5, "both", "in");

    //std::cout << y << std::endl;

    xt::xarray<double> z = tools::addTrend(x, "ctt", true);

    std::cout << z << std::endl;

    return 0;
}
