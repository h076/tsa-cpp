#include <iostream>

#include "coreTools.hpp"

int main(int argc, char *argv[]) {

    xt::xarray<double> x = {1., 2., 3., 4., 5., 6.};

    std::cout << x << std::endl;

    xt::xarray<double> y = tools::lagmat(x, 5, "both", "in");

    std::cout << y << std::endl;

    return 0;
}
