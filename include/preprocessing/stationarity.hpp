#ifndef STATIONARITY_H_
#define STATIONARITY_H_

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xadapt.hpp>

namespace preprocessing {

  xt::xtensor<double, 1> differencing(xt::xtensor<double, 1> exog) {
      return xt::diff(exog, 1, 0);
  }

}

#endif // STATIONARITY_H_
