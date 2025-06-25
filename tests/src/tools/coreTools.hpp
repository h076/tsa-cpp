#ifndef TOOLS_H_
#define TOOLS_H_

#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>

namespace tools {

    // lagmat will return a 2d array of lags
    // Only handling a 1d input array
    xt::xarray<double> lagmat(xt::xarray<double> x, int maxlag, std::string trim, std::string original) {
        /*
         * x : xarray<double>, input only as 1d array
         *
         * maxlag : int, all lags from zero to maxlag are included
         *
         * trim : std::string, the trimming method to use
         *     * 'forward' : trim invalid observations in front.
         *     * 'backward' : trim invalid initial observations.
         *     * 'both' : trim invalid observations on both sides.
         *     * 'none' : no trimming of observations.
         *
         * original : std::string, how the original is treated
         *     * 'ex' : drops the original array returning only the lagged values.
         *     * 'in' : returns the original array and the lagged values as a single
         *              array.
         */

        // should check strings ...


        std::size_t dropidx = 0;

        auto shape = x.shape();
        std::size_t nobs = shape[0]; // num of rows
        std::size_t nvar = shape[1]; // num of columns

        if (original == "ex")
            dropidx = nvar;

        if (maxlag >= nobs) {
            std::cout << "tools::lagmat : maxlag should be < nobs" << std::endl;
            return {};
        }

        // create shape of zero xarray
        std::vector<std::size_t> zerosShape = {nobs, nvar};
        // fill new zeros xarray
        xt::xarray<double> lm = xt::zeros<double>(shape);

        // equivelant of :
        // for k in range(0, (maxlag + 1)):
        //     lm [maxlag - k : nobs+maxlag-k, nvar*(maxlag-k) : nvar*(maxlag-k+1)] = x
        for(int k=0; k<maxlag+1; k++) {
            std::size_t r0 = maxlag - k;
            std::size_t r1 = nobs + maxlag - k;
            std::size_t c0 = nvar * (maxlag - k);
            std::size_t c1 = c0 + nvar;

            // take block view from zeros array
            auto block = xt::view(lm, xt::range(r0, r1), xt::range(c0, c1));
            block = x;
        }

        std::size_t startobs;
        if (trim == "none" || trim == "forward")
            startobs = 0;
        else if (trim == "backward" || trim == "both")
            startobs = maxlag;
        else {
            std::cout << "tools::lagmat : Invalid trim option" << std::endl;
        }

        std::size_t stopobs;
        if (trim == "none" || trim == "backward")
            stopobs = lm.size();
        else
            stopobs = nobs;

        xt::xarray<double> lags = xt::view(lm, xt::range(startobs, stopobs), xt::range(0, dropidx));
        return lags;
    }
}

#endif // TOOLS_H_
