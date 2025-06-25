#ifndef TOOLS_H_
#define TOOLS_H_

#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>


namespace tools {

    // lagmat will return a 2d array of lags
    xt::xarray<double> lagmat(xt::xarray<double>& x, int maxlag, std::string trim, std::string original) {
        /*
         * x : xarray<double>, input at most 2D
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

        if (x.dimension() != 1 && x.dimension() != 2) {
            throw std::invalid_argument("lagmat: input must be 1D or 2D");
        }

        // handle 1D arrays
        std::size_t nobs;
        std::size_t nvar;
        if (x.dimension() == 1) {
            nobs = x.shape()[0];
            nvar = 1;
            x = xt::reshape_view(x, std::vector<std::size_t>{nobs, 1});
        }else {
            nobs = x.shape()[0];
            nvar = x.shape()[1];
        }

        // validate maxlag
        if (maxlag < 0)
            throw std::invalid_argument("lagmat: maxlag must be non-negative");

        if (static_cast<std::size_t>(maxlag) >= nobs)
            throw std::invalid_argument("lagmat: maxlag must be < nobs");

        // validate trim
        if (trim != "forward" && trim != "backward" && trim != "both"    && trim != "none")
            throw std::invalid_argument("lagmat: trim must be 'forward', 'backward', 'both', or 'none'");

        // Validate original option
        if (original != "ex" && original != "in")
            throw std::invalid_argument("lagmat: original must be 'ex' or 'in'");

        std::size_t dropidx = (original == "ex") ? nvar : 0;

        // create shape of zero xarray
        std::vector<std::size_t> zerosShape = {nobs + static_cast<std::size_t>(maxlag),
                                               nvar * (static_cast<std::size_t>(maxlag) + 1)};
        // fill new zeros xarray
        xt::xarray<double> lm = xt::zeros<double>(zerosShape);

        // equivelant of :
        // for k in range(0, (maxlag + 1)):
        //     lm [maxlag - k : nobs+maxlag-k, nvar*(maxlag-k) : nvar*(maxlag-k+1)] = x
        for(int k=0; k<maxlag+1; ++k) {
            std::size_t r0 = static_cast<std::size_t>(maxlag) - k;
            std::size_t r1 = nobs + r0;
            std::size_t c0 = (static_cast<std::size_t>(maxlag) - k) * nvar;
            std::size_t c1 = nvar + c0;

            // take block view from zeros array
            auto block = xt::view(lm, xt::range(r0, r1), xt::range(c0, c1));
            block = x;
        }

        std::size_t startobs = (trim == "none" || trim == "forward") ? 0 : static_cast<std::size_t>(maxlag);
        std::size_t stopobs = (trim == "none" || trim == "backward") ? lm.shape()[0] : nobs;

        xt::xarray<double> lags = xt::view(lm, xt::range(startobs, stopobs), xt::range(dropidx, lm.shape()[1]));
        return lags;
    }

    // Prepends/appends columns for constant and/or (linear, quadratic) trend to the design matrix.
    // For example a 2D array of (n, p) will be reshaped to (n, p + k), k being the number of trend values
    xt::xarray<double> addTrend(xt::xarray<double>& x, std::string trend, bool prepend) {
        /*
         *
         * x : 2D array
         *
         * trend : string
         *     * "c" add constant only
         *     * "t" add trend only
         *     * "ct" add constant and linear trend
         *     * "ctt" add constant, linear trend and quadratic trend
         *
         * prepend : bool
         *     If true, prepends the new data to the columns of X.
         *
         */

        // turn trend to lower case
        std::transform(trend.begin(), trend.end(), trend.begin(), ::tolower);

        // Decide how many additional columns
        int order;
        if (trend == "c")
            order = 0;
        else if (trend == "t" || trend == "ct")
            order = 1;
        else if (trend == "ctt")
            order = 2;
        else
            throw std::invalid_argument("Trend '" + trend + "' is not valid");

        const auto& shape = x.shape();
        if (shape.size() != 2)
            throw std::invalid_argument("X must be of 2 dimensions");
        std::size_t nobs = shape[0];

        // build trend array of shape (nobs, order+1), possibly dropping the constant
        xt::xarray<double> trendarr;
        if (order == 0) {
            // constant only
            std::vector<std::size_t> tdrs = {nobs, 1};
            trendarr = xt::ones<double>(tdrs);
        } else if (order == 1) {
            // add time index 1, ...., nobs
            auto t = xt::arange<double>(1.0, double(nobs) + 1.0);
            if (trend == "t") {
                // only trend
                trendarr = t.reshape({nobs, 1});
            }else {
                // constant + trend
                // ones{nobs} is a 1D view, stack into columns
                trendarr = xt::stack(xt::xtuple(xt::ones<double>({nobs}), t), 1);
            }
        } else {
            // constant + linear + quadratic
            auto t = xt::arange<double>(1.0, double(nobs) + 1.0);
            auto t2 = xt::pow(t, 2);
            trendarr = xt::stack(xt::xtuple(xt::ones<double>({nobs}), t, t2), 1);
        }

        // concat along axis=1
        xt::xarray<double> result;
        if (prepend) {
            result = xt::concatenate(xt::xtuple(trendarr, x), 1);
        } else {
            result = xt::concatenate(xt::xtuple(x, trendarr), 1);
        }

        return result;
    }
}

#endif // TOOLS_H_
