#ifndef ROLLING_H_
#define ROLLING_H_

#include <deque>
#include <xtensor/containers/xtensor.hpp>

#include "autoReg.hpp"

namespace tools
{

    namespace rolling
    {

        template <typename T>
        class Rolling {

            public:

                Rolling(T initial, xt::xtensor<T, 1> window) : m_val(initial), m_w(window.begin(), window.end()), m_ws(window.size()) {}

                virtual T update(T next) = 0;

                T getCurr() {return m_val;}

            protected:
                T m_val;
                std::deque<T> m_w;
                int m_ws;
        };

        class Mean : public Rolling<double> {

            public:

                Mean(double initial, xt::xtensor<double, 1> window) : Rolling(initial, window) {
                    m_sum = xt::sum(window)();
                    m_val = m_sum / static_cast<double>(m_ws);
                }

                double update(double next) override {
                    m_sum -= m_w.front();
                    m_w.pop_front();

                    m_sum += next;
                    m_w.push_back(next);

                    m_val = m_sum / static_cast<double>(m_ws);

                    return m_val;
                }

            private:

                double m_sum;

        };

        class HalfLife : public Rolling<double> {

            public:

                HalfLife(double initial, xt::xtensor<double, 1> window) : Rolling(initial, window) {

                }

                // Must be a more efficient way than using OLS each time ...
                double update(double next) override {
                    m_w.pop_front();
                    m_w.push_back(next);

                    m_val = tools::AROneHalfLife(xt::adapt(m_w));

                    return m_val;
                }
             
        };

    }
}

#endif // ROLLING_H_
