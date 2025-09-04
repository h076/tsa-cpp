#ifndef KELLYCRITERION_H_
#define KELLYCRITERION_H_

#include <stdexcept>

namespace sizing {

    class Kelly {

        public:

            Kelly() : m_dWinSum(0.0), m_dLossSum(0.0), m_nWins(0), m_nLosses(0) {}

            void recordWin(double profit) {
                if (profit <= 0.0)
                    throw std::invalid_argument("Win must have positive profit");
                ++m_nWins;
                m_dWinSum += profit;;
            }

            void recordLoss(double loss) {
                if (loss <= 0.0)
                    throw std::invalid_argument("Loss must be positive");
                ++m_nLosses;
                m_dLossSum += loss;
            }

            double getKelly() {
                int total = m_nWins + m_nLosses;
                if (total == 0 || m_nWins == 0 || m_nLosses == 0)
                    return 0.0; // not enough data

                double W = static_cast<double>(m_nWins) / total;
                double avgWin = m_dWinSum / m_nWins;
                double avgLoss = m_dLossSum / m_nLosses;
                double R = avgWin / avgLoss;

                return W - ((1.0 - W) / R);
            }

        private:

            double m_dWinSum;
            double m_dLossSum;

            int m_nWins;
            int m_nLosses;
    };
}

#endif // KELLYCRITERION_H_
