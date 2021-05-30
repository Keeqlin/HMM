#ifndef HMM_HPP
#define HMM_HPP
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <vector>

class SimpleHMM {
   public:
    struct Lambda {
        Lambda(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B, std::vector<double> &PI) : A_(A), B_(B), PI_(PI) {}
        std::vector<std::vector<double>> &A_;  // Transition matrix
        std::vector<std::vector<double>> &B_;  // Emission matrix
        std::vector<double> &PI_;              // Initial probability
    };

   public:
    SimpleHMM();
    std::vector<size_t> generate(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B, const std::vector<double> &PI, size_t T);
    inline std::vector<size_t> generate(const Lambda &lambda, size_t T) {
        return generate(lambda.A_, lambda.B_, lambda.PI_, T);
    }
    double evaluate(std::vector<size_t> &observations, const Lambda &lambda);
    std::vector<size_t> decode(std::vector<size_t> &observations, const Lambda &lambda);

    template <typename T>
    void VecDisplay(const std::vector<T> &vec, const std::string &vecName) {
    }

   private:
    size_t getDataWithGivenProb(const std::vector<double> &prob);
    std::default_random_engine generator_;
    std::uniform_real_distribution<double> uniform_distribution_;
};

class State {
   public:
    State() {}
    explicit State(size_t id) : id_(id) {}
    size_t id_;
    friend std::ostream &operator<<(std::ostream &out, const State &rhs) {
        out << rhs.id_;
        return out;
    }
};

class Observation {
   public:
    Observation() {}
    explicit Observation(size_t id) : id_(id) {}
    size_t id_;
    friend std::ostream &operator<<(std::ostream &out, const Observation &rhs) {
        out << rhs.id_;
        return out;
    }
};

inline size_t indexOf(size_t id) { return (id - 1); }

class DataGenerator {
   public:
    DataGenerator(size_t N, size_t M) {
        generator_ = std::default_random_engine(time(NULL));
        uniform_distribution_ = std::uniform_real_distribution<double>(0, 1);
        A_ = std::vector<std::vector<double>>(N, std::vector<double>(N, 0));
        B_ = std::vector<std::vector<double>>(N, std::vector<double>(M, 0));

        // srand(time(NULL));
        srand(0);

        // Initialization of A;
        for (size_t i = 0; i < N; i++) {
            double random_A = 0.4 * rand() / (RAND_MAX + 1.0);
            double random_B = (0.9 - 0.6) * rand() / (RAND_MAX + 1.0) + 0.6;
            double norm_prob = 0;
            for (size_t j = 0; j < N; j++) {
                A_[i][j] = ((double)rand() / (RAND_MAX + 1.0));
                if (A_[i][j] < random_A) {
                    // if (A_[i][j]) {
                    //     A_[i][j] /= (rand() % (10 - 0 + 1) + 0);
                    // }
                    // A_[i][j] = ((double)rand() / (RAND_MAX + 1.0));
                    A_[i][j] = 0;
                }

                if (A_[i][j] > random_B) {
                    A_[i][j] *= (rand() % (10 - 0 + 1) + 0);
                }
                norm_prob += A_[i][j];
            }
            // normalization
            std::cout << "A[" << i << "]: ";
            for (size_t j = 0; j < N; j++) {
                if (A_[i][j]) {
                    A_[i][j] /= norm_prob;
                }
                std::cout << A_[i][j] << ", ";
            }
            std::cout << std::endl;
        }

        // Initialization of B;
        for (size_t i = 0; i < N; i++) {
            double random_A = 0.4 * rand() / (RAND_MAX + 1.0);
            double random_B = (0.9 - 0.75) * rand() / (RAND_MAX + 1.0) + 0.75;
            double norm_prob = 0;
            for (size_t j = 0; j < M; j++) {
                B_[i][j] = (double)rand() / (RAND_MAX + 1.0);
                if (B_[i][j] < random_A) {
                    // if (B_[i][j]) {
                    //     B_[i][j] /= (rand() % (10 - 0 + 1) + 0);
                    // }
                    B_[i][j] = 0;
                }

                if (B_[i][j] > random_B) {
                    B_[i][j] *= (rand() % (10 - 0 + 1) + 0);
                }
                norm_prob += B_[i][j];
            }
            // normalization
            std::cout << "B[" << i << "]: ";
            for (size_t j = 0; j < M; j++) {
                if (B_[i][j]) {
                    B_[i][j] /= norm_prob;
                }
                std::cout << B_[i][j] << ", ";
            }
            std::cout << std::endl;
        }
    }

    void generate(size_t T, std::vector<std::shared_ptr<State>> &sPtr, std::vector<std::shared_ptr<Observation>> &oPtr) {
        std::vector<std::shared_ptr<State>> sPtr_(T);
        std::vector<std::shared_ptr<Observation>> oPtr_(T);
        size_t N = A_.size();
        size_t M = B_.size();

        // data generation basd on A_ and B_
        sPtr_[0] = std::make_shared<State>(rand() % (N - 1 + 1) + 1);
        oPtr_[0] = std::make_shared<Observation>(getDataWithGivenProb(B_[indexOf(sPtr_[0]->id_)]));

        for (size_t i = 1; i < T; i++) {
            size_t prevStateId = sPtr_[i - 1]->id_;
            std::shared_ptr<State> statePtr = std::make_shared<State>(getDataWithGivenProb(A_[indexOf(prevStateId)]));

            sPtr_[i] = statePtr;
            oPtr_[i] = std::make_shared<Observation>(getDataWithGivenProb(B_[indexOf(statePtr->id_)]));
        }

        std::swap(sPtr, sPtr_);
        std::swap(oPtr, oPtr_);
    }

    const std::vector<std::vector<double>> &A() { return A_; }
    const std::vector<std::vector<double>> &B() { return B_; }

   private:
    size_t getDataWithGivenProb(const std::vector<double> &prob) {
        double rand_data = uniform_distribution_(generator_);
        double prob_accum = 0;
        size_t data = 0;
        for (size_t i = 0; i < prob.size(); i++) {
            prob_accum += prob[i];
            if (rand_data <= prob_accum) {
                // The range of state is [1, sizeof(prob)]
                data = (i + 1);
                break;
            }
        }
        return data;
    }

   private:
    std::default_random_engine generator_;
    std::uniform_real_distribution<double> uniform_distribution_;
    std::vector<std::vector<double>> A_;  // Transition matrix
    std::vector<std::vector<double>> B_;  // Emission matrix
};

class Viterbi {
   public:
    using ProbOfFn = std::function<double(size_t i, size_t j)>;
    Viterbi(ProbOfFn &A, ProbOfFn &B) : ProbOfTransition_(A), ProbOfEmission_(B) {}

    std::vector<size_t> decode(std::vector<std::shared_ptr<Observation>> &observationPtr, size_t N) {
        size_t T = observationPtr.size();
        std::vector<std::vector<double>> delta(T, std::vector<double>(N, 0));
        std::vector<std::vector<size_t>> psi(T, std::vector<size_t>(N, 0));
        std::vector<size_t> hiddenState(T, 0);

        // Initialization
        for (size_t S_n = 1; S_n <= N; S_n++) {
            size_t O_m = observationPtr[0]->id_;
            delta[0][indexOf(S_n)] = ProbOfEmission_(S_n, O_m);
            psi[0][indexOf(S_n)] = 0;
        }

        // Recursion
        for (size_t t = 1; t < T; t++) {
            size_t O_m = observationPtr[t]->id_;
            for (size_t S_j = 1; S_j <= N; S_j++) {
                double maxDelta = 0;
                size_t maxHiddenState = 0;
                for (size_t S_i = 1; S_i <= N; S_i++) {
                    double tempDelta = delta[t - 1][indexOf(S_i)] * ProbOfTransition_(S_i, S_j);
                    if (maxDelta < tempDelta) {
                        maxDelta = tempDelta;
                        maxHiddenState = S_i;
                    }
                }
                delta[t][indexOf(S_j)] = maxDelta * ProbOfEmission_(S_j, O_m);
                psi[t][indexOf(S_j)] = maxHiddenState;
            }
        }

        // Terminiation
        double lastProb = 0;
        for (size_t S_i = 1; S_i <= N; S_i++) {
            if (lastProb < delta[T - 1][indexOf(S_i)]) {
                lastProb = delta[T - 1][indexOf(S_i)];
                hiddenState[T - 1] = S_i;
            }
        }

        // Path backtracking
        for (int t = T - 2; t >= 0; t--) {
            size_t hiddenStateIdx = indexOf(hiddenState[t + 1]);
            hiddenState[t] = psi[t + 1][hiddenStateIdx];
        }

        return hiddenState;
    }

    // std::vector<size_t> onlineInitialDecode(std::vector<std::shared_ptr<Observation>> &observationPtr) {
    // }

   private:
    class CandidateState {
       public:
        void push(std::shared_ptr<State> &&ptr) {
            NodeList_.emplace_back(std::move(ptr));
        }
        size_t size() { return NodeList_.size(); }
        std::list<std::shared_ptr<State>> NodeList_;
    };

    ProbOfFn &ProbOfTransition_;
    ProbOfFn &ProbOfEmission_;
    std::list<std::shared_ptr<CandidateState>> hmmStates_;
};

#endif  // HMM_HPP
