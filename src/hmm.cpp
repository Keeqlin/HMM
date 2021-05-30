#include "hmm.hpp"

inline static size_t IndexOf(size_t state_id) { return (state_id - 1); }

SimpleHMM::SimpleHMM() {
    // set random generator
    // only for generating observations given lambda
    generator_ = std::default_random_engine(time(NULL));
    uniform_distribution_ = std::uniform_real_distribution<double>(0, 1);
}

std::vector<size_t> SimpleHMM::generate(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B, const std::vector<double> &PI, size_t T) {
    std::vector<size_t> observation_result(T, 0);
    std::vector<size_t> hiddenState_result(T, 0);

    // Initialization of first state
    hiddenState_result[0] = getDataWithGivenProb(PI);
    observation_result[0] = getDataWithGivenProb(B[IndexOf(hiddenState_result[0])]);

    for (size_t i = 1; i < T; i++) {
        size_t prev_state = hiddenState_result[i - 1];
        size_t hidden_state = getDataWithGivenProb(A[IndexOf(prev_state)]);

        hiddenState_result[i] = hidden_state;
        observation_result[i] = getDataWithGivenProb(B[IndexOf(hidden_state)]);
    }

    std::cout << "hiddenState: { ";
    for (auto &&data : hiddenState_result) {
        std::cout << data << " ";
    }
    std::cout << "}" << std::endl;

    return observation_result;
}

double SimpleHMM::evaluate(std::vector<size_t> &observations, const Lambda &lambda) {
    // Forward Algorithm
    size_t T = observations.size();
    size_t N = lambda.A_.size();
    std::vector<double> alpha(T, 0);
    std::vector<double> alpha_tmp(T, 0);
    // Initialization
    for (size_t S_n = 0; S_n < N; S_n++) {
        size_t Oidx = IndexOf(observations[0]);
        alpha[S_n] = lambda.PI_[S_n] * lambda.B_[S_n][Oidx];
    }

    // Induction
    for (size_t t = 1; t < T; t++) {
        size_t Oidx = IndexOf(observations[t]);
        for (size_t S_j = 0; S_j < N; S_j++) {
            alpha_tmp[S_j] = 0;
            for (size_t S_i = 0; S_i < N; S_i++) {
                alpha_tmp[S_j] += alpha[S_i] * lambda.A_[S_i][S_j];
            }
            alpha_tmp[S_j] = alpha_tmp[S_j] * lambda.B_[S_j][Oidx];
        }
        std::swap(alpha, alpha_tmp);
    }

    // Termination
    double probOfObservation = 0;
    for (size_t S_n = 0; S_n < N; S_n++) {
        probOfObservation += alpha[S_n];
    }

    return probOfObservation;
}

std::vector<size_t> SimpleHMM::decode(std::vector<size_t> &observations, const Lambda &lambda) {
    size_t T = observations.size();
    size_t N = lambda.A_.size();
    std::vector<std::vector<double>> delta(T, std::vector<double>(N, 0));
    std::vector<std::vector<size_t>> psi(T, std::vector<size_t>(N, 0));
    std::vector<size_t> hiddenState(T, 0);

    // Initialization
    for (size_t S_n = 0; S_n < N; S_n++) {
        size_t Oidx = IndexOf(observations[0]);
        delta[0][S_n] = lambda.PI_[S_n] * lambda.B_[S_n][Oidx];
        psi[0][S_n] = 0;
    }

    // Recursion
    for (size_t t = 1; t < T; t++) {
        size_t Oidx = IndexOf(observations[t]);
        for (size_t S_j = 0; S_j < N; S_j++) {
            double maxDelta = 0;
            size_t maxHiddenState = 0;
            for (size_t S_i = 0; S_i < N; S_i++) {
                double tempDelta = delta[t - 1][S_i] * lambda.A_[S_i][S_j];
                if (maxDelta < tempDelta) {
                    maxDelta = tempDelta;
                    maxHiddenState = S_i + 1;
                }
            }
            delta[t][S_j] = maxDelta * lambda.B_[S_j][Oidx];
            psi[t][S_j] = maxHiddenState;
        }
    }

    // Terminiation
    double lastProb = 0;
    for (size_t S_i = 0; S_i < N; S_i++) {
        if (lastProb < delta[T - 1][S_i]) {
            lastProb = delta[T - 1][S_i];
            hiddenState[T - 1] = (S_i + 1);
        }
    }

    // Path backtracking
    for (int t = T - 2; t >= 0; t--) {
        size_t hiddenStateIdx = IndexOf(hiddenState[t + 1]);
        hiddenState[t] = psi[t + 1][hiddenStateIdx];
    }
    return hiddenState;
}

size_t SimpleHMM::getDataWithGivenProb(const std::vector<double> &prob) {
    // check the validness of prob
    double sumCheck = 0;
    for (auto &&p : prob) {
        sumCheck += p;
    }
    if (std::abs(sumCheck - 1) > 1e-4) {
        throw std::runtime_error("ERROR: SUM OF PROB != 1");
    }

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
