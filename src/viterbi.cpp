#include "viterbi.hpp"

#include <algorithm>

DataGenerator::DataGenerator(size_t N, size_t M) {
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
                A_[i][j] = 0;
            }

            if (A_[i][j] > random_B) {
                A_[i][j] *= (rand() % (100 - 1 + 1) + 1);
            }
            norm_prob += A_[i][j];
        }
        // normalization
        for (size_t j = 0; j < N; j++) {
            if (A_[i][j]) {
                A_[i][j] /= norm_prob;
            }
        }
    }

    std::cout << std::endl
              << "[Transition Matrix]: " << std::endl;
    std::cout << std::setw(7) << ":";
    for (size_t i = 0; i < N; i++) {
        std::cout << std::setw(10) << (i + 1) << ", ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < N; i++) {
        std::cout << std::setw(4) << "A[" << (i + 1) << "]:";
        for (size_t j = 0; j < N; j++) {
            std::cout << std::setw(10) << A_[i][j] << ", ";
        }
        std::cout << std::endl;
    }

    // Initialization of B;
    for (size_t i = 0; i < N; i++) {
        double random_A = 0.45 * rand() / (RAND_MAX + 1.0);
        double random_B = (0.9 - 0.7) * rand() / (RAND_MAX + 1.0) + 0.7;
        double norm_prob = 0;
        for (size_t j = 0; j < M; j++) {
            B_[i][j] = (double)rand() / (RAND_MAX + 1.0);
            if (B_[i][j] < random_A) {
                B_[i][j] = 0;
            }

            if (B_[i][j] > random_B) {
                B_[i][j] *= (rand() % (100 - 1 + 1) + 1);
            }
            norm_prob += B_[i][j];
        }
        // normalization
        for (size_t j = 0; j < M; j++) {
            if (B_[i][j]) {
                B_[i][j] /= norm_prob;
            }
        }
    }

    std::cout << std::endl
              << "[Emission Matrix]: " << std::endl;
    std::cout << std::setw(7) << ":";
    for (size_t i = 0; i < M; i++) {
        std::cout << std::setw(10) << (i + 1) << ", ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < N; i++) {
        std::cout << std::setw(4) << "B[" << (i + 1) << "]:";
        for (size_t j = 0; j < M; j++) {
            std::cout << std::setw(10) << B_[i][j] << ", ";
        }
        std::cout << std::endl;
    }

    // Initialization of StatePool
    StatePool_.reserve(N);
    for (size_t id = 1; id <= N; id++) {
        StatePool_.emplace_back(std::make_shared<State>(id));
    }
}

void DataGenerator::generate(size_t T, std::vector<std::shared_ptr<State>> &sPtr, std::vector<std::shared_ptr<Observation>> &oPtr) {
    std::vector<std::shared_ptr<State>> sPtr_(T);
    std::vector<std::shared_ptr<Observation>> oPtr_(T);
    size_t N = A_.size();
    size_t M = B_.size();

    // data generation basd on A_ and B_
    size_t randomStateId = rand() % (N - 1 + 1) + 1;
    // std::cout << "randomStateId: " << randomStateId << std::endl;
    sPtr_[0] = StatePool_[indexOf(randomStateId)];
    oPtr_[0] = std::make_shared<Observation>(getDataWithGivenProb(B_[indexOf(sPtr_[0]->id_)]));
    // std::cout << "State[" << sPtr_[0]->id_ << "]_" << sPtr_[0].use_count() << std::endl;

    for (size_t i = 1; i < T; i++) {
        size_t prevStateId = sPtr_[i - 1]->id_;
        // std::cout << "prevStateId: " << prevStateId << std::endl;
        size_t generatedId = getDataWithGivenProb(A_[indexOf(prevStateId)]);
        std::shared_ptr<State> statePtr = StatePool_[indexOf(generatedId)];

        sPtr_[i] = statePtr;
        oPtr_[i] = std::make_shared<Observation>(getDataWithGivenProb(B_[indexOf(statePtr->id_)]));
        // std::cout << i << ": "
        //           << "State[" << sPtr_[i]->id_ << "]_" << sPtr_[i].use_count() << std::endl
        //           << std::endl;
    }

    // // check the use_count of StatePool_
    // std::cout << "StatePool_use_count: { ";
    // for (size_t i = 0; i < N; i++) {
    //     std::cout << "State[" << (i + 1) << "]_" << StatePool_[i].use_count() << " ";
    // }
    // std::cout << "}" << std::endl;

    std::swap(sPtr, sPtr_);
    std::swap(oPtr, oPtr_);
}

size_t DataGenerator::getDataWithGivenProb(const std::vector<double> &prob) {
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

std::vector<size_t> Viterbi::decode(std::vector<std::shared_ptr<Observation>> &observationPtr, size_t N) {
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

// online decode
std::vector<size_t> Viterbi::onlineInitialDecode(std::vector<std::shared_ptr<Observation>> &observationPtr) {
    size_t T = observationPtr.size();
    std::vector<size_t> hiddenState(T, 0);

    // add Nodes(candidateState) of each observation
    for (size_t i = 0; i < T; i++) {
        auto candidatesPtr = FindCandidateState_(observationPtr[i]);

        Nodes nodes;
        for (size_t i = 0; i < candidatesPtr.size(); i++) {
            nodes.push(candidatesPtr[i]);
        }
        hmmStates_.emplace_back(std::move(nodes));
    }

    // Initialization hmmStates using first Nodes
    size_t O_1 = observationPtr[0]->id_;
    auto &firstNodes = hmmStates_.front();
    for (size_t i = 0; i < firstNodes.size(); i++) {
        size_t state_Id = firstNodes.candidateVec_[i].statePtr_->id_;
        firstNodes.candidateVec_[i].delta_ = ProbOfEmission_(state_Id, O_1);
    }

    // Recursion
    for (size_t tIdx = 1; tIdx < T; tIdx++) {
        recursion(observationPtr[tIdx], tIdx);
    }

    terminationAndBackTracking(hiddenState);

    return hiddenState;
};

std::vector<size_t> Viterbi::onlineDecode(std::shared_ptr<Observation> &observationPtr) {
    // add Nodes(candidateState) of lastest observation
    auto candidatesPtr = FindCandidateState_(observationPtr);
    Nodes nodes;
    for (size_t i = 0; i < candidatesPtr.size(); i++) {
        nodes.push(candidatesPtr[i]);
    }
    hmmStates_.emplace_back(std::move(nodes));

    size_t T = hmmStates_.size();
    std::vector<size_t> hiddenState(T, 0);

    recursion(observationPtr, T - 1);

    terminationAndBackTracking(hiddenState);
    return hiddenState;
}

void Viterbi::recursion(std::shared_ptr<Observation> &observationPtr, size_t tIdx) {
    size_t O_m = observationPtr->id_;
    auto nodeCurrIter = hmmStates_.begin();
    auto nodePrevIter = hmmStates_.begin();
    std::advance(nodeCurrIter, tIdx);
    std::advance(nodePrevIter, tIdx - 1);

    for (size_t j = 0; j < nodeCurrIter->size(); j++) {
        double maxDelta = 0;
        Nodes::Node *psiNodePtr = nullptr;
        size_t currStateId = nodeCurrIter->candidateVec_[j].statePtr_->id_;
        for (size_t i = 0; i < nodePrevIter->size(); i++) {
            size_t prevStateId = nodePrevIter->candidateVec_[i].statePtr_->id_;
            double tempDelta = nodePrevIter->candidateVec_[i].delta_ * ProbOfTransition_(prevStateId, currStateId);
            if (maxDelta < tempDelta) {
                maxDelta = tempDelta;
                psiNodePtr = &(nodePrevIter->candidateVec_[i]);
            }
        }
        // update psi and delta of candidates of currNode
        nodeCurrIter->candidateVec_[j].delta_ = maxDelta * ProbOfEmission_(currStateId, O_m);
        nodeCurrIter->candidateVec_[j].psiNodePtr_ = psiNodePtr;
    }
}

void Viterbi::terminationAndBackTracking(std::vector<size_t> &hiddenState) {
    size_t T = hiddenState.size();

    // Terminiation
    Nodes::Node *psiNodePtr = nullptr;
    double lastProb = 0;
    auto &lastNode = hmmStates_.back();
    for (size_t i = 0; i < lastNode.size(); i++) {
        auto delta_T = lastNode.candidateVec_[i].delta_;
        if (lastProb < delta_T) {
            lastProb = delta_T;
            hiddenState[indexOf(T)] = lastNode.candidateVec_[i].statePtr_->id_;
            psiNodePtr = lastNode.candidateVec_[i].psiNodePtr_;
        }
    }

    // Path backtracking
    for (int t = T - 2; t >= 0; t--) {
        hiddenState[t] = psiNodePtr->statePtr_->id_;
        psiNodePtr = psiNodePtr->psiNodePtr_;
    }
}
