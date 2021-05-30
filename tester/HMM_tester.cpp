#include "viterbi.hpp"

//helper function
template <typename T>
void showVec(std::vector<T>& Vec, const std::string& name) {
    std::cout << name << ":{ ";
    for (auto&& p : Vec) {
        std::cout << std::setw(3) << p << " ";
    }
    std::cout << "} " << std::endl;
}

template <typename T>
void showVecPtr(std::vector<T>& Vec, const std::string& name) {
    std::cout << name << ":{ ";
    for (auto&& p : Vec) {
        std::cout << std::setw(3) << *p << " ";
    }
    std::cout << "} " << std::endl;
}

int main() {
    const size_t N = 10;  // number of state
    const size_t M = 5;   // number of observation
    const size_t T = 20;

    // data initialization
    std::vector<std::shared_ptr<State>> statePtr;
    std::vector<std::shared_ptr<Observation>> observationPtr;
    DataGenerator hmmData(N, M);

    hmmData.generate(T, statePtr, observationPtr);

    std::cout << std::endl
              << "===== Testing Data Generation ====== " << std::endl;
    showVecPtr(statePtr, "HiddenState ");
    showVecPtr(observationPtr, "Observation ");

    // obtain the transition prob from State_i to State_j
    Viterbi::ProbOfFn ProbOfTransition = [&hmmData](size_t i, size_t j) {
        const auto& A = hmmData.A();
        return A[i - 1][j - 1];
    };

    // obtain the emmision prob of State_i given Observation_j
    Viterbi::ProbOfFn ProbOfEmission = [&hmmData](size_t i, size_t j) {
        const auto& B = hmmData.B();
        return B[i - 1][j - 1];
    };

    Viterbi::FindCandidateStateFn FindCandidateStateId = [&hmmData](std::shared_ptr<Observation>& observationPtr) {
        const auto& B = hmmData.B();
        std::vector<std::shared_ptr<State>> candidateState_;
        candidateState_.reserve(B.size());

        // Iterate all states with given observation
        for (size_t i = 0; i < B.size(); i++) {
            size_t OIdx = indexOf(observationPtr->id_);
            if (B[i][OIdx]) {
                size_t StateIdx = i;
                candidateState_.emplace_back(hmmData.StatePool_[StateIdx]);
            }
        }
        return candidateState_;
    };

    std::cout << std::endl
              << "===== Example of offline viterbi ====== " << std::endl;
    Viterbi HmmSolver(ProbOfTransition, ProbOfEmission, FindCandidateStateId);
    auto ans = HmmSolver.decode(observationPtr, N);
    showVec(ans, "offlineAns  ");

    std::cout << std::endl
              << "===== Example of online viterbi ====== " << std::endl;
    size_t middleT = T / 2;
    // size_t middleT = T;
    std::vector<std::shared_ptr<Observation>> PartOfObservationPtr = {observationPtr.begin(), observationPtr.begin() + middleT};
    auto onlineAns = HmmSolver.onlineInitialDecode(PartOfObservationPtr);
    showVecPtr(PartOfObservationPtr, "PartOfObservationPtr ");
    showVec(onlineAns, "onlineAns   ");
    // // check the use_count of StatePool_
    // std::cout << "StatePool_use_count: { ";
    // for (size_t i = 0; i < N; i++) {
    //     std::cout << "State[" << (i + 1) << "]_" << hmmData.StatePool_[i].use_count() << " ";
    // }
    // std::cout << "}" << std::endl;

    for (size_t i = middleT; i < T; i++) {
        auto onlineAns = HmmSolver.onlineDecode(observationPtr[i]);
        showVec(onlineAns, "onlineAns   ");
    }

    return 0;
}