#ifndef HMM_HPP
#define HMM_HPP
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cstdlib>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <vector>

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

// The index of State_i is (i-1)
inline size_t indexOf(size_t id) { return (id - 1); }

class DataGenerator {
   public:
    DataGenerator(size_t N, size_t M);
    void generate(size_t T, std::vector<std::shared_ptr<State>> &sPtr, std::vector<std::shared_ptr<Observation>> &oPtr);

    const std::vector<std::vector<double>> &A() { return A_; }
    const std::vector<std::vector<double>> &B() { return B_; }
    std::vector<std::shared_ptr<State>> StatePool_;

   private:
    size_t getDataWithGivenProb(const std::vector<double> &prob);

   private:
    std::default_random_engine generator_;
    std::uniform_real_distribution<double> uniform_distribution_;
    std::vector<std::vector<double>> A_;  // Transition matrix
    std::vector<std::vector<double>> B_;  // Emission matrix
};

class Viterbi {
   public:
    using ProbOfFn = std::function<double(size_t i, size_t j)>;
    using FindCandidateStateFn = std::function<std::vector<std::shared_ptr<State>>(std::shared_ptr<Observation> &observationPtr)>;

   public:
    Viterbi(ProbOfFn &A, ProbOfFn &B, FindCandidateStateFn &C) : ProbOfTransition_(A), ProbOfEmission_(B), FindCandidateState_(C) {}
    std::vector<size_t> decode(std::vector<std::shared_ptr<Observation>> &observationPtr, size_t N);

    // online decode
    std::vector<size_t> onlineInitialDecode(std::vector<std::shared_ptr<Observation>> &observationPtr);
    std::vector<size_t> onlineDecode(std::shared_ptr<Observation> &observationPtr);

   private:
    void recursion(std::shared_ptr<Observation> &observationPtr, size_t tIdx);
    void terminationAndBackTracking(std::vector<size_t> &hiddenState);

   private:
    class Nodes {
       public:
        void push(std::shared_ptr<State> &ptr) {
            candidateVec_.emplace_back(Node{std::move(ptr), 0, nullptr});
        }
        size_t size() { return candidateVec_.size(); }
        struct Node {
            std::shared_ptr<State> statePtr_;
            double delta_;
            Node *psiNodePtr_;
        };
        std::vector<Node> candidateVec_;
    };

    FindCandidateStateFn &FindCandidateState_;
    ProbOfFn &ProbOfTransition_;
    ProbOfFn &ProbOfEmission_;
    std::list<Nodes> hmmStates_;
};

#endif  // HMM_HPP
