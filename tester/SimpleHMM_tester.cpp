#include "hmm.hpp"

int main() {
    const size_t N = 4;  // state number: box
    const size_t M = 2;  // measurement number: red, white

    // Transition Matrix A: NxN
    std::vector<std::vector<double>> A{
        {0, 1, 0, 0},
        {0.4, 0, 0.6, 0},
        {0, 0.4, 0, 0.6},
        {0, 0, 0.5, 0.5}};

    // std::vector<std::vector<double>> A{
    //     {0, 1, 0, 0},
    //     {0, 0, 1, 0},
    //     {0, 0, 0, 1},
    //     {0.8, 0, 0.2, 0}};

    // Emission Matrix B: NxM
    // std::vector<std::vector<double>> B{
    //     {0.5, 0.5},
    //     {0.3, 0.7},
    //     {0.6, 0.4},
    //     {0.8, 0.2}};

    std::vector<std::vector<double>> B{
        {1, 0},
        {1, 0},
        {0.8, 0.2},
        {0, 1}};

    // Initial state prob \pi: 1xN
    std::vector<double> PI{0.25, 0.25, 0.25, 0.25};

    SimpleHMM::Lambda lambda(A, B, PI);
    SimpleHMM hmm;
    auto observation = hmm.generate(lambda, 10);
    // std::cout << "observation: { ";
    // for (auto&& data : observation) {
    //     std::cout << data << " ";
    // }
    // std::cout << "}" << std::endl;

    // std::vector<size_t> test_observation{1, 1, 2, 2, 1};
    // std::cout << "hmm.evaluate = " << hmm.evaluate(test_observation, lambda) << std::endl;

    auto hiddenState = hmm.decode(observation, lambda);
    std::cout << "hiddenState: { ";
    for (auto&& data : hiddenState) {
        std::cout << data << " ";
    }
    std::cout << "}" << std::endl;

    return 0;
}