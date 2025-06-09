
#include "MLP.hpp"
#include "utils_data.hpp"

int main() {

    auto training_tuple = get_mmist_processed_data("train");
    auto& training_x = std::get<0>(training_tuple);
    auto& training_y = std::get<1>(training_tuple);

    auto testing_tuple = get_mmist_processed_data("test");
    auto& testing_x = std::get<0>(testing_tuple);
    auto& testing_y = std::get<1>(testing_tuple);

    Hyperparameters h;
    h.layers = { 784, 32, 16, 10 };
    h.epochs = 25;
    h.initializer = Initializer::HE;
    h.learning_rate = 0.001;
    h.optimizer = Optimizer::NONE; //Optimizer::ADAM, Optimizer::RMS
    h.shuffle = true;
    h.debug = true;
    h.batch = 16;

    MLP<ReluFunctor, DReluFunctor> mlp(h);

    mlp.train(training_x, training_y, testing_x, testing_y);
    mlp.save_weights("NONE");

    return 0;
}
