#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // Just create a placeholder to remind us to look at the model
    std::cout << "Model output shape: 1x84x8400" << std::endl;
    std::cout << "84 channels - maybe 4 (box) + 80 (cls+obj)?" << std::endl;
    std::cout << "Or some variants have different output layouts?" << std::endl;
    return 0;
}
