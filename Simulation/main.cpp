#include "YugenSimEnv.hpp"

#include <iostream>

int main() {
    YugenSimEnv env(YugenSimEnv::ClassSpec::Mystic);
    env.Reset();
    const auto result = env.Step(static_cast<int>(YugenSimEnv::Action::Attack));
    const auto& state = env.GetState();
    std::cout << "Step: " << state.current_step
              << ", HP: " << state.player.hp
              << ", enemies: " << state.enemies.size()
              << ", reward: " << result.reward << '\n';
}
