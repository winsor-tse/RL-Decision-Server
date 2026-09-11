#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Headless C++17 scaffold. Stats and spawn positions are illustrative,
// not verified Yugen Saga rules. Each Step represents one fixed simulation tick.
class YugenSimEnv {
public:
    static constexpr int MapWidth = 32;
    static constexpr int MapHeight = 32;
    static constexpr std::size_t SkillCount = 8; // Attack + spell slots 1..7.
    using StepIndex = std::uint64_t;

    enum class ClassSpec { Mystic, Knave, Swordsman };
    enum class Tile : std::uint8_t { Walkable, Wall };
    enum class AggroTarget : std::int8_t { None = -1, Player = 0 };

    // Matches Custom_enviornments/Test_Env/Env_16.py (spell 4 is omitted).
    enum class Action : int {
        Up, Down, Left, Right, Attack,
        Spell1, Spell2, Spell3, Spell5, Spell6, Spell7, Count
    };

    struct Player {
        int x = 16;
        int y = 16;
        int hp = 100;
        int max_hp = 100;
        int mp = 50;
        int max_mp = 50;
        ClassSpec class_spec = ClassSpec::Mystic;
    };

    struct Enemy {
        int x = 0;
        int y = 0;
        int hp = 30;
        AggroTarget aggro_target = AggroTarget::None;
    };

    struct Cooldown {
        StepIndex duration_steps = 0;
        StepIndex ready_at_step = 0;

        bool IsReady(StepIndex now) const noexcept {
            return now >= ready_at_step;
        }
        StepIndex Remaining(StepIndex now) const noexcept {
            return IsReady(now) ? 0 : ready_at_step - now;
        }
        // When a cast succeeds at step t, set ready_at_step = t + duration_steps.
        // No wall-clock time or per-tick decrement is needed.
    };

    struct State {
        // Row-major indexing: tiles[y * MapWidth + x].
        std::array<Tile, MapWidth * MapHeight> tiles{};
        Player player{};
        std::vector<Enemy> enemies;
        std::array<Cooldown, SkillCount> cooldowns{};
        StepIndex current_step = 0;
    };

    struct StepResult {
        float reward = 0.0F;
        bool terminated = false;
        bool truncated = false;
    };

    explicit YugenSimEnv(ClassSpec spec = ClassSpec::Mystic)
        : class_spec_(spec) {
        state_.enemies.reserve(5);
        Reset();
    }

    // Deterministic reset; vector capacity is retained across episodes.
    const State& Reset() {
        state_.current_step = 0;
        state_.tiles.fill(Tile::Walkable);
        for (int y = 0; y < MapHeight; ++y) {
            for (int x = 0; x < MapWidth; ++x) {
                if (x == 0 || y == 0 || x == MapWidth - 1 || y == MapHeight - 1) {
                    state_.tiles[y * MapWidth + x] = Tile::Wall;
                }
            }
        }
        state_.player = Player{};
        state_.player.class_spec = class_spec_;
        state_.enemies.clear();
        state_.enemies.push_back(Enemy{20, 16, 30, AggroTarget::None});
        state_.enemies.push_back(Enemy{12, 10, 30, AggroTarget::None});

        // Placeholder durations: index 0 = attack, indices 1..7 = spell slots.
        constexpr std::array<StepIndex, SkillCount> durations{2, 5, 8, 10, 12, 15, 20, 25};
        for (std::size_t i = 0; i < SkillCount; ++i) {
            state_.cooldowns[i] = Cooldown{durations[i], 0};
        }
        return state_;
    }

    StepResult Step(int action) {
        if (action < 0 || action >= static_cast<int>(Action::Count)) {
            throw std::out_of_range("YugenSimEnv action must be in [0, 10]");
        }
        // TODO: apply action at current_step (movement, collision, skill checks).
        // TODO: update enemy aggro, movement, damage, regeneration, and deaths.
        // TODO: compute reward, termination, and episode time-limit truncation.
        // STUB: every valid action currently only advances the simulation clock.
        ++state_.current_step;
        return {};
    }

    const State& GetState() const noexcept { return state_; }

    bool IsWalkable(int x, int y) const noexcept {
        return x >= 0 && x < MapWidth && y >= 0 && y < MapHeight
            && state_.tiles[y * MapWidth + x] == Tile::Walkable;
    }

private:
    ClassSpec class_spec_;
    State state_;
};
