"""Immutable configuration for the map53_open_entities_v1 baseline."""
from dataclasses import dataclass, field
from math import isfinite


def integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def interval(name, bounds, maximum):
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError(f"{name} must be an immutable (min, max) tuple")
    for value in bounds:
        integer(name, value)
    if not bounds[0] <= bounds[1] < maximum:
        raise ValueError(f"{name} must be ordered and inside 0..{maximum - 1}")


@dataclass(frozen=True, slots=True)
class PlayerConfig:
    level: int = 135
    max_hp: int = 9463
    max_mp: int = 15108
    ac: int = 3292
    strength: int = 100
    stamina: int = 100
    intelligence: int = 578
    wisdom: int = 578
    dexterity: int = 194
    ferocity: int = 11
    acuity: int = 22
    toughness: int = 7
    precision: int = 18
    class_spec: str = "Mystic"
    base_hp: int | None = None
    base_mp: int | None = None
    # Per-second rates: restore 226 HP / 1664 MP at the default 2000 ms interval.
    hp_regen_override: float | None = 113.0
    mp_regen_override: float | None = 832.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name in ("hp_regen_override", "mp_regen_override"):
                value = getattr(self, name)
                if value is not None and (not isfinite(value) or value < 0):
                    raise ValueError(f"{name} must be a nonnegative finite per-second rate")
            elif name in ("base_hp", "base_mp"):
                if getattr(self, name) is not None:
                    integer(name, getattr(self, name))
            elif name != "class_spec":
                integer(name, getattr(self, name), 1 if name in ("level", "max_hp", "max_mp") else 0)
        for resource in ("hp", "mp"):
            if getattr(self, resource + "_regen_override") is None and getattr(self, "base_" + resource) is None:
                raise ValueError(f"{resource} regeneration needs base stats or an explicit override")
        if self.class_spec != "Mystic":
            raise ValueError("This profile supports class_spec=Mystic only")


@dataclass(frozen=True, slots=True)
class InnieConfig:
    template_id: int = 5300
    effective_level: int = 150
    bulk_factor: float = 0.5
    max_hp: int = 274599
    max_mp: int = 0
    ac: int = 2250
    toughness: int = 15
    raw_damage: int = 3549
    attack_ms: int = 1000
    attack_radius: float = 1.0
    move_ms: tuple[int, int] = (900, 1100)
    respawn_ms: int = 50000
    aggro_radius: float = 4.0
    aggro_check_ms: int = 1500
    aggro_drop_distance: float = 18.0

    def __post_init__(self):
        for name in ("template_id", "effective_level", "max_hp", "attack_ms", "respawn_ms", "aggro_check_ms"):
            integer(name, getattr(self, name), 1)
        for name in ("max_mp", "ac", "toughness", "raw_damage"):
            integer(name, getattr(self, name))
        interval("move_ms", self.move_ms, 2**31)
        if self.move_ms[0] == 0:
            raise ValueError("move_ms must be positive")
        for name in ("bulk_factor", "attack_radius", "aggro_radius", "aggro_drop_distance"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class SpellConfig:
    spell_id: int
    slot: int
    cooldown_ms: int
    mp_cost: int
    mana_consumption: float
    mana_factor: float
    damage_percent: float
    target_mode: str
    family: str | None = None
    family_cooldown_ms: int = 0
    duration_ms: int = 0
    tick_interval_ms: int = 0
    initial_damage_percent: float = 1.0
    tick_percent: float = 0.0
    min_radius: float = 0.0
    max_radius: float = 0.0
    level: int = 0
    cast_level: int = 0
    effect_id: str | None = None
    effect_level: int = 0
    hp_cost: int = 0
    vita_consumption: float = 0.0
    base_damage: float = 1126.7

    def __post_init__(self):
        for name in ("spell_id", "slot", "cooldown_ms"):
            integer(name, getattr(self, name), 1)
        for name in ("mp_cost", "family_cooldown_ms", "duration_ms", "tick_interval_ms",
                     "level", "cast_level", "effect_level", "hp_cost"):
            integer(name, getattr(self, name))
        for name in ("mana_consumption", "mana_factor", "damage_percent", "initial_damage_percent", "tick_percent", "vita_consumption"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not (isfinite(self.min_radius) and isfinite(self.max_radius)
                and 0 <= self.min_radius <= self.max_radius):
            raise ValueError("spell radii must be finite, nonnegative, and ordered")
        if bool(self.duration_ms) != bool(self.tick_interval_ms):
            raise ValueError("Timed spells require both duration and tick interval")
        if self.duration_ms and self.tick_interval_ms > self.duration_ms:
            raise ValueError("tick interval cannot exceed duration")
        if self.target_mode not in ("targeted", "targetedOrSelf"):
            raise ValueError("Unsupported target mode")
        if bool(self.family) != bool(self.family_cooldown_ms):
            raise ValueError("Family and family cooldown must be specified together")
        if not isfinite(self.base_damage) or self.base_damage < 0:
            raise ValueError("base_damage must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class SpellRules:
    global_cooldown_ms: int = 300
    arcane_bomb: bool = False
    sunburnt: bool = False
    tempest_meteor: bool = False

    def __post_init__(self):
        integer("global_cooldown_ms", self.global_cooldown_ms)
        if any(getattr(self, name) is not False for name in
               ("arcane_bomb", "sunburnt", "tempest_meteor")):
            raise ValueError("Trinket/status modifiers are disabled in the baseline")


def baseline_spells():
    return (
        SpellConfig(416, 1, 3500, 500, .33, .45, .75, "targeted",
                    family="arcane-blast", family_cooldown_ms=3500),
        SpellConfig(417, 2, 5000, 1000, .33, .275, .75, "targetedOrSelf",
                    duration_ms=6000, tick_interval_ms=1000, initial_damage_percent=.76,
                    tick_percent=.04, min_radius=2.5, max_radius=4.25, level=70,
                    effect_id="acid", effect_level=2),
        SpellConfig(418, 3, 1750, 750, .20, .45, .775, "targetedOrSelf",
                    min_radius=.5, max_radius=1.5, level=125, cast_level=134),
    )


@dataclass(frozen=True, slots=True)
class TimingConfig:
    step_ms: int = 200
    regen_ms: int = 2000
    full_box_retry_ms: int = 200
    regen_enabled: bool = True

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name == "regen_enabled":
                if type(self.regen_enabled) is not bool:
                    raise ValueError("regen_enabled must be boolean")
            else:
                integer(name, getattr(self, name), 1)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    profile: str = "combat_reward_v1"
    win_kills: int = 5
    max_episode_steps: int = 256
    enemy_damage_weight: float = 1.0
    kill_bonus: float = 1.0
    player_damage_weight: float = 1.0
    death_penalty: float = 5.0
    time_cost: float = 0.001
    y_bounds: tuple[int, int] | None = None
    legacy_y_penalty_below: int | None = None

    def __post_init__(self):
        integer("win_kills", self.win_kills, 1)
        integer("max_episode_steps", self.max_episode_steps, 1)
        if self.profile not in ("combat_reward_v1", "legacy_reward_v0"):
            raise ValueError("Unknown reward profile")
        for name in ("enemy_damage_weight", "kill_bonus", "player_damage_weight", "death_penalty", "time_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.y_bounds is not None:
            interval("y_bounds", self.y_bounds, 100)
        if self.legacy_y_penalty_below is not None:
            integer("legacy_y_penalty_below", self.legacy_y_penalty_below)
            if self.legacy_y_penalty_below >= 100:
                raise ValueError("legacy_y_penalty_below must be within the map")


@dataclass(frozen=True, slots=True)
class GearConfig:
    weapon_damage: int
    attack_ms: int
    spell_multiplier: float = 0.0

    def __post_init__(self):
        integer("weapon_damage", self.weapon_damage, 1)
        integer("attack_ms", self.attack_ms, 1)
        if not isfinite(self.spell_multiplier) or self.spell_multiplier < 0:
            raise ValueError("spell_multiplier must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    profile: str = "map53_open_entities_v1"
    map_id: int = 53
    width: int = 100
    height: int = 100
    player_spawn_x: tuple[int, int] = (45, 55)
    player_spawn_y: tuple[int, int] = (40, 50)
    terrain_collision: bool = False
    entity_collision: bool = True
    gear_enabled: bool = False
    gear: GearConfig | None = None
    spell_rules: SpellRules = field(default_factory=SpellRules)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    innie: InnieConfig = field(default_factory=InnieConfig)
    spells: tuple[SpellConfig, ...] = field(default_factory=baseline_spells)
    timing: TimingConfig = field(default_factory=TimingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        for name, cls in (("player", PlayerConfig), ("innie", InnieConfig),
                          ("timing", TimingConfig), ("reward", RewardConfig), ("spell_rules", SpellRules)):
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"{name} must be a {cls.__name__}")
        for name in ("map_id", "width", "height"):
            integer(name, getattr(self, name), 1)
        if self.profile not in ("map53_open_entities_v1", "map53_blocked_v2") or (self.map_id, self.width, self.height) != (53,100,100):
            raise ValueError("Only the 100x100 map53 open and blocked profiles are implemented")
        if self.terrain_collision is not (self.profile == "map53_blocked_v2") or self.entity_collision is not True:
            raise ValueError("Terrain collision must match the profile; entity collision is required")
        if (type(self.gear_enabled) is not bool
                or (self.gear_enabled and not isinstance(self.gear, GearConfig))
                or (not self.gear_enabled and self.gear is not None)):
            raise ValueError("Enabled gear requires a GearConfig; disabled gear requires None")
        interval("player_spawn_x", self.player_spawn_x, self.width)
        interval("player_spawn_y", self.player_spawn_y, self.height)
        if not isinstance(self.spells, tuple) or not all(isinstance(s, SpellConfig) for s in self.spells):
            raise ValueError("spells must be a tuple of SpellConfig")
        if len({s.spell_id for s in self.spells}) != len(self.spells):
            raise ValueError("Spell IDs must be unique")
        if len({s.slot for s in self.spells}) != len(self.spells):
            raise ValueError("Spell slots must be unique")
        if {(s.slot, s.spell_id) for s in self.spells} != {(1, 416), (2, 417), (3, 418)}:
            raise ValueError("Baseline requires spell slots 1/2/3 mapped to 416/417/418")
