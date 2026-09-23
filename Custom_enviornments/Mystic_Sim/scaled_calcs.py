"""Translations of ScaledCalcs and RedBotNPCScript.OnCreated."""
import math
from dataclasses import replace


def expected_regression_dps(level):
    if level < 20: return (level + 20)**3 / 120 + 50
    if level < 50: return (level + 20)**3 / 120 + 450
    for end, start, low, high in ((110,68,8700,13600),(134,111,16800,20000),
            (180,135,37900,53200),(391,181,76000,167000),(612,392,240000,400000),
            (1068,613,500000,835000),(2050,1069,956000,2320000)):
        if level <= end: return low + (level-start) * ((high-low)/(end-start))
    return 2320000 + 550 * ((level-2050)/2.0)**1.16


def get_enemy_hp(level):
    return ((level+23)**2.7-5000)/2 if level < 150 else 1663.7*(level-50)**1.3-113132


class NPCRecursiveDamage:
    def __init__(self): self.lookup = [0.0]

    def get_damage(self, level):
        index = max(0, level)
        current = self.lookup[-1]
        for n in range(len(self.lookup), index+1):
            lv = n+1
            r = (.025*(45+112.5*lv)+lv**2/100)/2
            i = 1+(4/math.log(98/30))*math.log((lv+29)/30)
            current = min(r+10*i*(.1+lv/15)+1,
                          current+3+.5*math.log(1+math.exp(-.005*(lv-1464))))
            self.lookup.append(current)
        return self.lookup[index]


def scale_innie(config):
    level = config.effective_level
    d = NPCRecursiveDamage().get_damage(level)
    ac = (20/343)*(3*level**2+540*level-200) if level < 80 else 656.82*math.sqrt(level-49.5)
    tgh = int(level*.1)
    speed = config.attack_ms/1000
    reduction = (4*.95/3)*(.75-1/(1+.00000005*(ac+math.sqrt(1/(3*.00000005)))**2))
    adjusted = max(d*speed+ac/15, d*speed/(1-reduction))
    dodge = max(0,.24*(1-2**(-level/100)))
    block = max(0,.00055*tgh) if tgh<100 else .095*math.log(tgh-(100-.095/.00055))+100*.00055-.095*math.log(.095/.00055)
    amount = .4 if ac<50000 else .000002/(.085*50000**(.085-1))*ac**.085+.4-.000002/(.085*50000**(.085-1))*50000**.085
    scaled = int(adjusted/((1-dodge)*(1-block*amount)))
    radius = config.attack_radius
    factor = .597 if radius>2.5 else .638 if radius>2 else .697 if radius>1.5 else .824 if radius>1 else 1
    return replace(config, max_hp=int(get_enemy_hp(level)*config.bulk_factor),
                   ac=15*level, toughness=tgh, raw_damage=int(scaled*factor))
