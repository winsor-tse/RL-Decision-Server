"""Pure baseline PvE formulas. Optional status/PvP paths are not enabled."""
import math
from .state import PlayerState
from .movement import OFFSETS


def stat_factor(value):
    return 1+2.8*math.log((max(value,0)+27)/30)+.0125*max(value-2500,0)**.6


def melee_crit_chance(dex, actual_dex=None):
    # Preserve Player.cs's second branch using the actual Dexterity stat.
    actual_dex = dex if actual_dex is None else actual_dex
    if dex<300: return .4-.4*2**(-max(dex,0)/100)
    if actual_dex<3244: return .65-.3*2**(-(dex-300)/600)
    return .85-.21*2**(-(dex-3244)/12598)


def spell_crit_chance(dex, wis):
    mean = math.sqrt(8*dex*wis/3)
    if mean<300: return .4-.4*2**(-max(mean,0)/100)
    if mean<3244: return .65-.3*2**(-(mean-300)/600)
    return .85-.21*2**(-(mean-3244)/12598)


def crit_multiplier(stats, magic=False):
    value = max(0, stats.acuity if magic else stats.ferocity)
    a=.25
    c=1/(4*(1500**a+(a-1)*140**a))
    d=2+c*(a-1)*140**a
    g=a*c*((d-2)/(c*(a-1)))**((a-1)/a)
    initial=2+value*g if value<140 else d+c*value**a
    equivalent=20*value+136
    eq = spell_crit_chance(equivalent,equivalent) if magic else melee_crit_chance(equivalent,stats.dexterity)
    real = spell_crit_chance(stats.dexterity,stats.wisdom) if magic else melee_crit_chance(stats.dexterity)
    bonus=((1-eq)+eq*initial)/(1+eq)
    if real == 0:
        raise ValueError("Crit multiplier undefined at zero critical chance in supplied C#")
    return (bonus*(1+real)+real-1)/real


def raw_spell_damage(damage, stats, rng, spell_multiplier=0):
    crit=rng.chance(spell_crit_chance(stats.dexterity,stats.wisdom),"spell_crit")
    return int(damage*stat_factor(stats.intelligence)*(crit_multiplier(stats,True) if crit else 1)*(1+spell_multiplier)),crit


def dodge_chance(dex): return max(0,.24*(1-2**(-dex/100)))


def block_chance(toughness, precision, player_vs_monster=False):
    if toughness<=0: return 0.0
    factor=2 if precision==0 else min(2,max(.01,1-.2*math.log2(precision/toughness)))
    if player_vs_monster: factor=min(1,factor)
    b=100-.095/.00055
    chance=max(0,.00055*toughness) if toughness<100 else .095*math.log(toughness-b)+100*.00055-.095*math.log(100-b)
    return max(0,chance*factor)


def block_amount(ac):
    d=.000002/(.085*50000**(.085-1))
    return max(0,d*ac**.085+.4-d*50000**.085) if ac>50000 else .4


def npc_ac_reduction(ac):
    if ac < -900: return -math.sqrt(-(ac+400)/12500)+.0403
    c,r=.95,1/10000000
    return (.9827*c if ac<0 else c/3)*(.75-1/(1+r*(ac+math.sqrt(1/(3*r)))**2))


def player_ac_reduction(raw, ac):
    c,r=.95,1/20000000
    return max(ac/15,raw*(4*c/3)*(.75-1/(1+r*(ac+math.sqrt(1/(3*r)))**2)))


def facing_bonus(attacker,target):
    offset=OFFSETS.get(attacker.facing)
    if offset is None or (attacker.x+offset[0],attacker.y+offset[1])!=(target.x,target.y): return 1.0
    if attacker.facing==target.facing: return 1.5
    other=OFFSETS.get(target.facing)
    return 1.25 if other and offset[0]*other[0]+offset[1]*other[1]==0 else 1.0


def mitigate(damage, attacker, target, rng, *, magic=False):
    raw=damage*(1 if magic else facing_bonus(attacker,target))
    player=isinstance(target,PlayerState)
    chance=block_chance(target.stats.toughness,getattr(attacker.stats,"precision",0),
                        player and not isinstance(attacker,PlayerState))
    blocked=rng.chance(chance,"block")
    if blocked:
        raw*=1-block_amount(target.stats.ac)
        if not player: raw=int(raw)
    ac=target.stats.ac
    if player: reduction=player_ac_reduction(raw,ac)
    elif ac < -900: reduction=raw*npc_ac_reduction(ac)
    elif ac < 0: reduction=min(ac/15,raw*npc_ac_reduction(ac))
    else: reduction=max(min(100,int(ac/15)),raw*npc_ac_reduction(ac))
    return int(math.ceil(max(.02*raw,raw-reduction))),blocked


def apply_aoe(damage,target_count):
    if target_count<=0: raise ValueError("AoE requires at least one target")
    n=target_count
    factor=(25*n**(1/24)-24)/n**(1+1/24) if n<=3 else (2.45+.375*(n-4))/n if n<=12 else (1.16+(n-4)**.7)/n
    return int(damage*factor)


def can_hit(attacker,target,*,magic=False):
    return (target is not None and attacker is not target and attacker.alive and target.alive
            and isinstance(attacker,PlayerState)!=isinstance(target,PlayerState)
            and not (magic and target.magic_immune))


def regen_amount(stat,maximum,base_stat):
    stat_regen=3.0*stat-225 if stat>150 else max(stat,0)**2/100
    return int(round(.025*max(0,maximum-base_stat)+stat_regen))
