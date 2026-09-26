"""Optional Pygame client. Run with python -m Custom_enviornments.Mystic_Sim.viewer."""
import argparse
from collections import deque
from dataclasses import replace
import os
from pathlib import Path

from .config import ScenarioConfig
from .env import MysticSimEnv
from .movement import OFFSETS
from .targeting import select_target


class PlaySession:
    """UI state stays outside the Gym environment and never edits combat state."""
    def __init__(self, seed=42, training_rules=False):
        config = ScenarioConfig(profile="map53_blocked_v2", terrain_collision=True)
        if not training_rules:
            config = replace(config, reward=replace(config.reward,
                              win_kills=2**31-1, max_episode_steps=2**31-1))
        self.env = MysticSimEnv(config=config)
        self.seed, self.training_rules = seed, training_rules
        self.paused = True
        self.speed = 1.0
        self.reset()

    def reset(self, new_seed=False):
        if new_seed:
            self.seed += 1
        _, self.info = self.env.reset(seed=self.seed)
        self.total_reward = 0.0
        self.log = deque(["Ready. Space to play; 1 / 2 / 3 to cast."], maxlen=5)

    def step(self, action=None):
        if self.paused or self.env.episode_done:
            return None
        # The baseline disables melee. Action 4 advances time without a player
        # mutation, preserving the eight-action training contract for idle frames.
        _, reward, _, _, self.info = self.env.step(4 if action is None else action)
        self.total_reward += reward
        if action is not None and not self.info['action_applied']:
            message = self.info['action_failure_reason'].replace('_', ' ')
            if not self.log or self.log[-1] != message:
                self.log.append(message)
        for event in self.info['cast_events']:
            self.log.append({416:'Arcane Blast',417:'Acid Cloud',418:'Tempest Inferno'}[event['spell_id']])
        for event in self.info['death_events']:
            self.log.append('You died. R to restart.' if event['entity_id'] == 1 else f"Innie {event['entity_id']} defeated")
        for event in self.info['respawn_events']:
            self.log.append(f"Innie {event['entity_id']} respawned")
        return self.info

    def close(self):
        self.env.close()


class Viewer:
    BG = (12, 19, 29)
    PANEL = (19, 29, 43)
    TEXT = (224, 232, 240)
    MUTED = (132, 153, 173)
    BLUE = (96, 191, 255)
    GREEN = (108, 222, 151)
    RED = (245, 111, 111)

    def __init__(self, pygame, session, size=(1320, 860)):
        self.pg, self.session = pygame, session
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption('Mystic Sim | Map 53')
        self.font = pygame.font.Font(None, 23)
        self.small = pygame.font.Font(None, 19)
        self.title = pygame.font.Font(None, 34)
        self.zoom, self.overview = 24, False
        self.show_spawns, self.show_range = False, True
        self.pending = None
        self.accumulator = 0.0
        self.flashes = []
        self.floating = []
        self.running = True
        self.keys = {
            pygame.K_w:0, pygame.K_UP:0, pygame.K_s:1, pygame.K_DOWN:1,
            pygame.K_a:2, pygame.K_LEFT:2, pygame.K_d:3, pygame.K_RIGHT:3,
            pygame.K_1:5, pygame.K_2:6, pygame.K_3:7, pygame.K_f:4,
        }

    def text(self, value, x, y, color=None, font=None):
        self.screen.blit((font or self.font).render(str(value), True, color or self.TEXT), (x,y))

    def events(self):
        p = self.pg
        for e in p.event.get():
            if e.type == p.QUIT:
                self.running = False
            elif e.type == p.VIDEORESIZE:
                self.screen = p.display.set_mode((max(1000,e.w),max(760,e.h)),p.RESIZABLE)
            elif e.type == p.WINDOWFOCUSLOST:
                self.session.paused = True
                self.pending = None
                self.accumulator = 0
            elif e.type == p.MOUSEWHEEL:
                self.zoom = max(12,min(44,self.zoom+e.y*2))
            elif e.type == p.KEYDOWN:
                if e.key == p.K_ESCAPE:
                    self.running = False
                elif e.key == p.K_SPACE:
                    self.session.paused = not self.session.paused
                    self.pending = None
                    self.accumulator = 0
                elif e.key == p.K_r:
                    self.session.reset(new_seed=bool(e.mod & p.KMOD_SHIFT))
                    self.pending = None
                    self.accumulator = 0
                    self.flashes.clear()
                    self.floating.clear()
                elif e.key == p.K_TAB:
                    self.overview = not self.overview
                elif e.key == p.K_b:
                    self.show_spawns = not self.show_spawns
                elif e.key == p.K_c:
                    self.show_range = not self.show_range
                elif e.key in (p.K_EQUALS,p.K_PLUS,p.K_MINUS):
                    speeds = [.25,.5,1.,2.,4.]
                    index = speeds.index(self.session.speed)+( -1 if e.key == p.K_MINUS else 1)
                    self.session.speed = speeds[max(0,min(4,index))]
                elif e.key in self.keys and not self.session.paused:
                    self.pending = self.keys[e.key]

    def advance(self, seconds):
        if self.session.paused or self.session.env.episode_done:
            self.accumulator = 0
            return
        self.accumulator += min(seconds,.25)*self.session.speed
        held = self.pg.key.get_pressed()
        while self.accumulator >= .2 and not self.session.env.episode_done:
            action = self.pending
            self.pending = None
            if action is None:
                # Held spells take priority over held movement. One action per decision.
                action = next((self.keys[k] for k in (self.pg.K_1,self.pg.K_2,self.pg.K_3,
                    self.pg.K_w,self.pg.K_UP,self.pg.K_s,self.pg.K_DOWN,self.pg.K_a,self.pg.K_LEFT,
                    self.pg.K_d,self.pg.K_RIGHT,self.pg.K_f) if held[k]),None)
            info = self.session.step(action)
            self.accumulator -= .2
            if info:
                self.capture_effects(info)

    def capture_effects(self, info):
        w = self.session.env.world
        for cast in info['cast_events']:
            center = w.player if cast['target_id'] == 1 else w.monsters.get(cast['target_id'])
            if center:
                self.flashes.append([center.x,center.y,cast['spell_id'],.45])
        for event in info['damage_events']:
            target = w.player if event['target_id']==1 else w.monsters.get(event['target_id'])
            if target:
                label = f"-{event['damage']:,}" if event['damage'] else 'Dodge'
                self.floating.append([target.x,target.y,label,.9,event['target_id']==1])

    def projection(self):
        w = self.session.env.world
        width,height = self.screen.get_size()
        self.view = self.pg.Rect(18,76,width-366,height-108)
        self.scale = min(self.view.w/100,self.view.h/100) if self.overview else self.zoom
        tiles_x,tiles_y = self.view.w/self.scale,self.view.h/self.scale
        self.cam_x = (100-tiles_x)/2 if self.overview else max(0,min(100-tiles_x,w.player.x-tiles_x/2))
        self.cam_y = (100-tiles_y)/2 if self.overview else max(0,min(100-tiles_y,w.player.y-tiles_y/2))

    def point(self, x, y):
        return (round(self.view.x+(x-self.cam_x+.5)*self.scale),
                round(self.view.y+(y-self.cam_y+.5)*self.scale))

    def cell_rect(self, x, y, width=1, height=1):
        cx,cy=self.point(x-.5,y-.5)
        return self.pg.Rect(cx,cy,round(width*self.scale),round(height*self.scale))

    def bar(self, x, y, width, height, value, maximum, color):
        self.pg.draw.rect(self.screen,(35,47,61),(x,y,width,height),border_radius=3)
        fraction=max(0,min(1,value/maximum)) if maximum else 0
        if fraction:
            self.pg.draw.rect(self.screen,color,(x,y,max(1,round(width*fraction)),height),border_radius=3)

    def draw_world(self, seconds):
        pg,w=self.pg,self.session.env.world
        self.screen.set_clip(self.view)
        pg.draw.rect(self.screen,(22,36,43),self.cell_rect(0,0,100,100))
        # The supplied blocked layer is solid terrain in the viewer's profile.
        for x,y in w.map.blocked_cells:
            rect=self.cell_rect(x,y)
            if self.view.colliderect(rect):
                pg.draw.rect(self.screen,(10,20,27),rect)
        if not self.overview:
            for x in range(max(0,int(self.cam_x)), min(100,int(self.cam_x+self.view.w/self.scale)+1)+1):
                px=self.cell_rect(x,0).x
                pg.draw.line(self.screen,(30,45,53),(px,self.view.top),(px,self.view.bottom))
            for y in range(max(0,int(self.cam_y)), min(100,int(self.cam_y+self.view.h/self.scale)+1)+1):
                py=self.cell_rect(0,y).y
                pg.draw.line(self.screen,(30,45,53),(self.view.left,py),(self.view.right,py))
        if self.show_spawns:
            for box in w.map.spawn_boxes:
                pg.draw.rect(self.screen,(65,95,78),self.cell_rect(box.x,box.y,box.width,box.height),1)
        if self.show_range:
            pg.draw.rect(self.screen,(58,99,135),self.cell_rect(w.player.x-16,w.player.y-10,33,21),2)
        target=select_target(w,self.session.env.config.spells[0])
        acid_targets={e.target_id for e in w.effects}
        for npc in w.monsters.values():
            pos=self.point(npc.x,npc.y)
            if not self.view.collidepoint(pos): continue
            radius=max(3,int(self.scale*.32))
            if not npc.alive:
                pg.draw.circle(self.screen,(78,83,89),pos,max(2,radius-1),1)
                if not self.overview and npc.respawn_at_ms:
                    self.text(f"{max(0,(npc.respawn_at_ms-w.time_ms)/1000):.0f}s",pos[0]-10,pos[1]+8,self.MUTED,self.small)
                continue
            color=self.RED if npc.aggro_target else (212,165,104)
            pg.draw.circle(self.screen,color,pos,radius)
            if npc.entity_id in acid_targets:
                pg.draw.circle(self.screen,self.GREEN,pos,radius+3,2)
            if npc is target:
                pg.draw.circle(self.screen,(249,224,139),pos,radius+6,1)
            if not self.overview:
                self.bar(pos[0]-12,pos[1]-radius-7,24,3,npc.hp,npc.max_hp,self.RED)
        pos=self.point(w.player.x,w.player.y)
        radius=max(4,int(self.scale*.4))
        pg.draw.circle(self.screen,(14,26,39),pos,radius+3)
        pg.draw.circle(self.screen,self.BLUE if w.player.alive else self.MUTED,pos,radius)
        dx,dy=OFFSETS.get(w.player.facing,(0,0))
        pg.draw.line(self.screen,(238,248,255),pos,(pos[0]+dx*(radius+4),pos[1]+dy*(radius+4)),3)
        for flash in self.flashes:
            x,y,spell,life=flash
            color={416:self.BLUE,417:self.GREEN,418:(247,153,95)}[spell]
            radius={416:.65,417:4.25,418:1.5}[spell]
            pg.draw.circle(self.screen,color,self.point(x,y),max(2,int(radius*self.scale*(1-life/1.1))),2)
            flash[3]-=seconds
        self.flashes[:]=[f for f in self.flashes if f[3]>0]
        for floating in self.floating:
            x,y,label,life,player=floating
            px,py=self.point(x,y)
            self.text(label,px-16,py-25-int((.9-life)*28),self.RED if player else self.TEXT,self.small)
            floating[3]-=seconds
        self.floating[:]=[f for f in self.floating if f[3]>0]
        self.screen.set_clip(None)

    def draw_panel(self):
        pg,w=self.pg,self.session.env.world
        x=self.screen.get_width()-326
        pg.draw.rect(self.screen,self.PANEL,(x-12,76,320,self.screen.get_height()-108),border_radius=10)
        self.text('MYSTIC',x,94,self.BLUE,self.title)
        self.text(f'Level {w.player.stats.level}   /   ({w.player.x}, {w.player.y})',x,130,self.MUTED)
        self.text(f'HP   {w.player.hp:,} / {w.player.max_hp:,}',x,164)
        self.bar(x,190,290,9,w.player.hp,w.player.max_hp,self.RED)
        self.text(f'MP   {w.player.mp:,} / {w.player.max_mp:,}',x,212)
        self.bar(x,238,290,9,w.player.mp,w.player.max_mp,self.BLUE)
        names=['Arcane Blast','Acid Cloud','Tempest Inferno']
        for i,spell in enumerate(self.session.env.config.spells):
            y=271+i*47
            ready=max(w.player.cooldowns.slots.get(spell.slot,0),
                      w.player.cooldowns.families.get(spell.family,0),w.player.cooldowns.global_ready_at_ms)
            wait=max(0,ready-w.time_ms)/1000
            state=f'{wait:.1f}s' if wait else 'READY' if w.player.mp>=spell.mp_cost else 'LOW MP'
            self.text(f'{i+1}   {names[i]}',x,y)
            self.text(f'{spell.mp_cost:,} MP + {spell.mana_consumption:.0%}   |   {state}',x,y+22,self.MUTED,self.small)
        target=select_target(w,self.session.env.config.spells[0])
        self.text(f'Target: Innie {target.entity_id}' if target else 'Target: none in casting range',x,426,self.GREEN)
        if target:
            self.text(f'{target.hp:,} / {target.max_hp:,} HP',x,450,self.MUTED,self.small)
        self.text(f'Kills {w.kills}    Respawning {sum(not m.alive for m in w.monsters.values())}',x,479)
        self.text(f'Reward {self.session.total_reward:+.3f}',x,504,self.MUTED,self.small)
        self.text('CONTROLS',x,542,self.BLUE)
        for i,line in enumerate(('WASD / arrows   Move','1 / 2 / 3   Cast (hold to repeat)',
                                 'Space   Pause       R   Restart','Shift+R   New seed    Tab   Map',
                                 'Wheel   Zoom      - / +   Speed','C   Cast area     B   Spawn boxes')):
            self.text(line,x,570+i*23,self.MUTED,self.small)
        self.text('Melee disabled in baseline',x,718,self.MUTED,self.small)

    def draw(self, seconds=0):
        self.projection()
        self.screen.fill(self.BG)
        w=self.session.env.world
        self.text('MYSTIC / FIELD SIMULATOR',20,19,self.TEXT,self.title)
        mode='TRAINING' if self.session.training_rules else 'FREE PLAY'
        self.text(f'MAP 53   |   {mode}   |   SEED {self.session.seed}   |   {w.time_ms/1000:.1f}s   |   {self.session.speed:g}x',20,51,self.MUTED,self.small)
        self.draw_world(seconds)
        self.draw_panel()
        self.text('Dark terrain is solid. Players and Innies stay on walkable tiles.',20,self.screen.get_height()-25,self.MUTED,self.small)
        if self.session.log:
            self.text(self.session.log[-1],self.view.x+12,self.view.bottom-28,self.TEXT)
        if self.session.paused or self.session.env.episode_done:
            label='PAUSED  /  SPACE TO PLAY' if not self.session.env.episode_done else f"{self.session.info['episode_outcome'].upper()}  /  R TO RESTART"
            surface=self.title.render(label,True,self.TEXT)
            rect=surface.get_rect(center=self.view.center)
            self.pg.draw.rect(self.screen,self.PANEL,rect.inflate(40,28),border_radius=10)
            self.screen.blit(surface,rect)
        self.pg.display.flip()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--training-rules',action='store_true',help='Stop at five kills / 256 decisions')
    parser.add_argument('--smoke-test',action='store_true',help='Run scripted steps and render with SDL dummy driver')
    parser.add_argument('--screenshot',type=Path,help='Save the final frame as a PNG')
    args=parser.parse_args(argv)
    if args.seed<0: parser.error('--seed must be nonnegative')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
    if args.smoke_test:
        os.environ['SDL_VIDEODRIVER']='dummy'
    try:
        import pygame
    except ImportError:
        parser.exit(1,'Pygame is missing. Run: .\\RL_venv\\Scripts\\python.exe -m pip install -r requirements-viewer.txt\n')
    session=PlaySession(args.seed,args.training_rules)
    try:
        viewer=Viewer(pygame,session)
        if args.smoke_test:
            session.paused=False
            for action in [0,3,6,4,5,7,1,2]*5:
                if session.env.episode_done: break
                viewer.capture_effects(session.step(action))
                viewer.draw(.2)
            print(f'Viewer smoke test passed: {session.env.world.step_count} steps, {session.env.world.time_ms} ms')
        else:
            clock=pygame.time.Clock()
            while viewer.running:
                elapsed=clock.tick(60)/1000
                viewer.events()
                viewer.advance(elapsed)
                viewer.draw(elapsed)
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            pygame.image.save(viewer.screen,str(args.screenshot))
    finally:
        session.close()
        pygame.quit()


if __name__=='__main__':
    main()
