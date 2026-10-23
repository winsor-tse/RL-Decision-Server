"""Policy-watch dashboard. Inspector/probability panels consume detached snapshots.

TODO(live): reuse the same display contract with an authoritative ZMQ producer;
one frozen-actor inference episode per launch, retaining the final frame on death,
without a reset loop or live training. See WATCH_POLICY.md. This viewer currently
runs the local simulation only.
"""
from Custom_enviornments.Mystic_Sim.viewer import Viewer
from .watch_state import ACTION_NAMES, FEATURE_NAMES
from .history import LIVE_HISTORY_FEATURES


class PolicyViewer(Viewer):
    BG = (10, 16, 25)
    PANEL = (18, 27, 40)
    CARD = (25, 37, 52)
    LINE = (40, 56, 73)
    GOLD = (245, 201, 112)
    TABS = ('Summary', 'Player', 'Enemies', 'History', 'Inputs')

    def __init__(self, pygame, session, size=(1480, 920)):
        super().__init__(pygame, session, size)
        self.keys.clear()
        self.tab, self.state_view, self.scroll = 'Summary', 'current', 0
        self.content_height = 0
        self.checkpoint_name = 'Frozen checkpoint'
        self.tiny = pygame.font.Font(None, 18)
        self.mono = pygame.font.SysFont('consolas', 15)
        self.projection()

    def projection(self):
        width, height = self.screen.get_size()
        side_width = 444 if width >= 1280 else 404
        self.inspector = self.pg.Rect(width - side_width - 18, 86, side_width, height - 114)
        self.probability_panel = self.pg.Rect(18, height - 280,
                                            self.inspector.x - 32, 252)
        self.view = self.pg.Rect(18, 86, self.inspector.x - 32,
                                 self.probability_panel.y - 98)
        self.inspector_body = self.pg.Rect(self.inspector.x + 16, 210,
                                           self.inspector.w - 36, height - 258)
        w = self.session.env.world
        self.scale = min(self.view.w / w.map.width, self.view.h / w.map.height) if self.overview else self.zoom
        tiles_x, tiles_y = self.view.w / self.scale, self.view.h / self.scale
        self.cam_x = ((w.map.width - tiles_x) / 2 if self.overview else
                      max(0, min(w.map.width - tiles_x, w.player.x - tiles_x / 2)))
        self.cam_y = ((w.map.height - tiles_y) / 2 if self.overview else
                      max(0, min(w.map.height - tiles_y, w.player.y - tiles_y / 2)))

    def fit_text(self, value, x, y, width, color=None, font=None):
        font = font or self.small
        value = str(value)
        if font.size(value)[0] > width:
            while value and font.size(value + '...')[0] > width:
                value = value[:-1]
            value += '...'
        self.text(value, x, y, color, font)

    def card(self, rect, color=None):
        self.pg.draw.rect(self.screen, color or self.PANEL, rect, border_radius=10)
        self.pg.draw.rect(self.screen, self.LINE, rect, 1, border_radius=10)

    def pill(self, label, rect, active=False):
        self.pg.draw.rect(self.screen, self.CARD if not active else (34, 65, 85), rect, border_radius=6)
        surface = self.small.render(label, True, self.BLUE if active else self.MUTED)
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def tab_rects(self):
        width = (self.inspector.w - 24) / len(self.TABS)
        return [(name, self.pg.Rect(self.inspector.x + 12 + i * width, 172, width - 3, 27))
                for i, name in enumerate(self.TABS)]

    def state_rects(self):
        x = self.inspector.x + 16
        width = (self.inspector.w - 36) // 2
        return [(name, label, self.pg.Rect(x + i * (width + 4), 131, width, 28))
                for i, (name, label) in enumerate((('current', 'Current state'),
                                                  ('decision', 'Previous state')))]

    def handle_event(self, event):
        pg = self.pg
        if event.type == pg.MOUSEWHEEL and self.inspector.collidepoint(pg.mouse.get_pos()):
            self.scroll = max(0, min(self.max_scroll, self.scroll - event.y * 42))
            return
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            for name, rect in self.tab_rects():
                if rect.collidepoint(event.pos):
                    self.tab, self.scroll = name, 0
                    return
            for name, _, rect in self.state_rects():
                if rect.collidepoint(event.pos):
                    self.state_view, self.scroll = name, 0
                    return
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_n:
                if self.session.paused and not self.session.env.episode_done:
                    self.session.paused = False
                    try:
                        info = self.session.step()
                        if info:
                            self.capture_effects(info)
                    finally:
                        self.session.paused = True
                        self.accumulator = 0
                return
            if event.key in (pg.K_PAGEUP, pg.K_PAGEDOWN, pg.K_HOME):
                change = -self.inspector_body.h * .8 if event.key == pg.K_PAGEUP else self.inspector_body.h * .8
                self.scroll = 0 if event.key == pg.K_HOME else max(0, min(self.max_scroll, self.scroll + change))
                return
        super().handle_event(event)
        if event.type == pg.VIDEORESIZE:
            self.projection()
            self.scroll = 0

    @property
    def max_scroll(self):
        return max(0, self.content_height - self.inspector_body.h)

    def section(self, title, y):
        x = self.inspector_body.x
        self.text(title.upper(), x, y + 4, self.BLUE, self.small)
        self.pg.draw.line(self.screen, self.LINE, (x, y + 25), (self.inspector_body.right - 4, y + 25))
        return y + 35

    def line(self, label, value, y, color=None):
        x, right = self.inspector_body.x, self.inspector_body.right - 6
        self.text(label, x, y, self.MUTED, self.small)
        value = str(value)
        self.text(value, right - self.small.size(value)[0], y, color, self.small)
        return y + 24

    def note(self, value, y, color=None):
        self.fit_text(value, self.inspector_body.x, y, self.inspector_body.w - 8,
                      color or self.MUTED, self.tiny)
        return y + 22

    def player_content(self, state, y):
        p = state.player
        y = self.section('Player / observed state', y)
        y = self.line('Position / map', f'({p["x"]}, {p["y"]}) / {p["map_id"]}', y)
        y = self.line('Facing', p['facing'].upper(), y)
        for resource, color in (('hp', self.RED), ('mp', self.BLUE)):
            maximum, value = p['max_' + resource], p[resource]
            y = self.line(resource.upper(), f'{value:,} / {maximum:,}  ({value / maximum:.1%})', y, color)
            self.bar(self.inspector_body.x, y, self.inspector_body.w - 8, 7, value, maximum, color)
            y += 19
        return y

    def spell_content(self, state, y):
        y = self.section('Cooldowns / engine diagnostics', y)
        y = self.note('Exact timers are not part of the policy input.', y)
        for spell in state.spells:
            wait = spell['cooldown_seconds']
            status = f'{wait:.2f}s' if wait else 'READY'
            if not wait and state.player['mp'] < spell['mp_cost']:
                status = 'LOW MP'
            y = self.line(spell['name'], status, y, self.GOLD if wait else self.GREEN)
            y = self.note(f'{spell["mp_cost"]:,} MP + {spell["mana_consumption"]:.0%} consumption', y)
        return y

    def enemy_content(self, state, y, limit=10):
        directions = ('up', 'down', 'left', 'right', 'idle')
        y = self.section('Enemies / policy distance order', y)
        y = self.note('IDs are diagnostic labels. Only slots 1-5 expose dx/dy.', y)
        if not state.enemies:
            return self.note('No living enemies in observation. Slots are padded.', y)
        for enemy in state.enemies[:limit]:
            marker = ' [target]' if enemy['entity_id'] == state.target_id else ''
            y = self.line(f'#{enemy["rank"]}  Innie {enemy["entity_id"]}{marker}',
                          f'{enemy["distance"]:g} tiles / {enemy["hp_fraction"]:.0%} HP', y,
                          self.GOLD if marker else self.TEXT)
            delta = (f'({enemy["dx"]:+g}, {enemy["dy"]:+g})'
                     if enemy['dx'] is not None else 'not in input')
            y = self.note(f'{directions[enemy["relative_direction"]].upper()}  |  MP {enemy["mp_fraction"]:.0%}  |  dx/dy {delta}', y)
        return y

    def history_content(self, state, y):
        y = self.section('Observable history / policy inputs', y)
        if not state.history:
            return self.note('This feed-forward policy has no history inputs.', y)
        h = state.history
        previous = next((ACTION_NAMES[i] for i, value in enumerate(h[:7]) if value), 'None / reset')
        y = self.line('Previous chosen action', previous, y)
        if len(h) == 15:
            y = self.note('Chosen commands only; execution outcomes are not policy inputs.', y)
            y = self.section('Local terrain / four-tile rays', y + 8)
            for i, name in enumerate(('Up', 'Down', 'Left', 'Right')):
                free, detected = h[7 + 2*i:9 + 2*i]
                y = self.line(name, f'{free * 4:g} free / blocker' if detected else '4+ free / clear', y)
            return y
        for label, value in zip(('Applied', 'Movement blocked', 'Cooldown rejected'), h[7:10]):
            y = self.line(label, 'Yes' if value else 'No', y, self.GREEN if value else self.MUTED)
        y = self.section('Confirmed casts / capped ages', y + 8)
        for index, name in enumerate(ACTION_NAMES[4:]):
            age, seen = h[10 + index], h[13 + index]
            value = ('>=10.0s' if age >= 1 else f'{age * 10:.1f}s ago') if seen else 'Not seen'
            y = self.line(name, value, y, self.TEXT if seen else self.MUTED)
        y = self.section('Local terrain / four-tile rays', y + 8)
        for i, name in enumerate(('Up', 'Down', 'Left', 'Right')):
            free, detected = h[16 + 2 * i:18 + 2 * i]
            value = f'{free * 4:g} free / blocker' if detected else '4+ free / clear'
            y = self.line(name, value, y)
            self.bar(self.inspector_body.x, y, self.inspector_body.w - 8, 5, free, 1, self.BLUE)
            y += 14
        return self.note('Static terrain only; moving entities are excluded.', y + 4)

    def summary_content(self, state, y):
        y = self.player_content(state, y)
        y = self.section('Encounter / engine diagnostics', y + 4)
        y = self.line('Kills / goal', f'{state.kills} / {state.kill_goal}', y)
        y = self.line('Alive / respawning', f'{state.alive_enemies} / {state.respawning}', y)
        y = self.line('Arcane target', f'Innie {state.target_id}' if state.target_id is not None else 'None', y)
        if state.last_action is not None:
            y = self.line('Last action', ACTION_NAMES[state.last_action], y)
            result = 'Applied' if state.action_applied else (state.failure_reason or 'Not applied').replace('_', ' ')
            y = self.line('Result', result, y, self.GREEN if state.action_applied else self.GOLD)
        y = self.section('Reward / latest transition', y + 4)
        y = self.line('Episode return', f'{state.total_reward:+.3f}', y)
        for name, value in state.reward_components:
            if value:
                y = self.line(name.replace('_', ' ').capitalize(), f'{value:+.3f}', y,
                              self.GREEN if value > 0 else self.RED)
        y = self.section('Event feed', y + 4)
        for message in state.events[-5:]:
            y = self.note(message, y, self.TEXT)
        return y

    def input_content(self, state, y):
        y = self.section(f'Exact input tensor / {len(state.observation)} values', y)
        y = self.note('Columns: index / feature   |   raw   |   normalized', y)
        x, right = self.inspector_body.x, self.inspector_body.right - 8
        names = (FEATURE_NAMES[:56] + tuple(LIVE_HISTORY_FEATURES)
                 if len(state.observation) == 71 else FEATURE_NAMES)
        for index, (value, scaled) in enumerate(zip(state.observation, state.normalized_observation)):
            terrain_index = 63 if len(state.observation) == 71 else 72
            if index in (6, 46, 56, terrain_index):
                title = {6: 'Enemy slots', 46: 'Signed relative offsets',
                         56: 'Observable history', terrain_index: 'Terrain sensors'}[index]
                y = self.section(title, y + 4)
            value_text = f'{value:8.3f}  {scaled:7.3f}'
            values_x = right - self.mono.size(value_text)[0]
            self.fit_text(f'{index:02}  {names[index]}', x, y,
                          values_x - x - 16, self.MUTED, self.tiny)
            self.text(value_text, values_x, y, self.TEXT, self.mono)
            y += 23
        return y

    def draw_inspector(self):
        self.card(self.inspector)
        self.text('STATE INSPECTOR', self.inspector.x + 16, 100, self.TEXT)
        for name, label, rect in self.state_rects():
            self.pill(label, rect, self.state_view == name)
        for name, rect in self.tab_rects():
            self.pill(name, rect, self.tab == name)
        state = self.session.snapshot
        if self.state_view == 'decision':
            state = self.session.decision.input_snapshot if self.session.decision else None
        self.screen.set_clip(self.inspector_body)
        y = self.inspector_body.y - self.scroll
        if state is None:
            y = self.note('Space to play or N to inspect the first decision.', y + 10)
        else:
            y = self.note(f'Step {state.step}  /  {state.time_ms / 1000:.1f}s  |  '
                          + ('Used to choose the last action' if self.state_view == 'decision' else 'Latest world state'), y)
            if self.tab == 'Summary':
                y = self.summary_content(state, y)
            elif self.tab == 'Player':
                y = self.spell_content(state, self.player_content(state, y))
            elif self.tab == 'Enemies':
                y = self.enemy_content(state, y)
            elif self.tab == 'History':
                y = self.history_content(state, y)
            else:
                y = self.input_content(state, y)
        self.content_height = y + self.scroll - self.inspector_body.y + 12
        self.screen.set_clip(None)
        self.scroll = min(self.scroll, self.max_scroll)
        if self.max_scroll:
            track = self.pg.Rect(self.inspector.right - 10, self.inspector_body.y, 3, self.inspector_body.h)
            self.pg.draw.rect(self.screen, self.LINE, track, border_radius=1)
            handle = max(24, int(track.h * self.inspector_body.h / self.content_height))
            top = track.y + int((track.h - handle) * self.scroll / self.max_scroll)
            self.pg.draw.rect(self.screen, self.BLUE, (track.x, top, 3, handle), border_radius=1)

    def draw_probabilities(self):
        rect = self.probability_panel
        self.card(rect)
        x, y = rect.x + 16, rect.y + 14
        self.text('ACTION PROBABILITIES', x, y, self.TEXT)
        decision = self.session.decision
        if decision:
            detail = (f'{self.session.mode.upper()}  |  input step {decision.input_snapshot.step}'
                      f'  |  entropy {decision.entropy_bits:.2f} bits')
        else:
            detail = 'Space to play / N to step'
        self.fit_text(detail, x, y + 24, rect.w - 32, self.MUTED, self.small)
        column_width = (rect.w - 44) // 2
        top = max(range(7), key=lambda i: decision.probabilities[i]) if decision else None
        for index, label in enumerate(ACTION_NAMES):
            col, row = (0, index) if index < 4 else (1, index - 4)
            bx = x + col * (column_width + 12)
            by = y + 54 + row * 36
            chosen = decision is not None and decision.action == index
            color = self.BLUE if index < 4 else (self.BLUE, self.GREEN, (247, 153, 95))[index - 4]
            if chosen:
                self.pg.draw.rect(self.screen, self.CARD, (bx - 6, by - 4, column_width + 8, 34), border_radius=5)
                self.pg.draw.rect(self.screen, self.GOLD, (bx - 6, by - 4, 3, 34), border_radius=1)
            self.fit_text(label + (' *' if chosen else ''), bx, by, column_width - 66,
                          self.TEXT if chosen else self.MUTED, self.small)
            value = f'{decision.probabilities[index]:.1%}' if decision else '--'
            self.text(value, bx + column_width - self.small.size(value)[0], by,
                      self.GOLD if index == top else self.TEXT, self.small)
            self.bar(bx, by + 21, column_width, 5,
                     decision.probabilities[index] if decision else 0, 1, color)
        self.fit_text('* Selected action   /   Gold value = highest probability   /   No action masking',
                      x, rect.bottom - 22, rect.w - 32, self.MUTED, self.tiny)

    def draw(self, seconds=0):
        self.projection()
        self.screen.fill(self.BG)
        state = self.session.snapshot
        status = (state.last_outcome or 'finished').upper() if self.session.env.episode_done else (
            'PAUSED' if self.session.paused else 'PLAYING')
        self.text('MYSTIC WATCH', 20, 17, self.TEXT, self.title)
        self.fit_text(self.checkpoint_name, 250, 27, self.screen.get_width() - 520, self.MUTED)
        self.pill('SIMULATION', self.pg.Rect(self.screen.get_width() - 246, 17, 112, 29))
        self.pill(status, self.pg.Rect(self.screen.get_width() - 126, 17, 108, 29), True)
        self.fit_text(f'MAP {state.player["map_id"]}   |   SEED {self.session.seed}   |   '
                      f'{state.time_ms / 1000:.1f}s   |   STEP {state.step} / {state.step_limit}   |   '
                      f'KILLS {state.kills} / {state.kill_goal}   |   {self.session.speed:g}x',
                      20, 57, self.screen.get_width() - 40, self.MUTED)
        self.draw_world(seconds)
        # Keep the map label legible against terrain without covering the player.
        self.pg.draw.rect(self.screen, self.PANEL, (self.view.x + 8, self.view.y + 8, 240, 26), border_radius=5)
        self.text('FIELD VIEW  /  ' + ('OVERVIEW' if self.overview else 'FOLLOW PLAYER'),
                  self.view.x + 16, self.view.y + 14, self.MUTED, self.tiny)
        if self.session.env.episode_done:
            self.card(self.pg.Rect(self.view.x + 12, self.view.bottom - 66, self.view.w - 24, 54))
            self.fit_text(f'{status}: {(state.end_reason or "").replace("_", " ")}  /  R to replay',
                          self.view.x + 26, self.view.bottom - 51, self.view.w - 52, self.GOLD)
        elif self.session.paused:
            self.pill('PAUSED  /  SPACE TO PLAY  /  N TO STEP',
                      self.pg.Rect(self.view.x + 12, self.view.bottom - 40, self.view.w - 24, 28), True)
        self.draw_probabilities()
        self.draw_inspector()
        self.fit_text('SPACE pause   N step   R replay   SHIFT+R new seed   +/- speed   TAB map   B spawns   C range   '
                      'Wheel: map zoom / inspector scroll   PgUp/PgDn scroll   ESC quit',
                      20, self.screen.get_height() - 20, self.screen.get_width() - 40, self.MUTED, self.tiny)
        self.pg.display.flip()
