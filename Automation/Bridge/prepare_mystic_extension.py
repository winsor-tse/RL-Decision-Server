"""Prepare reviewable extension updates in a separate directory; never edit the source."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Extension source changed: expected one occurrence of {old[:70]!r}')
    return text.replace(old, new, 1)


def prepare(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    originals = {name: (source / name).read_text(encoding='utf-8')
                 for name in ('battler.js', 'content-bridge.js', 'background.js')}
    changed = dict(originals)
    text = changed['battler.js']
    text = replace_once(text, 'let aiLoopStarted = false;', 'let aiLoopStarted = false;\nlet aiLoopGeneration = 0;')
    text = replace_once(text, '                worldState,\n                pageUrl:',
                        '                worldState,\n                capabilities: ["mystic_explicit_target_v1"],\n                pageUrl:')
    text = replace_once(text, 'function applyAiAction(actionData) {', '''function explicitMysticTarget(command, actionData) {
    const player = getPlayer();
    if (![1, 2, 3].includes(command.spellIndex) || command.targetId === null ||
        String(actionData.targetId) !== command.targetId) return null;
    if (actionData.targetKind === "self") {
        return [2, 3].includes(command.spellIndex) && player.hp > 0 &&
            String(player.id) === command.targetId ? player : null;
    }
    if (actionData.targetKind !== "monster") return null;
    return getValidSpellTargets().find(entity => entity.type === "monster" &&
        entity.hp > 0 && String(entity.id) === command.targetId &&
        Math.abs(entity.mapX - player.mapX) <= 16 &&
        Math.abs(entity.mapY - player.mapY) <= 10) || null;
}

function applyAiAction(actionData) {''')
    text = replace_once(text, '    const parsedCommand = parseAiMoveCommand(actionData.move);', '''    if (actionData.stopInference) {
        stopAiLoop();
        return;
    }
    const explicit = actionData.commandSchema === "mystic_explicit_target_v1";
    if (actionData.commandSchema && !explicit) return;
    const parsedCommand = parseAiMoveCommand(actionData.move);
    if (explicit && (actionData.reset || !["move", "castSpell", "none"].includes(parsedCommand.type))) return;''')
    text = replace_once(text, '''        const target = parsedCommand.targetId
            ? getEntityById(parsedCommand.targetId)
            : getClosestSpellTarget();''', '''        const target = explicit ? explicitMysticTarget(parsedCommand, actionData) :
            (parsedCommand.targetId ? getEntityById(parsedCommand.targetId) : getClosestSpellTarget());''')
    text = replace_once(text, '        if (direction) {\n            playerFaceDirection(direction);',
                        '        if (direction && !explicit) {\n            playerFaceDirection(direction);')
    text = replace_once(text, '    aiLoopRunning = true;\n', '    aiLoopRunning = true;\n    const generation = ++aiLoopGeneration;\n')
    text = replace_once(text, '    while (aiLoopRunning) {', '    while (aiLoopRunning && generation === aiLoopGeneration) {')
    text = replace_once(text, '            applyAiAction(actionData);',
                        '            if (!aiLoopRunning || generation !== aiLoopGeneration) break;\n            applyAiAction(actionData);')
    text = replace_once(text, 'function stopAiLoop() {\n', 'function stopAiLoop() {\n    aiLoopGeneration += 1;\n')
    changed['battler.js'] = text
    for name in ('content-bridge.js', 'background.js'):
        changed[name] = replace_once(changed[name], 'worldState: message.worldState,',
                                    'worldState: message.worldState,\n        capabilities: message.capabilities,')
    manifest = {}
    diffs = []
    for name, text in changed.items():
        original_bytes = (source / name).read_bytes()
        manifest[name] = dict(original_sha256=hashlib.sha256(original_bytes).hexdigest(),
                              prepared_sha256=hashlib.sha256(text.encode()).hexdigest())
        with (output / name).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        diffs.extend(difflib.unified_diff(originals[name].splitlines(True), text.splitlines(True),
                                        fromfile=f'a/{name}', tofile=f'b/{name}'))
    (output / 'extension.patch').write_text(''.join(diffs), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
