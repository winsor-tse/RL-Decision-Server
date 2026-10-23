// Executes the actual prepared/installed browser functions with a controlled registry.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
const directory = process.argv[2];
const source = fs.readFileSync(path.join(directory, 'battler.js'), 'utf8');
let player = {id:42, hp:10, mapX:50, mapY:50, isCurrentPlayer:true};
let monsters = [{id:42, hp:10, mapX:51, mapY:50, isCurrentPlayer:false, type:'monster'}];
let calls = [];
const pending = [];
const sandbox = {
    console: {log(){}, warn(){}, error(){}},
    getPlayer:()=>player, getRegistry:()=>({entities:[player,...monsters]}),
    getNetwork:()=>({playerSpellCast:(...args)=>calls.push(['cast',...args])}),
    getDistance:()=>1, getDirection:()=> 'left',
    movePlayerDirection:direction=>calls.push(['move',direction]),
    playerFaceDirection:direction=>calls.push(['face',direction]),
    buildWorldState:()=>({}), sendAiTick:()=>new Promise(resolve=>pending.push(resolve)),
    delay:()=>Promise.resolve(), printPlayerProperties:()=>{},
};
vm.createContext(sandbox);
vm.runInContext('let aiLoopRunning=false; let aiLoopStarted=false; let aiLoopGeneration=0; const AI_LOOP_DELAY_MS=200;\n' +
    source.slice(source.indexOf('function getValidSpellTargets()'), source.indexOf('window.__aiBattler =')), sandbox);
const command = (slot, kind, id=42)=>({move:`castSpell:${slot}:${id}`, targetId:id,
    targetKind:kind, commandSchema:'mystic_explicit_target_v1', reset:false});
for (const direction of ['up','down','left','right']) sandbox.applyAiAction({move:direction, commandSchema:'mystic_explicit_target_v1'});
assert.deepEqual(calls, ['up','down','left','right'].map(d=>['move',d]));
calls=[];
for (const slot of [1,2,3]) sandbox.applyAiAction(command(slot,'monster'));
assert.deepEqual(calls, [[ 'cast',1,42],['cast',2,42],['cast',3,42]]); // No auto-facing.
calls=[]; monsters=[];
for (const slot of [2,3]) sandbox.applyAiAction(command(slot,'self'));
assert.deepEqual(calls, [['cast',2,42],['cast',3,42]]);
calls=[];
sandbox.applyAiAction(command(1,'self'));
sandbox.applyAiAction(command(2,'monster'));
sandbox.applyAiAction({move:null, commandSchema:'mystic_explicit_target_v1'});
assert.equal(calls.length,0); // Never substitute another target.
monsters=[{id:7,hp:10,mapX:67,mapY:50,type:'monster'}];
sandbox.applyAiAction(command(1,'monster',7));
assert.equal(calls.length,0);
monsters[0].mapX=66; monsters[0].mapY=60;
sandbox.applyAiAction(command(1,'monster',7));
assert.deepEqual(calls,[['cast',1,7]]);
calls=[];
(async()=>{
    const first=sandbox.runAiLoop();
    sandbox.stopAiLoop();
    const second=sandbox.runAiLoop();
    pending.shift()({move:'up'}); // Old generation cannot apply after restart.
    await first;
    assert.equal(calls.length,0);
    pending.shift()({stopInference:true,move:'up'});
    await second;
    assert.equal(calls.length,0);
    for (const name of ['content-bridge.js','background.js'])
        assert.match(fs.readFileSync(path.join(directory,name),'utf8'), /capabilities: message.capabilities/);
    console.log('Browser command, target, facing, stop/restart and capability checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1});
