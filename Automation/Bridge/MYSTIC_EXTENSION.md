# Install the Mystic browser command update

`extension_update_required` means snapshots reached inference but their
`capabilities` list does not include `mystic_explicit_target_v1`. It is not a
checkpoint load failure. Updating only the Python repository is insufficient:
the loaded browser extension must also contain the command changes.

Pull the latest JS plugin in its own repository. The update includes
`battler.js`, `content-bridge.js`, and `background.js`: the page script implements
explicit targets and stop handling, and both bridge scripts forward capabilities.

1. Open `chrome://extensions`, locate the extension, and click Reload. Ensure
   this is the unpacked directory you just updated.
2. Refresh the game tab to replace the already-injected page script.
3. Launch inference from the RL repository:

   ```powershell
   .\RunInference.ps1 -Checkpoint "runs\mystic_sim\checkpoint_07.pt" -AssumeSimParity
   ```

4. Enable the game's extension AI loop. Successful startup prints
   `Live snapshots accepted; policy actions are flowing.`

The launcher starts and stops Python and the WebSocket bridge automatically.
Browser reload is required once after installing the update. A new run also
requires restarting the AI loop if the preceding run stopped it.

The Windows Proactor warning is separate from this capability mismatch. The
bridge entry point now selects the Windows selector event loop required by ZMQ.
