import type { KeepAwakeMode } from "../voice/useKeepAwake";
import type { VoiceState } from "../voice/useVoiceSession";
import "./CoachPopover.css";

interface VoiceCoachPopoverProps {
  state: VoiceState;
  say: string;
  error: string | null;
  keepAwakeMode: KeepAwakeMode;
  onStop: () => void;
}

const STATE_LABEL: Record<VoiceState, string> = {
  idle: "",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking_coach: "Coach is speaking",
  sending: "Sending…",
  speaking_partner: "Partner is speaking",
  error: "Problem",
};

// Voice mode's slice of the coach popover: the same docked box the text
// "Check" uses (same CSS), showing what the spoken coach is saying while
// the draft itself lives in the chat input and the sent/partner messages
// are ordinary chat bubbles - voice mode is the chat, not a separate
// surface.
export function VoiceCoachPopover({ state, say, error, keepAwakeMode, onStop }: VoiceCoachPopoverProps) {
  return (
    <div className="coach-popover coach-popover--voice" role="status">
      <div className="coach-popover__header-buttons">
        <button type="button" className="coach-popover__close" onClick={onStop} aria-label="End voice chat" title="End voice chat">
          ✕
        </button>
      </div>
      <div className={`coach-popover__voice-state coach-popover__voice-state--${state}`}>{STATE_LABEL[state]}</div>
      {error && <div className="coach-popover__status coach-popover__status--error">{error}</div>}
      {say && (
        <div className="coach-popover__row coach-popover__voice-say">
          <span className="coach-popover__label">Coach</span>
          {say}
        </div>
      )}
      <div className="coach-popover__feedback">
        {keepAwakeMode === "dim"
          ? "Opened from the home screen, so the screen has to stay on to keep listening - turn the brightness down. For true screen-off, open this page in Safari instead."
          : "Keeps listening with the screen off. Lock the phone whenever you like; the lock screen's pause button ends the chat."}
      </div>
    </div>
  );
}
