import { useVoiceSession } from "../voice/useVoiceSession";
import type { VoiceState } from "../voice/useVoiceSession";
import "./VoicePanel.css";

interface VoicePanelProps {
  sessionId: string;
  onSend: (text: string) => Promise<{ text: string } | null>;
  partnerVoice?: string;
}

const STATE_LABEL: Record<VoiceState, string> = {
  idle: "",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking_coach: "Coach",
  sending: "Sending…",
  speaking_partner: "Partner",
  error: "Problem",
};

export function VoicePanel({ sessionId, onSend, partnerVoice }: VoicePanelProps) {
  const voice = useVoiceSession({ sessionId, onSend, partnerVoice });
  const active = voice.state !== "idle";
  const dim = active && voice.keepAwakeMode === "dim";

  return (
    <div className={`voice-panel${active ? " voice-panel--active" : ""}`}>
      <div className="voice-panel__bar">
        <button
          type="button"
          className={`voice-panel__toggle${active ? " voice-panel__toggle--active" : ""}`}
          onClick={() => (active ? voice.stop() : void voice.start())}
        >
          {active ? "■ End voice chat" : "🗣 Start voice chat"}
        </button>
        {active && (
          <span className={`voice-panel__state voice-panel__state--${voice.state}`}>{STATE_LABEL[voice.state]}</span>
        )}
      </div>
      {active && (
        <div className="voice-panel__body">
          {voice.interim && <div className="voice-panel__interim">{voice.interim}</div>}
          {voice.error && <div className="voice-panel__error">{voice.error}</div>}
          {voice.dialog.slice(-4).map((turn, i) => (
            <div key={i} className={`voice-panel__turn voice-panel__turn--${turn.role}`}>
              <span className="voice-panel__who">{turn.role === "coach" ? "Coach" : "You"}</span>
              {turn.text}
            </div>
          ))}
          {voice.coachSay && voice.state !== "listening" && (
            <div className="voice-panel__turn voice-panel__turn--coach voice-panel__turn--live">
              <span className="voice-panel__who">Coach</span>
              {voice.coachSay}
            </div>
          )}
          {voice.draft && (
            <div className="voice-panel__draft">
              <span className="voice-panel__who">Ready to send</span>
              {voice.draft}
            </div>
          )}
          <div className="voice-panel__hint">
            {voice.keepAwakeMode === "dim"
              ? "Opened from the home screen, so the screen has to stay on to keep listening - turn the brightness down. For true screen-off, open this page in Safari instead."
              : "Keeps listening with the screen off. Lock the phone whenever you like; the lock screen's pause button ends the chat."}
          </div>
        </div>
      )}
      {dim && (
        <div className="voice-dim" role="status">
          <div className="voice-dim__state">{STATE_LABEL[voice.state]}</div>
          {voice.interim && <div className="voice-dim__text">{voice.interim}</div>}
          {!voice.interim && voice.coachSay && <div className="voice-dim__text">{voice.coachSay}</div>}
          <button type="button" className="voice-dim__end" onClick={() => voice.stop()}>
            End voice chat
          </button>
        </div>
      )}
    </div>
  );
}
