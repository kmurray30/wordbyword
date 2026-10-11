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

// iOS home-screen web apps lose the microphone when the screen locks; a
// plain Safari tab keeps it. Only worth saying when it applies.
const standalone = typeof navigator !== "undefined" && (navigator as { standalone?: boolean }).standalone === true;

export function VoicePanel({ sessionId, onSend, partnerVoice }: VoicePanelProps) {
  const voice = useVoiceSession({ sessionId, onSend, partnerVoice });
  const active = voice.state !== "idle";

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
          {standalone && (
            <div className="voice-panel__hint">For screen-off use, open this page in Safari rather than from the home-screen icon.</div>
          )}
        </div>
      )}
    </div>
  );
}
