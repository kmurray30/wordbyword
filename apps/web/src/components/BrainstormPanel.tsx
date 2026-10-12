import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { BrainstormLogTurn } from "../api/client";
import type { KeepAwakeMode } from "../voice/useKeepAwake";
import type { VoiceState } from "../voice/useVoiceSession";
import "./BrainstormPanel.css";

const VOICE_STATE_LABEL: Record<VoiceState, string> = {
  idle: "",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking_coach: "Coach is speaking",
  sending: "Sending…",
  speaking_partner: "Partner is speaking",
  error: "Problem",
};

interface BrainstormPanelProps {
  turns: BrainstormLogTurn[];
  live: { text: string; draft: string } | null;
  error: string | null;
  value: string;
  onChange: (value: string) => void;
  onAsk: () => void;
  canCheckReply: boolean;
  onCheckReply: () => void;
  onUseDraft: (draft: string) => void;
  voiceState: VoiceState;
  voiceError: string | null;
  interim: string;
  keepAwakeMode: KeepAwakeMode;
  onMicClick: () => void;
}

function DraftChip({ draft, onUse }: { draft: string; onUse?: () => void }) {
  return onUse ? (
    <button type="button" className="brainstorm__draft" onClick={onUse} title="Put this back in the Reply box">
      ↓ {draft}
    </button>
  ) : (
    <span className="brainstorm__draft">↓ {draft}</span>
  );
}

// The Brainstorm half of the composer: the open side-chat log with the
// coach (since the last sent reply), its input, and the mic - everything
// spoken in voice mode lands here too. Kept visually apart from the main
// chat; once a reply is sent, App folds this log into that reply's notes.
export function BrainstormPanel({
  turns,
  live,
  error,
  value,
  onChange,
  onAsk,
  canCheckReply,
  onCheckReply,
  onUseDraft,
  voiceState,
  voiceError,
  interim,
  keepAwakeMode,
  onMicClick,
}: BrainstormPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const logRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const voiceActive = voiceState !== "idle";
  const hasLog = turns.length > 0 || live !== null;

  // Newest line in view as the log grows or the coach's answer streams.
  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [turns.length, live?.text, collapsed]);

  // Grow with the text (spoken utterances can run long), up to the CSS cap.
  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [value]);

  return (
    <section className="brainstorm" aria-label="Brainstorm with the coach">
      <div className="brainstorm__header">
        <span className="brainstorm__title">Brainstorm</span>
        {voiceActive && (
          <span className={`brainstorm__voice-state brainstorm__voice-state--${voiceState}`} role="status">
            {VOICE_STATE_LABEL[voiceState]}
          </span>
        )}
        <span className="brainstorm__spacer" />
        {canCheckReply && (
          <button type="button" className="brainstorm__chip" onClick={onCheckReply}>
            Check my reply
          </button>
        )}
        {turns.length > 0 && (
          <button
            type="button"
            className="brainstorm__chip brainstorm__chip--quiet"
            onClick={() => setCollapsed((c) => !c)}
            aria-expanded={!collapsed}
          >
            {collapsed ? `Show (${turns.length})` : "Hide"}
          </button>
        )}
      </div>

      {hasLog && !collapsed && (
        <div className="brainstorm__log" ref={logRef}>
          {turns.map((turn) => (
            <div key={turn.id} className={`brainstorm__turn brainstorm__turn--${turn.role}`}>
              {turn.text && <span>{turn.text}</span>}
              {turn.draft && <DraftChip draft={turn.draft} onUse={() => onUseDraft(turn.draft)} />}
            </div>
          ))}
          {live && (
            <div className="brainstorm__turn brainstorm__turn--coach brainstorm__turn--live">
              <span>{live.text || "…"}</span>
              {live.draft && <DraftChip draft={live.draft} />}
            </div>
          )}
        </div>
      )}

      {(error || voiceError) && <div className="brainstorm__error">{voiceError ?? error}</div>}
      {voiceActive && (
        <div className="brainstorm__hint">
          {keepAwakeMode === "dim"
            ? "Opened from the home screen, so the screen has to stay on to keep listening - turn the brightness down. For true screen-off, open this page in Safari instead."
            : "Keeps listening with the screen off. Say \"send it\" to send the Reply; the lock screen's pause button ends voice."}
        </div>
      )}

      <div className="brainstorm__row">
        <div className="brainstorm__field">
          {interim && <div className="brainstorm__mic-pill">{interim}</div>}
          <textarea
            ref={textareaRef}
            className="brainstorm__textarea"
            rows={1}
            value={value}
            placeholder="Think out loud, ask anything - English is fine"
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onAsk();
              }
            }}
          />
        </div>
        <button
          type="button"
          className={`brainstorm__mic${voiceActive ? " brainstorm__mic--active" : ""}${voiceState === "listening" ? " brainstorm__mic--listening" : ""}`}
          onClick={onMicClick}
          aria-pressed={voiceActive}
          aria-label={voiceActive ? "End voice" : "Start voice"}
          title={voiceActive ? "End voice" : "Talk it through out loud - the coach answers and fills in your Reply"}
        >
          {voiceActive ? "⏹" : "🎤"}
        </button>
        <button type="button" className="brainstorm__ask" onClick={onAsk} disabled={!value.trim() || live !== null}>
          Ask
        </button>
      </div>
    </section>
  );
}
