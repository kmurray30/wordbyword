import { parseCoachText } from "../lib/coachText";
import "./CoachText.css";

// A coach reply: plain text, with each [[Spanish | English]] suggestion
// shown as a card. `onUse` adds a "Use this" button inside the card;
// `usedText` marks the card whose Spanish is in the message box right now.
export function CoachText({
  text,
  streaming = false,
  onUse,
  usedText,
}: {
  text: string;
  streaming?: boolean;
  onUse?: (spanish: string) => void;
  usedText?: string | null;
}) {
  const parts = parseCoachText(text);
  if (!parts.length) return <span className="coach-text__pending">{streaming ? "…" : ""}</span>;
  return (
    <span className="coach-text">
      {parts.map((part, i) => {
        if (part.kind === "text") return <span key={i}>{part.text}</span>;
        const used = !!usedText && part.complete && usedText.trim() === part.spanish.trim();
        return (
          <span
            key={i}
            className={`coach-text__suggestion${part.complete ? "" : " coach-text__suggestion--streaming"}${
              used ? " coach-text__suggestion--used" : ""
            }`}
          >
            <span className="coach-text__suggestion-body">
              <span className="coach-text__es">{part.spanish || "…"}</span>
              {part.english && <span className="coach-text__en">{part.english}</span>}
            </span>
            {onUse && part.complete && part.spanish && (
              <button
                type="button"
                className="coach-text__use"
                onClick={() => onUse(part.spanish)}
                disabled={used}
                aria-label={used ? "In your message" : `Use “${part.spanish}”`}
              >
                {used ? "✓ In message" : "Use this"}
              </button>
            )}
          </span>
        );
      })}
    </span>
  );
}
