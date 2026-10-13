import { parseCoachText } from "../lib/coachText";
import { ReadingText } from "./ReadingText";
import "./CoachText.css";

// A coach reply: plain text, with each [[target | English]] suggestion
// shown as a card (the target-language text shown with its readings, see
// ReadingText). `onUse` adds a "Use this" button inside the card;
// `usedText` marks the card whose text is in the message box right now.
export function CoachText({
  text,
  streaming = false,
  onUse,
  usedText,
}: {
  text: string;
  streaming?: boolean;
  onUse?: (text: string) => void;
  usedText?: string | null;
}) {
  const parts = parseCoachText(text);
  if (!parts.length) return <span className="coach-text__pending">{streaming ? "…" : ""}</span>;
  return (
    <span className="coach-text">
      {parts.map((part, i) => {
        if (part.kind === "text") return <span key={i}>{part.text}</span>;
        const used = !!usedText && part.complete && usedText.trim() === part.target.trim();
        return (
          <span
            key={i}
            className={`coach-text__suggestion${part.complete ? "" : " coach-text__suggestion--streaming"}${
              used ? " coach-text__suggestion--used" : ""
            }`}
          >
            <span className="coach-text__suggestion-body">
              <span className="coach-text__target">
                {part.complete && part.target ? <ReadingText text={part.target} /> : part.target || "…"}
              </span>
              {part.native && <span className="coach-text__native">{part.native}</span>}
            </span>
            {onUse && part.complete && part.target && (
              <button
                type="button"
                className="coach-text__use"
                onClick={() => onUse(part.target)}
                disabled={used}
                aria-label={used ? "In your message" : `Use “${part.target}”`}
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
