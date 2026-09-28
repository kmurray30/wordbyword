import "./TranslatePopover.css";

export interface PopoverCandidate {
  translation: string;
  description?: string;
}

interface TranslatePopoverProps {
  candidates: PopoverCandidate[];
  onSelect?: (translation: string) => void;
  // "down" (default) suits a word inside a chat bubble, which has room
  // below it. "up" is for anything pinned near the bottom of the page
  // (the input box's draft-translate button) - opening downward there
  // would render off the bottom of the viewport.
  direction?: "down" | "up";
}

export function TranslatePopover({ candidates, onSelect, direction = "down" }: TranslatePopoverProps) {
  if (candidates.length === 0) return null;

  return (
    <div className={`translate-popover translate-popover--${direction}`} role="tooltip">
      <ul>
        {candidates.map((c, i) => (
          <li key={i}>
            {onSelect ? (
              <button type="button" onClick={() => onSelect(c.translation)}>
                <span className="translation">{c.translation || "…"}</span>
                {c.description && <span className="description">{c.description}</span>}
              </button>
            ) : (
              <>
                <span className="translation">{c.translation || "…"}</span>
                {c.description && <span className="description">{c.description}</span>}
              </>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
