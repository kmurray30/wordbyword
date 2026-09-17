import "./TranslatePopover.css";

export interface PopoverCandidate {
  translation: string;
  description?: string;
}

interface TranslatePopoverProps {
  candidates: PopoverCandidate[];
  onSelect?: (translation: string) => void;
}

export function TranslatePopover({ candidates, onSelect }: TranslatePopoverProps) {
  if (candidates.length === 0) return null;

  return (
    <div className="translate-popover" role="tooltip">
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
