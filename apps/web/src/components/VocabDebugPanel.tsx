import type { VocabDebug, VocabDebugWord } from "../api/client";
import "./VocabDebugPanel.css";

// What the embedding search did before one partner reply (the debug view
// next to its audio button): every word it scored, how far above the pool's
// average each one sat, which cleared the cutoff and were offered to the
// model, and the random wildcard picks - plus whether the reply used them.

const STATUS_LABEL: Record<string, string> = {
  offered: "offered",
  passed: "cleared cutoff, not picked",
  cooldown: "cleared cutoff, on cooldown",
  below: "below cutoff",
  wildcard: "random pick",
};

function Row({ word }: { word: VocabDebugWord }) {
  return (
    <li className={`vocab-debug__row vocab-debug__row--${word.status}`}>
      <span className="vocab-debug__word">
        {word.lemma}
        {word.saved && <span className="vocab-debug__saved" title="From your saved word list"> ★</span>}
        {word.translation && <span className="vocab-debug__gloss"> {word.translation}</span>}
      </span>
      <span className="vocab-debug__nums">
        {word.sim !== null && word.sim !== undefined && <span title="Cosine similarity to the conversation">{word.sim.toFixed(2)}</span>}
        {word.z !== null && word.z !== undefined && <span title="Standard deviations above the pool's average">{word.z >= 0 ? "+" : ""}{word.z.toFixed(1)}σ</span>}
        <span title="How due for review (0-1)">due {word.urgency.toFixed(2)}</span>
      </span>
      <span className="vocab-debug__tags">
        <span className="vocab-debug__status">{STATUS_LABEL[word.status] ?? word.status}</span>
        {word.used && <span className="vocab-debug__used">used in reply</span>}
      </span>
    </li>
  );
}

export function VocabDebugPanel({ debug }: { debug?: VocabDebug | null }) {
  if (!debug) {
    return (
      <div className="vocab-debug">
        <p className="vocab-debug__empty">
          No vocabulary search for this message (contextual vocabulary was off, or this reply came before it).
        </p>
      </div>
    );
  }
  const offered = debug.search.filter((w) => w.status === "offered").length;
  return (
    <div className="vocab-debug">
      <div className="vocab-debug__head">
        Embedding search - cutoff {debug.cutoff_z.toFixed(1)}σ · {offered} offered
      </div>
      {!debug.searched && (
        <p className="vocab-debug__empty">Search didn't run{debug.error ? `: ${debug.error}` : "."}</p>
      )}
      {debug.search.length > 0 && (
        <ul className="vocab-debug__list">
          {debug.search.map((w) => (
            <Row key={`s-${w.lemma}`} word={w} />
          ))}
        </ul>
      )}
      <div className="vocab-debug__head vocab-debug__head--sub">Random wildcards</div>
      {debug.wildcards.length > 0 ? (
        <ul className="vocab-debug__list">
          {debug.wildcards.map((w) => (
            <Row key={`w-${w.lemma}`} word={w} />
          ))}
        </ul>
      ) : (
        <p className="vocab-debug__empty">None picked.</p>
      )}
    </div>
  );
}
