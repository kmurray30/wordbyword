// The coach writes suggestions inside its reply as [[target language |
// English]].
// These helpers split a (possibly still streaming) reply into plain text
// and suggestion parts, and turn it into what's read aloud.

export type CoachPart =
  | { kind: "text"; text: string }
  | { kind: "suggestion"; target: string; native: string; complete: boolean };

export function parseCoachText(text: string): CoachPart[] {
  const parts: CoachPart[] = [];
  let i = 0;
  while (i < text.length) {
    const open = text.indexOf("[[", i);
    if (open === -1) {
      parts.push({ kind: "text", text: text.slice(i) });
      break;
    }
    if (open > i) parts.push({ kind: "text", text: text.slice(i, open) });
    const close = text.indexOf("]]", open + 2);
    const body = text.slice(open + 2, close === -1 ? undefined : close);
    const bar = body.indexOf("|");
    parts.push({
      kind: "suggestion",
      target: (bar === -1 ? body : body.slice(0, bar)).trim(),
      native: bar === -1 ? "" : body.slice(bar + 1).trim(),
      complete: close !== -1,
    });
    if (close === -1) break;
    i = close + 2;
  }
  return parts.filter((p) => p.kind === "suggestion" || p.text.length > 0);
}

// The suggestions in a reply, complete ones only.
export function coachSuggestions(text: string): { target: string; native: string }[] {
  return parseCoachText(text).flatMap((p) => (p.kind === "suggestion" && p.complete && p.target ? [p] : []));
}

function quoted(text: string): string {
  return `“${text.trim().replace(/[.!。！]+$/, "")}”`;
}

// What a reply sounds like: each suggestion block becomes
// “target”, which means “English”. Text from an unclosed block onward is
// held back as `rest` (it isn't speakable yet).
export function speakableCoachText(text: string): { speakable: string; rest: string } {
  let out = "";
  let i = 0;
  while (i < text.length) {
    const open = text.indexOf("[[", i);
    if (open === -1) {
      // A trailing "[" might be the start of "[[" - hold it back.
      const tail = text.endsWith("[") ? text.length - 1 : text.length;
      return { speakable: out + text.slice(i, tail), rest: text.slice(tail) };
    }
    out += text.slice(i, open);
    const close = text.indexOf("]]", open + 2);
    if (close === -1) return { speakable: out, rest: text.slice(open) };
    const body = text.slice(open + 2, close);
    const bar = body.indexOf("|");
    const target = (bar === -1 ? body : body.slice(0, bar)).trim();
    const native = bar === -1 ? "" : body.slice(bar + 1).trim();
    if (target) out += native ? `${quoted(target)}, which means ${quoted(native)}.` : `${quoted(target)}.`;
    i = close + 2;
  }
  return { speakable: out, rest: "" };
}
