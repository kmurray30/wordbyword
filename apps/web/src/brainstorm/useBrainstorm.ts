import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { BrainstormLogTurn, BrainstormStreamEvent, BrainstormTurnRequest } from "../api/client";

// Local ids for turns added optimistically on this page (the server's own
// ids only matter once a reload re-fetches the log).
let nextLocalId = -1;

// Only `text` is needed - the rest have server-side defaults.
export type BrainstormAsk = Pick<BrainstormTurnRequest, "text"> &
  Partial<Omit<BrainstormTurnRequest, "session_id" | "text">>;

// The open Brainstorm log - the learner's side chat with the coach about
// their next reply, typed or spoken. The server is the source of truth (it
// persists every turn and stamps the log onto the reply when it's sent);
// this mirrors it for display and streams new turns. Voice mode feeds its
// utterances through the same `stream`, so both inputs share one log.
export function useBrainstorm(sessionId: string) {
  const [turns, setTurns] = useState<BrainstormLogTurn[]>([]);
  const turnsRef = useRef<BrainstormLogTurn[]>([]);
  // The coach's answer while it's still streaming in.
  const [live, setLive] = useState<{ text: string; draft: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The latest draft the coach judged ready to send, if any - the Reply's
  // Send button lights up while the Reply box still holds exactly this.
  const [readyDraft, setReadyDraft] = useState<string | null>(null);

  const update = (next: BrainstormLogTurn[]) => {
    turnsRef.current = next;
    setTurns(next);
  };

  const reload = useCallback(() => {
    api
      .brainstormOpen(sessionId)
      .then((res) => update(res.turns))
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    reload();
  }, [reload]);

  // Adds the learner's line, streams the coach's answer into `live`, and
  // re-yields every event so the caller can act on intent/verdict/draft
  // (fill the Reply box, send, speak...). The finished coach turn joins
  // the log when the stream ends.
  const stream = useCallback(
    async function* (ask: BrainstormAsk): AsyncGenerator<BrainstormStreamEvent> {
      update([...turnsRef.current, { id: nextLocalId--, role: "learner", text: ask.text.trim(), draft: "" }]);
      setLive({ text: "", draft: "" });
      setError(null);
      let say = "";
      let draft = "";
      let verdict = "";
      try {
        for await (const event of api.brainstormStream({ words: [], reply_draft: "", source: "text", ...ask, session_id: sessionId })) {
          if (event.type === "say_chunk") {
            say += event.data.delta;
            setLive({ text: say, draft });
          } else if (event.type === "say_complete") {
            say = event.data.say;
            setLive({ text: say, draft });
          } else if (event.type === "verdict") {
            verdict = event.data.verdict;
          } else if (event.type === "draft") {
            draft = event.data.draft;
            setLive({ text: say, draft });
            setReadyDraft(draft && verdict === "ready" ? draft : null);
          } else if (event.type === "error") {
            throw new Error(event.data.message);
          }
          yield event;
        }
        update([...turnsRef.current, { id: nextLocalId--, role: "coach", text: say, draft }]);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        throw err;
      } finally {
        setLive(null);
      }
    },
    [sessionId],
  );

  // Sending a reply closes the open log (the server stamps it onto that
  // reply) - hands the log over for the reply's bubble and starts fresh.
  const takeLog = useCallback((): BrainstormLogTurn[] => {
    const log = turnsRef.current;
    update([]);
    setReadyDraft(null);
    setError(null);
    return log;
  }, []);

  return { turns, live, error, readyDraft, stream, takeLog, reload };
}
