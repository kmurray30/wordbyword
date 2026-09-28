import { AudioButton } from "./AudioButton";
import "./TranslationRow.css";

interface TranslationRowProps {
  variant: "assistant" | "user-native" | "user-target";
  label: string;
  text: string | null;
  loading: boolean;
  language: string;
  voice?: string;
}

// One translation line under a chat bubble - e.g. "EN: I like reading."
export function TranslationRow({ variant, label, text, loading, language, voice }: TranslationRowProps) {
  if (loading) {
    return (
      <div className={`translation-row translation-row--${variant}`}>
        <span className="translation-row__badge">{label}</span>
        <span className="translation-row__loading">Translating…</span>
      </div>
    );
  }

  if (text === null) return null;

  return (
    <div className={`translation-row translation-row--${variant}`}>
      <span className="translation-row__badge">{label}</span>
      <span className="translation-row__text">{text || "…"}</span>
      {text && <AudioButton text={text} language={language} voice={voice} label={`Play ${label} audio`} />}
    </div>
  );
}
