import App from "./App";
import { setLanguage, useLanguage } from "./lib/language";

// Each language is its own conversation: switching starts the app afresh
// on that language's session.
export function Root() {
  const lang = useLanguage();
  return <App key={lang} lang={lang} onLanguageChange={setLanguage} />;
}
