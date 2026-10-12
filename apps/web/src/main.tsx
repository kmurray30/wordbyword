import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { LlmGate } from './components/LlmGate.tsx'
import { trackVisibleViewport } from './lib/visibleViewport'

trackVisibleViewport()

// iOS Safari ignores user-scalable=no for pinch-zoom; its own gesture
// events are the only way to stop it.
for (const type of ['gesturestart', 'gesturechange', 'gestureend']) {
  document.addEventListener(type, (e) => e.preventDefault(), { passive: false })
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LlmGate>
      <App />
    </LlmGate>
  </StrictMode>,
)
