import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { LlmGate } from './components/LlmGate.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LlmGate>
      <App />
    </LlmGate>
  </StrictMode>,
)
