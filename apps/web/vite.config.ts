import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import type { Plugin } from 'vite'

// Every build gets its own version, baked into the JS as __APP_VERSION__
// and published next to it as /version.json. An app that has been sitting
// in the background (iOS resumes a home-screen app or tab without
// reloading it) compares the two when it comes back and updates itself -
// see src/lib/useAppUpdate.ts.
const APP_VERSION = new Date().toISOString()

function versionFile(): Plugin {
  return {
    name: 'version-file',
    apply: 'build',
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'version.json', source: JSON.stringify({ version: APP_VERSION }) })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), versionFile()],
  define: { __APP_VERSION__: JSON.stringify(APP_VERSION) },
})
