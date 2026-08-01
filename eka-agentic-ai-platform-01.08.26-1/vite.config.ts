import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      port: 3000,
      host: '0.0.0.0',
      proxy: {
        '/api': {
          target: process.env.VITE_API_URL || 'http://localhost:3001',
          changeOrigin: true,
          secure: false,
        },
      },
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify — file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {
        // Only watch source files — ignore ALL runtime-generated output.
        ignored: (filePath: string) => {
          // Always watch src/ and config files
          if (filePath.includes('/src/') || filePath.includes('\\src\\')) return false;
          // Ignore dedicated runtime directories
          if (/[/\\](app-log|app-output|workflow-logs|app-config|\.workflows|Eka-Agentic)[/\\]/.test(filePath)) return true;
          // Allow root-level config files (.ts, .tsx, .css, .json, .html) not in node_modules
          if (/\.(ts|tsx|css|json|html)$/.test(filePath) && !/node_modules/.test(filePath)) return false;
          // Ignore everything else
          return true;
        },
      },
    },
  };
});
