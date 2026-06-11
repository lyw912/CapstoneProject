import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/static/signal-studio/',
  plugins: [react()],
  build: {
    outDir: '../static/signal-studio',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: 'assets/app.js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name][extname]',
        manualChunks: {
          antd: ['antd', '@ant-design/icons'],
          editor: ['@tiptap/react', '@tiptap/starter-kit', '@tiptap/extension-highlight', '@tiptap/extension-link', '@tiptap/extension-placeholder', '@tiptap/extension-underline'],
          visuals: ['recharts', 'framer-motion']
        }
      }
    }
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:5000'
    }
  }
});
