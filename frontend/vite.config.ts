import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import ElementPlus from 'unplugin-element-plus/vite'

export default defineConfig({
  plugins: [
    vue(),
    ElementPlus(),
    AutoImport({
      dts: './auto-imports.d.ts',
      resolvers: [ElementPlusResolver()],
    }),
    Components({
      dts: './components.d.ts',
      resolvers: [ElementPlusResolver()],
    }),
  ],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    target: 'es2020',
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (id.includes('echarts')) return 'echarts'
          if (id.includes('element-plus')) return 'element-plus'
          if (id.includes('vue') || id.includes('pinia') || id.includes('vue-router')) return 'vue-vendor'
          if (id.includes('axios')) return 'http'
        },
      },
    },
  },
  optimizeDeps: {
    include: ['vue', 'vue-router', 'axios'],
    exclude: [],
  },
})
