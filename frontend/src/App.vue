<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { useTaskWorkflow } from './composables/useTaskWorkflow'

const route = useRoute()
const router = useRouter()
const workflow = useTaskWorkflow()

const navItems = [
  {
    key: 'task-create',
    label: '任务创建',
    to: '/tasks/create',
    badge: '01',
  },
  {
    key: 'postprocess',
    label: '结果与后处理',
    to: '/postprocess',
    badge: '02',
  },
  {
    key: 'statistics',
    label: '统计分析',
    to: '/statistics',
    badge: '03',
  },
  {
    key: 'history',
    label: '历史记录',
    to: '/history',
    badge: '04',
  },
]

const currentSectionKey = computed(() => String(route.meta.sectionKey ?? 'task-create'))
const currentSection = computed(
  () => navItems.find((item) => item.key === currentSectionKey.value) ?? navItems[0],
)
const routedRunId = computed(() => {
  const raw = route.params.id
  if (typeof raw === 'string' && raw.trim()) return raw
  return null
})
const isMobileViewport = ref(false)
const isMobileSidebarOpen = ref(false)
const sidebarToggleLabel = computed(() => (isMobileSidebarOpen.value ? '关闭导航' : '展开导航'))
const sidebarToggleSymbol = computed(() => (isMobileSidebarOpen.value ? '‹' : '›'))

let navigationInProgress = false

const toggleSidebar = () => {
  isMobileSidebarOpen.value = !isMobileSidebarOpen.value
}

const syncViewport = () => {
  isMobileViewport.value = window.innerWidth <= 820
  if (!isMobileViewport.value) {
    isMobileSidebarOpen.value = false
  }
}

const closeMobileSidebar = () => {
  if (isMobileViewport.value) {
    isMobileSidebarOpen.value = false
  }
}

const navigateTo = async (targetKey: string) => {
  if (navigationInProgress) return
  navigationInProgress = true
  try {
    if (targetKey === 'task-create') {
      await router.push('/tasks/create')
      closeMobileSidebar()
      return
    }

    if (targetKey === 'postprocess') {
      if (routedRunId.value) {
        await router.push(`/runs/${routedRunId.value}`)
        closeMobileSidebar()
        return
      }
      const activeId = workflow.activeRunId.value
      if (!activeId) {
        ElMessage.warning('请先在任务创建页开始处理，再进入结果展示与后处理。')
        return
      }
      await router.push(`/runs/${activeId}`)
      closeMobileSidebar()
      return
    }

    if (targetKey === 'statistics') {
      if (routedRunId.value) {
        await router.push(`/runs/${routedRunId.value}/statistics`)
        closeMobileSidebar()
        return
      }
      const activeId = workflow.activeRunId.value
      if (!activeId) {
        ElMessage.warning('当前还没有可进入统计页的任务。')
        return
      }
      await router.push(`/runs/${activeId}/statistics`)
      closeMobileSidebar()
      return
    }

    if (targetKey === 'history') {
      await router.push('/history')
      closeMobileSidebar()
    }
  } finally {
    navigationInProgress = false
  }
}

onMounted(() => {
  syncViewport()
  window.addEventListener('resize', syncViewport)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', syncViewport)
})
</script>

<template>
  <div class="app-shell" :class="{ 'is-mobile-sidebar-open': isMobileSidebarOpen }">
    <header v-if="!isMobileViewport" class="desktop-nav glass-card">
      <nav class="desktop-nav-grid" aria-label="主流程导航">
        <button
          v-for="item in navItems"
          :key="item.key"
          type="button"
          class="desktop-nav-item"
          :class="{ 'is-active': currentSectionKey === item.key }"
          :aria-current="currentSectionKey === item.key ? 'page' : undefined"
          @click="navigateTo(item.key)"
        >
          <span class="desktop-nav-badge">{{ item.badge }}</span>
          <span class="desktop-nav-label">{{ item.label }}</span>
        </button>
      </nav>
    </header>

    <button
      v-if="isMobileViewport && isMobileSidebarOpen"
      type="button"
      class="sidebar-backdrop"
      aria-label="关闭导航"
      @click="isMobileSidebarOpen = false"
    />

    <aside v-if="isMobileViewport" class="app-sidebar" :class="{ 'is-mobile-open': isMobileSidebarOpen }">
      <div class="sidebar-head">
        <div class="sidebar-brand">
          <div class="brand-copy">
            <h1>
              <span>镍基单晶</span>
              <span>SEM 分割</span>
              <span>统计平台</span>
            </h1>
          </div>
        </div>
        <button
          type="button"
          class="sidebar-toggle"
          :aria-label="sidebarToggleLabel"
          :title="sidebarToggleLabel"
          @click="toggleSidebar"
        >
          <span>{{ sidebarToggleSymbol }}</span>
        </button>
      </div>

      <nav class="sidebar-nav">
        <button
          v-for="item in navItems"
          :key="item.key"
          type="button"
          class="nav-item"
          :class="{ 'is-active': currentSectionKey === item.key }"
          :title="item.label"
          :aria-label="item.label"
          :aria-current="currentSectionKey === item.key ? 'page' : undefined"
          @click="navigateTo(item.key)"
        >
          <span class="nav-badge">{{ item.badge }}</span>
          <span class="nav-copy">
            <strong>{{ item.label }}</strong>
          </span>
        </button>
      </nav>
    </aside>

    <div class="app-workspace">
      <div v-if="isMobileViewport" class="workspace-topbar glass-card">
        <button type="button" class="mobile-nav-toggle" @click="isMobileSidebarOpen = true">菜单</button>
        <strong>{{ currentSection.label }}</strong>
      </div>
      <main class="workspace-content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<style scoped>
.app-shell {
  --shell-padding: 12px;
  --desktop-nav-offset: 104px;
  height: 100vh;
  min-height: 100vh;
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
  padding: var(--shell-padding);
  align-items: start;
  width: 100%;
  min-width: 0;
  overflow-x: clip;
  position: relative;
}

.desktop-nav {
  position: fixed;
  top: var(--shell-padding);
  left: var(--shell-padding);
  right: var(--shell-padding);
  z-index: 40;
  padding: 14px;
  border-radius: 28px;
  box-sizing: border-box;
  contain: none;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}

.desktop-nav-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.desktop-nav-item {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  min-height: 58px;
  padding: 0 16px;
  border-radius: 20px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.62);
  color: var(--ink);
  cursor: pointer;
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    box-shadow 0.2s ease,
    background 0.2s ease;
}

.desktop-nav-item:hover {
  transform: translateY(-1px);
  border-color: rgba(23, 96, 135, 0.14);
  background: rgba(255, 255, 255, 0.78);
  box-shadow: 0 10px 20px rgba(44, 32, 20, 0.05);
}

.desktop-nav-item.is-active {
  border-color: rgba(23, 96, 135, 0.16);
  background: linear-gradient(135deg, rgba(184, 90, 43, 0.1), rgba(23, 96, 135, 0.12));
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.34);
}

.desktop-nav-badge {
  width: 40px;
  height: 40px;
  flex: 0 0 40px;
  display: grid;
  place-items: center;
  border-radius: 14px;
  background: rgba(23, 96, 135, 0.08);
  border: 1px solid rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  transition:
    background 0.2s ease,
    color 0.2s ease,
    box-shadow 0.2s ease;
}

.desktop-nav-item.is-active .desktop-nav-badge {
  background: linear-gradient(135deg, #4d9cf6, #176087);
  color: #ffffff;
  box-shadow: 0 10px 20px rgba(23, 96, 135, 0.18);
}

.desktop-nav-label {
  min-width: 0;
  font-size: 17px;
  font-weight: 700;
  line-height: 1.3;
  white-space: nowrap;
}

.app-sidebar {
  display: none;
}

.sidebar-head {
  position: relative;
  padding-right: 52px;
  padding-bottom: 22px;
  border-bottom: 1px solid rgba(31, 40, 48, 0.08);
  transition: padding 0.3s cubic-bezier(0.22, 1, 0.36, 1), min-height 0.3s cubic-bezier(0.22, 1, 0.36, 1);
}

.sidebar-brand {
  display: block;
  min-width: 0;
  max-height: 200px;
  overflow: hidden;
  transform-origin: top left;
  transition:
    opacity 0.22s ease,
    transform 0.3s cubic-bezier(0.22, 1, 0.36, 1),
    max-height 0.3s cubic-bezier(0.22, 1, 0.36, 1);
}

.brand-copy {
  position: relative;
  min-width: 0;
  max-width: 100%;
  padding: 2px 8px 0 14px;
}

.brand-copy::before {
  content: '';
  position: absolute;
  left: 0;
  top: 6px;
  bottom: 6px;
  width: 5px;
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(184, 90, 43, 0.92), rgba(23, 96, 135, 0.92));
}

.sidebar-brand h1 {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 22px;
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: 0.01em;
}

.sidebar-brand h1 span {
  display: block;
  white-space: nowrap;
}

.sidebar-toggle {
  position: absolute;
  top: 2px;
  right: 0;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  display: grid;
  place-items: center;
  border-radius: 13px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.78);
  color: var(--muted);
  cursor: pointer;
  transition:
    background 0.2s ease,
    border-color 0.2s ease,
    transform 0.26s cubic-bezier(0.22, 1, 0.36, 1),
    color 0.2s ease,
    border-radius 0.26s cubic-bezier(0.22, 1, 0.36, 1);
}

.sidebar-toggle:hover {
  background: rgba(255, 255, 255, 0.92);
  border-color: rgba(23, 96, 135, 0.12);
  color: var(--accent);
  transform: translateY(-1px);
}

.sidebar-toggle span {
  font-size: 18px;
  line-height: 1;
}

.sidebar-nav {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 14px;
  flex: 1 1 auto;
  min-height: 0;
  padding-block: 6px 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 14px;
  width: 100%;
  padding: 16px 14px 16px 12px;
  border: 1px solid transparent;
  border-radius: 20px;
  background: transparent;
  color: var(--ink);
  text-align: left;
  cursor: pointer;
  transition:
    background 0.2s ease,
    border-color 0.2s ease,
    transform 0.24s ease,
    padding 0.3s cubic-bezier(0.22, 1, 0.36, 1),
    gap 0.3s cubic-bezier(0.22, 1, 0.36, 1),
    border-radius 0.3s cubic-bezier(0.22, 1, 0.36, 1);
}

.nav-item:hover {
  background: rgba(255, 255, 255, 0.62);
  border-color: rgba(23, 96, 135, 0.12);
}

.nav-item.is-active {
  background: linear-gradient(135deg, rgba(184, 90, 43, 0.12), rgba(23, 96, 135, 0.12));
  border-color: rgba(23, 96, 135, 0.18);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.32);
  position: relative;
}

.nav-item.is-active::before {
  content: '';
  position: absolute;
  left: 0;
  top: 10px;
  bottom: 10px;
  width: 4px;
  border-radius: 999px;
  background: linear-gradient(180deg, #176087, #b85a2b);
}

.nav-badge {
  width: 46px;
  height: 46px;
  display: grid;
  place-items: center;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(31, 40, 48, 0.08);
  font-size: 12px;
  font-weight: 700;
  color: var(--accent);
  transition:
    width 0.3s cubic-bezier(0.22, 1, 0.36, 1),
    height 0.3s cubic-bezier(0.22, 1, 0.36, 1),
    border-radius 0.3s cubic-bezier(0.22, 1, 0.36, 1),
    transform 0.24s ease,
    background 0.2s ease;
}

.nav-copy {
  display: flex;
  flex-direction: column;
  min-width: 0;
  max-width: 140px;
  overflow: hidden;
  opacity: 1;
  transform: translateX(0);
  transition:
    opacity 0.18s ease,
    transform 0.28s cubic-bezier(0.22, 1, 0.36, 1),
    max-width 0.28s cubic-bezier(0.22, 1, 0.36, 1);
}

.nav-copy strong {
  display: block;
  font-size: 18px;
  line-height: 1.35;
  white-space: normal;
  word-break: keep-all;
  overflow: visible;
  text-overflow: clip;
}

.app-workspace {
  min-width: 0;
  min-height: 0;
  height: 100%;
  display: flex;
  flex-direction: column;
  padding-top: var(--desktop-nav-offset);
  overflow: hidden;
  overflow-x: hidden;
}

.workspace-content {
  flex: 1 1 auto;
  min-width: 0;
  min-height: 0;
  display: grid;
  gap: 16px;
  width: 100%;
  padding: 8px 14px 14px;
  border-radius: 0 0 28px 28px;
  background: rgba(255, 249, 240, 0.34);
  box-shadow: inset 0 0 0 1px rgba(31, 40, 48, 0.05);
  overflow-y: auto;
  overflow-x: hidden;
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
}

.workspace-content::-webkit-scrollbar-thumb {
  border-radius: 999px;
}

.workspace-topbar {
  display: none;
}

.sidebar-backdrop {
  display: none;
}

@media (max-width: 1280px) {
  .desktop-nav-label {
    font-size: 15px;
  }
}

@media (max-width: 820px) {
  .desktop-nav {
    display: none;
  }

  .app-shell {
    height: auto;
  }

  .app-workspace {
    height: auto;
    padding-top: 0;
    overflow: visible;
  }

  .workspace-content {
    padding: 0;
    border-radius: 0;
    background: transparent;
    box-shadow: none;
    overflow: visible;
  }

  .app-sidebar {
    display: flex;
    flex-direction: column;
    gap: 18px;
    position: fixed;
    top: 14px;
    left: 14px;
    bottom: 14px;
    width: min(300px, calc(100vw - 28px));
    height: auto;
    max-height: none;
    padding: 14px;
    background: rgba(255, 250, 243, 0.9);
    border: 1px solid rgba(31, 40, 48, 0.08);
    border-radius: 28px;
    box-shadow: 0 14px 34px rgba(44, 32, 20, 0.08);
    backdrop-filter: blur(8px);
    transform: translateX(calc(-100% - 28px));
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    overflow: auto;
    scrollbar-width: thin;
    scrollbar-color: transparent transparent;
    z-index: 20;
  }

  .app-sidebar:hover,
  .app-sidebar:focus-within {
    scrollbar-color: rgba(31, 40, 48, 0.34) transparent;
  }

  .app-sidebar::-webkit-scrollbar {
    width: 10px;
  }

  .app-sidebar::-webkit-scrollbar-track {
    background: transparent;
  }

  .app-sidebar::-webkit-scrollbar-thumb {
    background: transparent;
    border-radius: 999px;
  }

  .app-sidebar:hover::-webkit-scrollbar-thumb,
  .app-sidebar:focus-within::-webkit-scrollbar-thumb {
    background: rgba(31, 40, 48, 0.28);
    border: 2px solid transparent;
    background-clip: padding-box;
  }

  .app-sidebar:hover::-webkit-scrollbar-thumb:hover,
  .app-sidebar:focus-within::-webkit-scrollbar-thumb:hover {
    background: rgba(31, 40, 48, 0.44);
  }

  .sidebar-head {
    display: block;
    min-height: 72px;
    padding-right: 52px;
    padding-bottom: 24px;
  }

  .sidebar-toggle {
    position: absolute;
    margin: 0;
  }

  .nav-item {
    justify-content: flex-start;
    padding: 14px 14px 14px 12px;
  }

  .brand-copy {
    max-width: 182px;
  }

  .nav-badge {
    width: 40px;
    height: 40px;
    border-radius: 13px;
  }

  .nav-copy strong {
    font-size: 15px;
    line-height: 1.22;
  }

  .app-sidebar.is-mobile-open {
    transform: translateX(0);
  }

  .sidebar-backdrop {
    display: block;
    position: fixed;
    inset: 0;
    border: none;
    background: rgba(20, 22, 26, 0.32);
    z-index: 15;
  }

  .workspace-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 14px;
  }

  .workspace-topbar strong {
    font-size: 16px;
  }

  .mobile-nav-toggle {
    padding: 10px 14px;
    border-radius: 12px;
    border: 1px solid rgba(31, 40, 48, 0.08);
    background: rgba(255, 255, 255, 0.76);
    color: var(--ink);
    font-weight: 700;
    cursor: pointer;
  }
}

@media (max-width: 360px) {
  .app-sidebar {
    top: 10px;
    left: 10px;
    bottom: 10px;
    width: min(300px, calc(100vw - 20px));
  }

  .nav-item {
    gap: 10px;
    padding: 13px 12px 13px 10px;
  }
}
</style>
