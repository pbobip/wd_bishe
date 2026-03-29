<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const SIDEBAR_STORAGE_KEY = 'wd_bishe_sidebar_collapsed'

const navItems = [
  {
    key: 'projects',
    label: '项目创建',
    to: '/projects',
    badge: '01',
  },
  {
    key: 'task-create',
    label: '任务创建',
    to: '/tasks/create',
    badge: '02',
  },
  {
    key: 'postprocess',
    label: '后处理',
    to: '/postprocess',
    badge: '03',
  },
  {
    key: 'results',
    label: '结果展示',
    to: '/results',
    badge: '04',
  },
  {
    key: 'history',
    label: '历史记录',
    to: '/history',
    badge: '05',
  },
]

const currentSectionKey = computed(() => String(route.meta.sectionKey ?? 'projects'))
const currentSection = computed(
  () => navItems.find((item) => item.key === currentSectionKey.value) ?? navItems[0],
)
const isSidebarCollapsed = ref(false)
const isMobileViewport = ref(false)
const isMobileSidebarOpen = ref(false)
const effectiveSidebarCollapsed = computed(() => isSidebarCollapsed.value && !isMobileViewport.value)
const sidebarToggleLabel = computed(() => {
  if (isMobileViewport.value) {
    return isMobileSidebarOpen.value ? '关闭导航' : '展开侧栏'
  }
  return effectiveSidebarCollapsed.value ? '展开侧栏' : '收起侧栏'
})
const sidebarToggleSymbol = computed(() => {
  if (isMobileViewport.value) {
    return isMobileSidebarOpen.value ? '‹' : '›'
  }
  return effectiveSidebarCollapsed.value ? '›' : '‹'
})

const toggleSidebar = () => {
  if (isMobileViewport.value) {
    isMobileSidebarOpen.value = !isMobileSidebarOpen.value
    return
  }
  isSidebarCollapsed.value = !isSidebarCollapsed.value
}

const syncViewport = () => {
  isMobileViewport.value = window.innerWidth <= 820
  if (!isMobileViewport.value) {
    isMobileSidebarOpen.value = false
  }
}

const navigateTo = (to: string) => {
  router.push(to)
  if (isMobileViewport.value) {
    isMobileSidebarOpen.value = false
  }
}

onMounted(() => {
  try {
    const cached = window.localStorage.getItem(SIDEBAR_STORAGE_KEY)
    if (cached === '1') {
      isSidebarCollapsed.value = true
    }
  } catch {
    // ignore local storage failures
  }
  syncViewport()
  window.addEventListener('resize', syncViewport)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', syncViewport)
})

watch(isSidebarCollapsed, (value) => {
  try {
    window.localStorage.setItem(SIDEBAR_STORAGE_KEY, value ? '1' : '0')
  } catch {
    // ignore local storage failures
  }
})
</script>

<template>
  <div
    class="app-shell"
    :class="{
      'is-sidebar-collapsed': effectiveSidebarCollapsed,
      'is-mobile-sidebar-open': isMobileSidebarOpen,
    }"
  >
    <button
      v-if="isMobileViewport && isMobileSidebarOpen"
      type="button"
      class="sidebar-backdrop"
      aria-label="关闭导航"
      @click="isMobileSidebarOpen = false"
    />

    <aside class="app-sidebar" :class="{ 'is-collapsed': effectiveSidebarCollapsed, 'is-mobile-open': isMobileSidebarOpen }">
      <div class="sidebar-head">
        <div class="sidebar-brand">
          <div class="brand-copy">
            <h1>
              <span>镍基单晶</span>
              <span>SEM 分割</span>
              <span>统计平台</span>
            </h1>
            <p>
              <span>批量导入与双路线分割</span>
              <span>结果比对和统计导出</span>
            </p>
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
          @click="navigateTo(item.to)"
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
  --sidebar-width: clamp(204px, 13vw, 236px);
  min-height: 100vh;
  display: grid;
  grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
  gap: 12px;
  padding: 12px;
  align-items: start;
  width: 100%;
  min-width: 0;
  overflow-x: clip;
}

.app-shell.is-sidebar-collapsed {
  --sidebar-width: 80px;
}

.app-sidebar {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: calc(100vh - 28px);
  padding: 16px 14px;
  background: rgba(255, 250, 243, 0.84);
  border: 1px solid rgba(31, 40, 48, 0.08);
  border-radius: 28px;
  box-shadow: 0 14px 34px rgba(44, 32, 20, 0.08);
  backdrop-filter: blur(8px);
  position: sticky;
  top: 14px;
  max-height: calc(100vh - 28px);
  overflow: auto;
  scrollbar-width: thin;
  scrollbar-color: transparent transparent;
  box-sizing: border-box;
  scrollbar-gutter: stable;
  transition: padding 0.2s ease, box-shadow 0.2s ease;
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
  position: relative;
  padding-right: 52px;
  padding-bottom: 20px;
  border-bottom: 1px solid rgba(31, 40, 48, 0.08);
}

.sidebar-brand {
  display: block;
  min-width: 0;
}

.brand-copy {
  position: relative;
  min-width: 0;
  max-width: 168px;
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
  font-size: 18px;
  font-weight: 800;
  line-height: 1.08;
  letter-spacing: 0.01em;
}

.sidebar-brand h1 span {
  display: block;
  white-space: nowrap;
}

.sidebar-brand p {
  margin: 14px 0 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.55;
}

.sidebar-brand p span {
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
  transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease, color 0.18s ease;
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
  gap: 12px;
  flex: 1 1 auto;
  min-height: 0;
  padding-block: 4px 2px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  padding: 14px 14px 14px 12px;
  border: 1px solid transparent;
  border-radius: 18px;
  background: transparent;
  color: var(--ink);
  text-align: left;
  cursor: pointer;
  transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
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
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(31, 40, 48, 0.08);
  font-size: 12px;
  font-weight: 700;
  color: var(--accent);
}

.nav-copy {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.nav-copy strong {
  display: block;
  font-size: 16px;
  line-height: 1.3;
  white-space: normal;
  word-break: keep-all;
  overflow: visible;
  text-overflow: clip;
}

.app-workspace {
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow-x: hidden;
}

.workspace-content {
  min-width: 0;
  display: grid;
  gap: 16px;
  width: 100%;
  overflow-x: hidden;
}

.workspace-topbar {
  display: none;
}

.sidebar-backdrop {
  display: none;
}

.app-sidebar.is-collapsed {
  padding: 12px 10px 16px;
}

.app-sidebar.is-collapsed .sidebar-head {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 48px;
  padding: 0 0 14px;
}

.app-sidebar.is-collapsed .sidebar-brand {
  display: none;
}

.app-sidebar.is-collapsed .sidebar-toggle {
  position: static;
  margin: 0 auto;
  width: 40px;
  height: 40px;
  flex: 0 0 40px;
  border-radius: 16px;
}

.app-sidebar.is-collapsed .nav-item {
  justify-content: center;
  padding: 12px 8px;
}

.app-sidebar.is-collapsed .nav-copy {
  display: none;
}

.app-sidebar.is-collapsed .nav-badge {
  width: 46px;
  height: 46px;
}

@media (max-width: 1280px) {
  .app-shell {
    --sidebar-width: 92px;
  }

  .sidebar-head {
    padding-right: 0;
  }

  .sidebar-brand {
    align-items: center;
  }

  .brand-copy,
  .nav-copy {
    display: none;
  }

  .sidebar-toggle {
    position: static;
    margin: 0 auto;
  }

  .nav-item {
    justify-content: center;
    padding: 12px 8px;
  }

}

@media (max-width: 820px) {
  .app-shell {
    grid-template-columns: 1fr;
  }

  .app-sidebar {
    position: fixed;
    top: 14px;
    left: 14px;
    bottom: 14px;
    width: min(300px, calc(100vw - 28px));
    height: auto;
    max-height: none;
    padding: 14px;
    transform: translateX(calc(-100% - 28px));
    transition: transform 0.22s ease, box-shadow 0.22s ease;
  }

  .sidebar-head {
    padding-right: 52px;
  }

  .brand-copy,
  .nav-copy {
    display: block;
  }

  .brand-copy {
    max-width: 182px;
  }

  .sidebar-toggle {
    position: absolute;
    margin: 0;
  }

  .nav-item {
    justify-content: flex-start;
    padding: 14px 14px 14px 12px;
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

  .app-sidebar.is-collapsed .sidebar-head {
    padding-right: 52px;
  }

  .app-sidebar.is-collapsed .brand-copy,
  .app-sidebar.is-collapsed .nav-copy {
    display: block;
  }

  .app-sidebar.is-collapsed .nav-item {
    justify-content: flex-start;
    padding: 14px 14px 14px 12px;
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

  .nav-item,
  .app-sidebar.is-collapsed .nav-item {
    gap: 10px;
    padding: 13px 12px 13px 10px;
  }
}
</style>
