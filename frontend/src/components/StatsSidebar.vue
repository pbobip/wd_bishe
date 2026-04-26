<script setup lang="ts">
import JSZip from 'jszip'
import { ElMessage } from 'element-plus'
import { computed, ref, watch } from 'vue'

const props = defineProps<{
  exports: Array<{ id: number; kind: string; url: string; path: string }>
}>()

const EXPORT_KIND_META: Record<string, { label: string; note: string; imageKind?: boolean }> = {
  xlsx: {
    label: '统计总表 XLSX',
    note: '图像级主统计表',
  },
  batch_xlsx: {
    label: '批次汇总 XLSX',
    note: '批量任务均值与区间汇总',
  },
  particles_xlsx: {
    label: '颗粒明细 XLSX',
    note: '对象级颗粒统计明细',
  },
  config: {
    label: '配置快照 JSON',
    note: '本次任务参数记录',
  },
  overlay: {
    label: '分割叠加图',
    note: '原图 + 分割掩膜叠加',
    imageKind: true,
  },
  mask: {
    label: '分割掩膜图',
    note: '纯二值分割掩膜',
    imageKind: true,
  },
  docx_report: {
    label: '实验报告 Word',
    note: '规范化实验报告（.docx）',
  },
}

const IMAGE_KINDS = new Set(
  Object.entries(EXPORT_KIND_META)
    .filter(([, v]) => v.imageKind)
    .map(([k]) => k),
)

const exportPriority: Record<string, number> = {
  docx_report: 0,
  xlsx: 1,
  batch_xlsx: 2,
  particles_xlsx: 3,
  overlay: 4,
  mask: 5,
  config: 6,
}

const getExportMeta = (kind: string) =>
  EXPORT_KIND_META[kind] ?? {
    label: kind,
    note: '导出文件',
  }

const singleExports = computed(() =>
  props.exports
    .filter((file) => file.kind !== 'bundle')
    .slice()
    .sort((left, right) => {
      const leftPriority = exportPriority[left.kind] ?? Number.MAX_SAFE_INTEGER
      const rightPriority = exportPriority[right.kind] ?? Number.MAX_SAFE_INTEGER
      return leftPriority - rightPriority || left.id - right.id
    }),
)

const selectedIds = ref<number[]>([])
const isExporting = ref(false)
const selectionInitialized = ref(false)

const availableSingleIds = computed(() => singleExports.value.map((file) => file.id))

watch(
  availableSingleIds,
  (ids) => {
    if (!ids.length) {
      selectedIds.value = []
      return
    }

    if (!selectionInitialized.value) {
      selectedIds.value = ids.slice()
      selectionInitialized.value = true
      return
    }

    const available = new Set(ids)
    const nextSelected = selectedIds.value.filter((id) => available.has(id))
    if (nextSelected.length !== selectedIds.value.length) {
      selectedIds.value = nextSelected
    }
  },
  { immediate: true },
)

const selectedFiles = computed(() => singleExports.value.filter((file) => selectedIds.value.includes(file.id)))
const selectedCount = computed(() => selectedFiles.value.length)
const allSelected = computed(() => singleExports.value.length > 0 && selectedCount.value === singleExports.value.length)

const selectAll = () => {
  selectedIds.value = availableSingleIds.value.slice()
  if (availableSingleIds.value.length) {
    selectionInitialized.value = true
  }
}

const clearSelection = () => {
  selectedIds.value = []
  if (availableSingleIds.value.length) {
    selectionInitialized.value = true
  }
}

const toggleSelectionAll = () => {
  if (allSelected.value) {
    clearSelection()
    return
  }

  selectAll()
}

const toggleFileSelection = (fileId: number) => {
  if (selectedIds.value.includes(fileId)) {
    selectedIds.value = selectedIds.value.filter((id) => id !== fileId)
    return
  }

  selectedIds.value = [...selectedIds.value, fileId]
  selectionInitialized.value = true
}

const normalizeZipEntryName = (path: string) => {
  const cleaned = path
    .replace(/\\/g, '/')
    .replace(/^([a-zA-Z]:)/, '')
    .replace(/^\/+/, '')
  return cleaned
    .split('/')
    .filter((segment) => segment && segment !== '.' && segment !== '..')
    .join('/')
}

const getFallbackEntryName = (file: { id: number; kind: string }) => {
  const kindExtensionMap: Record<string, string> = {
    csv: '.csv',
    xlsx: '.xlsx',
    json: '.json',
  }

  return `export-${file.id}${kindExtensionMap[file.kind] ?? ''}`
}

const makeUniqueEntryName = (entryName: string, usedNames: Set<string>) => {
  if (!usedNames.has(entryName)) {
    usedNames.add(entryName)
    return entryName
  }

  const lastSlashIndex = entryName.lastIndexOf('/')
  const prefix = lastSlashIndex >= 0 ? `${entryName.slice(0, lastSlashIndex + 1)}` : ''
  const baseName = lastSlashIndex >= 0 ? entryName.slice(lastSlashIndex + 1) : entryName
  const extensionMatch = baseName.match(/(\.[^.]+)$/)
  const suffix = extensionMatch?.[1] ?? ''
  const stem = suffix ? baseName.slice(0, -suffix.length) : baseName

  let attempt = 2
  let candidate = `${prefix}${stem} (${attempt})${suffix}`
  while (usedNames.has(candidate)) {
    attempt += 1
    candidate = `${prefix}${stem} (${attempt})${suffix}`
  }

  usedNames.add(candidate)
  return candidate
}

const fetchExportBlob = async (file: { id: number; url: string; path: string }) => {
  const response = await fetch(file.url, { credentials: 'include' })
  if (!response.ok) {
    throw new Error(`文件 ${file.path} 下载失败（${response.status}）`)
  }
  return response.blob()
}

const downloadBlob = (blob: Blob, fileName: string) => {
  const downloadUrl = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = downloadUrl
  anchor.download = fileName
  anchor.rel = 'noopener'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => {
    URL.revokeObjectURL(downloadUrl)
  }, 1000)
}

const exportSelectedZip = async () => {
  const filesToExport = selectedFiles.value.slice()
  if (!filesToExport.length || isExporting.value) return

  isExporting.value = true
  try {
    const zip = new JSZip()
    const usedNames = new Set<string>()

    for (const file of filesToExport) {
      const blob = await fetchExportBlob(file)
      const normalized = normalizeZipEntryName(file.path)
      const entryName = makeUniqueEntryName(normalized || getFallbackEntryName(file), usedNames)
      zip.file(entryName, blob)
    }

    const zipBlob = await zip.generateAsync({
      type: 'blob',
      compression: 'DEFLATE',
      compressionOptions: { level: 6 },
    })

    downloadBlob(zipBlob, 'selected-files.zip')
    ElMessage.success(`已导出 ${filesToExport.length} 个文件`)
  } catch (error) {
    const message = error instanceof Error ? error.message : '导出失败'
    ElMessage.error(message)
  } finally {
    isExporting.value = false
  }
}
</script>

<template>
  <section class="glass-card export-panel" aria-labelledby="statistics-export-title">
    <div class="export-panel__head">
      <div class="export-panel__copy">
        <h3 id="statistics-export-title" class="section-title">导出结果</h3>
        <p class="section-subtitle">勾选需要的统计文件或分割图像后统一打包导出，统计表、颗粒明细、配置快照和分割叠加图均在此处。</p>
      </div>
    </div>

    <div v-if="singleExports.length" class="export-stack">
      <div class="export-toolbar" role="group" aria-label="导出选择操作">
        <div class="export-toolbar-copy">
          <span class="export-toolbar-kicker">导出选择</span>
          <strong>勾选后统一打包为 ZIP</strong>
          <small aria-live="polite">已选 {{ selectedCount }} / {{ singleExports.length }}</small>
        </div>

        <div class="export-toolbar-actions">
          <button
            type="button"
            class="toolbar-action"
            :disabled="isExporting"
            :aria-disabled="isExporting"
            :aria-pressed="allSelected"
            @click="toggleSelectionAll"
          >
            {{ allSelected ? '取消全选' : '全选' }}
          </button>
          <button
            type="button"
            class="toolbar-action toolbar-primary"
            :disabled="!selectedCount || isExporting"
            :aria-disabled="!selectedCount || isExporting"
            @click="exportSelectedZip"
          >
            {{ isExporting ? '正在导出...' : `导出所选 ZIP（${selectedCount}）` }}
          </button>
        </div>
      </div>

      <div class="export-list" role="list" aria-label="可导出的统计文件">
        <label
          v-for="file in singleExports"
          :key="file.id"
          class="export-link"
          :class="{
            'is-selected': selectedIds.includes(file.id),
          }"
          role="listitem"
        >
          <input
            :checked="selectedIds.includes(file.id)"
            type="checkbox"
            class="export-checkbox"
            :disabled="isExporting"
            :aria-label="`选择 ${getExportMeta(file.kind).label}`"
            @change="toggleFileSelection(file.id)"
          />
          <div class="export-icon" aria-hidden="true">
            <svg v-if="IMAGE_KINDS.has(file.kind)" viewBox="0 0 28 28" fill="none">
              <rect x="3" y="6" width="22" height="16" rx="2.5" stroke="currentColor" stroke-width="1.6" />
              <circle cx="10" cy="12" r="2" fill="currentColor" />
              <path d="M7 18l4-4 3 3 4-5 3 3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
            <svg v-else viewBox="0 0 28 28" fill="none">
              <rect x="5" y="4" width="18" height="20" rx="3" stroke="currentColor" stroke-width="1.6" />
              <path d="M9 10h10M9 14h7M9 18h5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
            </svg>
          </div>
          <div class="export-copy">
            <strong>{{ getExportMeta(file.kind).label }}</strong>
            <small>{{ getExportMeta(file.kind).note }}</small>
            <span class="export-path">{{ file.path }}</span>
          </div>
        </label>
      </div>
    </div>
    <el-empty v-else description="统计结果生成后会在这里提供导出内容" />
  </section>
</template>

<style scoped>
.export-panel {
  --panel-border: var(--stats-border, rgba(148, 163, 184, 0.22));
  --panel-shadow: var(--stats-shadow, 0 16px 30px rgba(15, 23, 42, 0.08));
  --panel-accent: var(--stats-accent, #2f6bff);
  --panel-deep: var(--stats-deep, #0c2850);
  --panel-surface: rgba(255, 255, 255, 0.96);
  padding: 20px;
  border-radius: 20px;
  border: 1px solid var(--panel-border);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(246, 249, 253, 0.95));
  box-shadow: var(--panel-shadow);
}

.export-panel__head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 14px;
}

.export-panel__copy {
  display: grid;
  gap: 4px;
}

.export-stack {
  display: grid;
  gap: 14px;
}

.export-toolbar {
  display: grid;
  gap: 12px;
  padding: 16px;
  border-radius: 16px;
  border: 1px solid rgba(47, 107, 255, 0.14);
  background: linear-gradient(180deg, rgba(235, 242, 255, 0.82), rgba(248, 250, 253, 0.98));
}

.export-toolbar-copy {
  display: grid;
  gap: 4px;
}

.export-toolbar-kicker {
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.export-toolbar-copy strong {
  color: var(--panel-deep);
  font-size: 15px;
}

.export-toolbar-copy small {
  color: var(--muted);
  font-size: 12px;
}

.export-toolbar-actions {
  display: grid;
  grid-template-columns: 122px minmax(0, 1fr);
  gap: 10px;
  align-items: stretch;
}

.toolbar-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  min-height: 38px;
  padding: 0 14px;
  border-radius: 12px;
  border: 1px solid rgba(31, 40, 48, 0.12);
  background: rgba(255, 255, 255, 0.72);
  color: var(--ink);
  font-size: 13px;
  font-weight: 700;
  white-space: nowrap;
  cursor: pointer;
  transition:
    transform 0.18s ease,
    box-shadow 0.18s ease,
    border-color 0.18s ease,
    background 0.18s ease,
    color 0.18s ease,
    opacity 0.18s ease;
}

.export-toolbar-actions .toolbar-action:first-child {
  min-width: 0;
}

.export-toolbar-actions .toolbar-primary {
  min-width: 0;
  font-variant-numeric: tabular-nums;
}

.toolbar-action:hover:not(:disabled),
.toolbar-action:focus-visible:not(:disabled) {
  transform: translateY(-1px);
  border-color: rgba(47, 107, 255, 0.18);
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0 8px 18px rgba(47, 107, 255, 0.1);
  outline: none;
}

.toolbar-action:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.toolbar-primary {
  border-color: transparent;
  background: linear-gradient(135deg, rgba(63, 125, 255, 0.98), var(--panel-accent));
  color: white;
}

.toolbar-primary:hover:not(:disabled) {
  filter: brightness(1.03);
  box-shadow: 0 10px 22px rgba(47, 107, 255, 0.16);
}

.export-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.export-link {
  display: grid;
  grid-template-columns: auto auto minmax(0, 1fr);
  align-items: flex-start;
  gap: 10px;
  padding: 11px 12px;
  min-height: 72px;
  border-radius: 16px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.66);
  text-decoration: none;
  color: inherit;
  transition:
    border-color 0.18s ease,
    transform 0.18s ease,
    background 0.18s ease,
    box-shadow 0.18s ease;
}

.export-link:hover,
.export-link:focus-within {
  transform: translateY(-1px);
  border-color: rgba(47, 107, 255, 0.16);
  background: rgba(255, 255, 255, 0.78);
  box-shadow: 0 8px 18px rgba(47, 107, 255, 0.08);
}

.export-link.is-selected {
  border-color: rgba(47, 107, 255, 0.22);
  background: linear-gradient(180deg, rgba(240, 246, 255, 0.92), rgba(255, 255, 255, 0.9));
  box-shadow: 0 8px 18px rgba(47, 107, 255, 0.08);
}

.export-checkbox {
  margin-top: 2px;
  width: 16px;
  height: 16px;
  flex: 0 0 auto;
  accent-color: var(--panel-accent);
  cursor: pointer;
}

.export-checkbox:focus-visible {
  outline: 2px solid rgba(47, 107, 255, 0.24);
  outline-offset: 2px;
}

.export-copy {
  display: grid;
  gap: 3px;
  min-width: 0;
}

.export-copy strong {
  font-size: 13px;
}

.export-copy small {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.4;
}

.export-path {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.4;
  word-break: break-all;
}

.export-icon {
  width: 36px;
  height: 36px;
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  border-radius: 10px;
  border: 1px solid rgba(31, 40, 48, 0.1);
  background: rgba(22, 50, 79, 0.06);
  color: var(--panel-accent);
  transition: background 0.18s ease, border-color 0.18s ease;
}

.export-icon svg {
  width: 18px;
  height: 18px;
}

.is-selected .export-icon {
  background: rgba(47, 107, 255, 0.1);
  border-color: rgba(47, 107, 255, 0.2);
}

@media (max-width: 960px) {
  .export-list {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .export-panel {
    padding: 16px;
  }

  .export-toolbar-actions {
    grid-template-columns: 1fr;
  }
}
</style>
