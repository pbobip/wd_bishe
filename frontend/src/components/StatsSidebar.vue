<script setup lang="ts">
import { computed } from 'vue'

import type { RunRecord } from '../types'

const props = defineProps<{
  run: RunRecord
  summary: Record<string, any> | null | undefined
  exports: Array<{ id: number; kind: string; url: string; path: string }>
}>()

const modeLabel = (mode: string) => {
  if (mode === 'traditional') return '传统分割'
  if (mode === 'dl') return '深度学习'
  return mode
}

const formatNumber = (value: unknown, digits = 2) => {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numeric)) return '--'
  return numeric.toFixed(digits)
}

const batchModes = computed(() => Object.entries((props.summary?.batch ?? {}) as Record<string, Record<string, unknown>>))

const runOverview = computed(() => {
  const firstMode = batchModes.value[0]?.[1] as Record<string, unknown> | undefined
  return [
    { label: '状态', value: props.run.status },
    { label: '模式', value: props.run.segmentation_mode },
    { label: '输入', value: props.run.input_mode },
    { label: '图像数', value: String(firstMode?.image_count ?? '--') },
  ]
})

const metricGroups = computed(() =>
  batchModes.value.map(([mode, info]) => {
    const row = info as Record<string, unknown>
    const metrics = [
      { label: '平均 Vf', value: `${formatNumber((Number(row.avg_volume_fraction) || 0) * 100, 2)}%` },
      { label: '平均颗粒数', value: formatNumber(row.avg_particle_count, 1) },
      { label: '图均平均尺寸', value: `${formatNumber(row.avg_image_mean_size, 2)} ${String(row.size_unit ?? 'px')}` },
    ]

    const channelX = Number(row.avg_image_mean_channel_width_x)
    const channelY = Number(row.avg_image_mean_channel_width_y)
    if (Number.isFinite(channelX) || Number.isFinite(channelY)) {
      metrics.push({
        label: '水平 W',
        value: `${formatNumber(row.avg_image_mean_channel_width_x, 2)} ${String(row.channel_width_unit ?? row.size_unit ?? 'px')}`,
      })
      metrics.push({
        label: '垂直 W',
        value: `${formatNumber(row.avg_image_mean_channel_width_y, 2)} ${String(row.channel_width_unit ?? row.size_unit ?? 'px')}`,
      })
    }

    return {
      mode,
      label: modeLabel(mode),
      metrics,
    }
  }),
)

const EXPORT_KIND_META: Record<string, { label: string; note: string }> = {
  bundle: {
    label: '全部结果包',
    note: 'ZIP 整包，包含统计表、统计图、预测结果和配置快照',
  },
  csv: {
    label: '统计总表 CSV',
    note: '图像级主统计表',
  },
  xlsx: {
    label: '统计总表 XLSX',
    note: '图像级主统计表',
  },
  batch_csv: {
    label: '批次汇总 CSV',
    note: '批量任务均值与区间汇总',
  },
  batch_xlsx: {
    label: '批次汇总 XLSX',
    note: '批量任务均值与区间汇总',
  },
  particles_csv: {
    label: '颗粒明细 CSV',
    note: '对象级颗粒统计明细',
  },
  particles_xlsx: {
    label: '颗粒明细 XLSX',
    note: '对象级颗粒统计明细',
  },
  config: {
    label: '配置快照 JSON',
    note: '本次任务参数记录',
  },
}

const exportPriority: Record<string, number> = {
  csv: 1,
  xlsx: 2,
  batch_csv: 3,
  batch_xlsx: 4,
  particles_csv: 5,
  particles_xlsx: 6,
  config: 7,
}

const getExportMeta = (kind: string) =>
  EXPORT_KIND_META[kind] ?? {
    label: kind,
    note: '导出文件',
  }

const bundleExport = computed(() => props.exports.find((file) => file.kind === 'bundle') ?? null)

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
</script>

<template>
  <div class="sidebar-stack">
    <section class="glass-card sidebar-card">
      <div class="sidebar-heading">
        <div>
          <span class="panel-eyebrow">概览</span>
          <h3 class="section-title">结果摘要</h3>
        </div>
      </div>

      <div class="overview-grid">
        <article v-for="item in runOverview" :key="item.label" class="overview-item">
          <span>{{ item.label }}</span>
          <strong>{{ item.value }}</strong>
        </article>
      </div>
    </section>

    <section class="glass-card sidebar-card">
      <div class="sidebar-heading">
        <div>
          <span class="panel-eyebrow">统计</span>
          <h3 class="section-title">关键指标</h3>
        </div>
      </div>

      <div v-if="metricGroups.length" class="mode-group-list">
        <article v-for="group in metricGroups" :key="group.mode" class="mode-group">
          <div class="mode-group-head">
            <span class="status-chip">{{ group.label }}</span>
          </div>
          <div class="mode-metric-list">
            <div v-for="metric in group.metrics" :key="metric.label" class="mode-metric-item">
              <span>{{ metric.label }}</span>
              <strong>{{ metric.value }}</strong>
            </div>
          </div>
        </article>
      </div>
      <el-empty v-else description="尚未生成统计摘要" />
    </section>

    <section class="glass-card sidebar-card">
      <div class="sidebar-heading">
        <div>
          <span class="panel-eyebrow">导出</span>
          <h3 class="section-title">结果文件</h3>
        </div>
      </div>

      <div v-if="exports.length" class="export-stack">
        <article v-if="bundleExport" class="bundle-card">
          <div class="bundle-copy">
            <span class="bundle-badge">推荐</span>
            <strong>一键导出全部结果</strong>
            <p>下载 ZIP 整包，内含统计图、预测结果、统计表与配置快照，答辩或归档时直接使用。</p>
          </div>
          <a :href="bundleExport.url" download class="bundle-action">
            导出全部 ZIP
          </a>
          <span class="bundle-path">{{ bundleExport.path }}</span>
        </article>

        <div v-if="singleExports.length" class="export-subhead">
          <span>单文件下载</span>
          <small>{{ singleExports.length }} 项</small>
        </div>

        <div v-if="singleExports.length" class="export-list">
          <a v-for="file in singleExports" :key="file.id" :href="file.url" target="_blank" class="export-link">
            <div class="export-copy">
              <strong>{{ getExportMeta(file.kind).label }}</strong>
              <small>{{ getExportMeta(file.kind).note }}</small>
            </div>
            <span>{{ file.path }}</span>
          </a>
        </div>
      </div>
      <el-empty v-else description="任务完成后会在这里生成导出内容" />
    </section>
  </div>
</template>

<style scoped>
.sidebar-stack {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.sidebar-card {
  padding: 16px;
}

.sidebar-heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 12px;
}

.panel-eyebrow {
  display: inline-flex;
  align-items: center;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 11px;
  font-weight: 700;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.overview-item,
.mode-group,
.export-link {
  border-radius: 16px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.66);
}

.overview-item {
  padding: 12px 14px;
}

.overview-item span,
.mode-metric-item span {
  display: block;
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 12px;
}

.overview-item strong,
.mode-metric-item strong {
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.mode-group-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.mode-group {
  padding: 12px 14px;
}

.mode-group-head {
  margin-bottom: 10px;
}

.mode-metric-list {
  display: grid;
  gap: 8px;
}

.mode-metric-item {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
}

.mode-metric-item span {
  margin-bottom: 0;
}

.mode-metric-item strong {
  text-align: right;
  flex: 0 0 auto;
}

.export-stack {
  display: grid;
  gap: 12px;
}

.bundle-card {
  display: grid;
  gap: 12px;
  padding: 14px;
  border-radius: 18px;
  border: 1px solid rgba(23, 96, 135, 0.12);
  background:
    linear-gradient(135deg, rgba(23, 96, 135, 0.08), rgba(184, 90, 43, 0.05)),
    rgba(255, 255, 255, 0.82);
}

.bundle-copy {
  display: grid;
  gap: 6px;
}

.bundle-badge {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.12);
  color: var(--accent);
  font-size: 11px;
  font-weight: 700;
}

.bundle-copy strong {
  font-size: 18px;
  line-height: 1.3;
}

.bundle-copy p {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.6;
}

.bundle-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 42px;
  padding: 0 16px;
  border-radius: 14px;
  background: var(--accent);
  color: white;
  font-size: 14px;
  font-weight: 700;
  text-decoration: none;
  transition:
    transform 0.18s ease,
    box-shadow 0.18s ease,
    background 0.18s ease;
}

.bundle-action:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(23, 96, 135, 0.18);
}

.bundle-path {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.45;
  word-break: break-all;
}

.export-subhead {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding-inline: 2px;
}

.export-subhead span {
  color: var(--ink);
  font-size: 13px;
  font-weight: 700;
}

.export-subhead small {
  color: var(--muted);
  font-size: 12px;
}

.export-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 280px;
  overflow: auto;
  padding-right: 2px;
}

.export-link {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 11px 12px;
  text-decoration: none;
  transition:
    border-color 0.18s ease,
    transform 0.18s ease,
    background 0.18s ease;
}

.export-link:hover {
  transform: translateY(-1px);
  border-color: rgba(23, 96, 135, 0.16);
  background: rgba(255, 255, 255, 0.78);
}

.export-copy {
  display: grid;
  gap: 2px;
}

.export-copy strong {
  font-size: 13px;
}

.export-copy small {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.4;
}

.export-link span {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.4;
  word-break: break-all;
}

@media (max-width: 640px) {
  .overview-grid {
    grid-template-columns: 1fr;
  }

  .mode-metric-item {
    flex-direction: column;
    align-items: flex-start;
  }

  .mode-metric-item strong {
    text-align: left;
  }
}
</style>
