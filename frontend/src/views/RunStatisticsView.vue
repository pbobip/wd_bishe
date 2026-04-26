<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '../api'
import StatsSidebar from '../components/StatsSidebar.vue'
import StatisticsAnalysisPanel from '../components/statistics/StatisticsAnalysisPanel.vue'
import StatisticsDescriptionPanel from '../components/statistics/StatisticsDescriptionPanel.vue'
import StatisticsImageCheckTable from '../components/statistics/StatisticsImageCheckTable.vue'
import StatisticsMetricSelector from '../components/statistics/StatisticsMetricSelector.vue'
import type { RunResultsPayload } from '../types'
import {
  buildModeStatistics,
  formatNumber,
  getAvailableModes,
  metricDefinitions,
  type StatisticMetricKey,
} from '../utils/runStatistics'

type ChannelAxis = 'x' | 'y'

const route = useRoute()
const router = useRouter()

const runId = computed(() => Number(route.params.id))
const payload = ref<RunResultsPayload | null>(null)
const selectedMode = ref('')
const selectedMetric = ref<StatisticMetricKey>('size')
const channelAxis = ref<ChannelAxis>('x')
const imagePage = ref(1)
const showAllImages = ref(false)
const formulaDialogVisible = ref(false)

const IMAGE_PAGE_SIZE = 5

let timer: number | null = null

const loadPayload = async () => {
  const response = await api.get<RunResultsPayload>(`/runs/${runId.value}/results`)
  payload.value = response.data
}

const stopPolling = () => {
  if (timer) {
    window.clearInterval(timer)
    timer = null
  }
}

const startPolling = () => {
  stopPolling()
  timer = window.setInterval(async () => {
    await loadPayload()
    const status = payload.value?.run.status
    if (status && ['completed', 'failed', 'partial_success'].includes(status)) {
      stopPolling()
    }
  }, 2500)
}

const activeModes = computed(() => getAvailableModes(payload.value))
const analysis = computed(() => buildModeStatistics(payload.value, selectedMode.value))
const batch = computed(() => analysis.value.batch)
const imageRows = computed(() => analysis.value.imageRows)
const inputUrlByImageId = computed(() => {
  const entries = payload.value?.images ?? []
  return new Map(entries.map((item) => [item.image_id, item.input_url ?? null]))
})

const statisticsReady = computed(() =>
  Boolean(payload.value && ['completed', 'partial_success'].includes(payload.value.run.status)),
)

const statisticsPendingCopy = computed(() => {
  if (!payload.value) return '正在加载当前任务统计。'
  if (payload.value.run.status === 'running' || payload.value.run.status === 'queued') {
    return '主分割仍在执行中，请先回到结果页核查当前分割效果，待结果确认后再进入统计分析。'
  }
  if (payload.value.run.status === 'draft') {
    return '当前任务还没有开始处理，请先回到任务创建页启动主分割。'
  }
  if (payload.value.run.status === 'failed') {
    return '当前任务没有产出可确认的统计结果，请先回到结果页检查失败原因。'
  }
  return '当前任务统计结果尚未就绪，请先回到结果页确认分割结果。'
})

watch(activeModes, (modes) => {
  if (!modes.length) {
    selectedMode.value = ''
    return
  }
  if (!selectedMode.value || !modes.includes(selectedMode.value)) {
    selectedMode.value = modes[0]
  }
}, { immediate: true })

watch(selectedMetric, () => {
  imagePage.value = 1
})

const modeLabel = (mode: string) => {
  if (mode === 'traditional') return '传统分割'
  if (mode === 'dl') return '深度学习'
  return mode || '当前模式'
}

const percent = (value: number, digits = 2) => `${formatNumber(value * 100, digits)}%`
const withUnit = (value: number, unit: string, digits = 2) => `${formatNumber(value, digits)} ${unit}`.trim()

const average = (values: number[]) => {
  if (!values.length) return 0
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

const median = (values: number[]) => {
  if (!values.length) return 0
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2
  }
  return sorted[middle]
}

const std = (values: number[]) => {
  if (!values.length) return 0
  const avg = average(values)
  return Math.sqrt(values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length)
}

const quantile = (values: number[], ratio: number) => {
  if (!values.length) return 0
  const sorted = [...values].sort((left, right) => left - right)
  const position = (sorted.length - 1) * ratio
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) return sorted[lower]
  const weight = position - lower
  return sorted[lower] * (1 - weight) + sorted[upper] * weight
}

const currentMetricTitle = computed(() => {
  if (selectedMetric.value === 'vf') return 'Vf 按图片排序'
  if (selectedMetric.value === 'particle_count') return '颗粒数按图片排序'
  if (selectedMetric.value === 'area') return '面积分布分析'
  if (selectedMetric.value === 'size') return '尺寸分布分析'
  return channelAxis.value === 'x' ? '水平 W 分布分析' : '垂直 W 分布分析'
})

const currentMetricLabel = computed(() => {
  if (selectedMetric.value === 'vf') return 'Vf'
  if (selectedMetric.value === 'particle_count') return '颗粒数'
  if (selectedMetric.value === 'area') return '面积'
  if (selectedMetric.value === 'size') return '尺寸'
  return channelAxis.value === 'x' ? '水平 W' : '垂直 W'
})


const hasDistributionValues = computed(() =>
  distributionValues.value.some((item) => Number.isFinite(item)),
)

const currentMetricSubtitle = computed(() => {
  if (selectedMetric.value === 'vf') {
    if (!imageRows.value.length) return '当前模式暂无图片级统计结果。'
    return '按图片 Vf 降序排列，可直接比较各图 γ′ 相含量高低。'
  }
  if (selectedMetric.value === 'particle_count') {
    if (!imageRows.value.length) return '当前模式暂无图片级统计结果。'
    return '按图片颗粒数降序排列，可直接比较各图 γ′ 相数量与批内离散程度。'
  }
  if (!hasDistributionValues.value) {
    return '当前模式暂无可用于分布统计的颗粒级数据。'
  }
  if (selectedMetric.value === 'area') return '按颗粒面积分箱统计，直观查看各区间颗粒数与整体离散程度。'
  if (selectedMetric.value === 'size') return '按颗粒等效直径分箱统计，直观查看各区间颗粒数与整体离散程度。'
  return '按通道测量结果分箱统计，直观查看各区间颗粒数与整体离散程度。'
})

const currentUnit = computed(() => {
  if (selectedMetric.value === 'vf') return '%'
  if (selectedMetric.value === 'particle_count') return '个'
  if (selectedMetric.value === 'area') return batch.value.areaUnit
  if (selectedMetric.value === 'size') return batch.value.sizeUnit
  return batch.value.channelWidthUnit
})

const chartKind = computed(() => (
  selectedMetric.value === 'vf' || selectedMetric.value === 'particle_count' ? 'by_image' : 'distribution'
))

const distributionValues = computed(() => {
  if (selectedMetric.value === 'area') return batch.value.areas
  if (selectedMetric.value === 'size') return batch.value.sizes
  if (selectedMetric.value === 'channel_width') {
    return channelAxis.value === 'x' ? batch.value.channelWidthsX : batch.value.channelWidthsY
  }
  return []
})

const imageSeries = computed(() => {
  if (selectedMetric.value === 'vf') {
    return imageRows.value.map((row) => ({ label: row.imageName, value: row.volumeFraction * 100 }))
  }
  if (selectedMetric.value === 'particle_count') {
    return imageRows.value.map((row) => ({ label: row.imageName, value: row.particleCount }))
  }
  if (selectedMetric.value === 'area') {
    return imageRows.value.map((row) => ({ label: row.imageName, value: row.meanArea }))
  }
  if (selectedMetric.value === 'size') {
    return imageRows.value.map((row) => ({ label: row.imageName, value: row.meanSize }))
  }
  return imageRows.value.map((row) => ({
    label: row.imageName,
    value: channelAxis.value === 'x' ? row.meanChannelWidthX : row.meanChannelWidthY,
  }))
})

const imageMetricLabel = computed(() => {
  if (selectedMetric.value === 'vf') return 'Vf (%)'
  if (selectedMetric.value === 'particle_count') return '颗粒数'
  if (selectedMetric.value === 'area') return `平均面积（${batch.value.areaUnit}）`
  if (selectedMetric.value === 'size') return `平均尺寸（${batch.value.sizeUnit}）`
  return channelAxis.value === 'x' ? `水平 W（${batch.value.channelWidthUnit}）` : `垂直 W（${batch.value.channelWidthUnit}）`
})

const currentValues = computed(() => {
  if (selectedMetric.value === 'vf') return imageRows.value.map((row) => row.volumeFraction * 100)
  if (selectedMetric.value === 'particle_count') return imageRows.value.map((row) => row.particleCount)
  return distributionValues.value
})

const formatMetricValue = (value: number) => {
  if (!Number.isFinite(value)) return '--'
  if (selectedMetric.value === 'vf') return `${formatNumber(value, 2)}%`
  if (selectedMetric.value === 'particle_count') return formatNumber(value, 0)
  return withUnit(value, currentUnit.value)
}

const descriptionRows = computed(() => {
  const values = currentValues.value.filter((item) => Number.isFinite(item))
  if (!values.length) {
    return [
      { label: '样本数 n', value: '0' },
      { label: '均值', value: '--' },
      { label: '中位数', value: '--' },
      { label: '标准差', value: '--' },
      { label: 'P10–P90', value: '--' },
      { label: '最小值 / 最大值', value: '--' },
    ]
  }
  return [
    { label: '样本数 n', value: formatNumber(values.length, 0) },
    { label: '均值', value: formatMetricValue(average(values)) },
    { label: '中位数', value: formatMetricValue(median(values)) },
    { label: '标准差', value: formatMetricValue(std(values)) },
    { label: 'P10–P90', value: `${formatMetricValue(quantile(values, 0.1))} – ${formatMetricValue(quantile(values, 0.9))}` },
    { label: '最小值 / 最大值', value: `${formatMetricValue(Math.min(...values))} – ${formatMetricValue(Math.max(...values))}` },
  ]
})

const overviewCards = computed(() => {
  const imageCount = batch.value.imageCount
    || payload.value?.images.filter((item) => Boolean(item.modes?.[selectedMode.value]?.summary)).length
    || 0
  const particleCount = batch.value.totalParticleCount || 0
  const calibratedImageCount = batch.value.calibratedImageCount || 0
  const unitValue = currentUnit.value
  return [
    { key: 'images', title: '图像数', value: String(imageCount), icon: 'image' },
    { key: 'particles', title: '有效颗粒数', value: String(particleCount), icon: 'particles' },
    { key: 'calibrated', title: '已标定图像', value: String(calibratedImageCount), icon: 'target' },
    { key: 'unit', title: '当前单位', value: unitValue, icon: 'ruler' },
  ]
})

const statisticsScopeNotice = computed(() => {
  if (batch.value.mixedCalibration) {
    return '口径说明：当前批次包含已标定与未标定图片；Vf 和颗粒数按全图统计，物理量（面积 / 尺寸 / 通道宽度）分布仅基于已标定图片，表格中仍保留全部图片并按各自单位显示。'
  }
  if (batch.value.physicalStatsScope === 'pixels_only') {
    return '口径说明：当前批次未检测到标定信息，所有物理量（面积 / 尺寸 / 通道宽度）均以 px / px² 为单位统计。'
  }
  return ''
})

const pageSize = computed(() => {
  const total = imageRows.value.length
  return showAllImages.value ? Math.max(total, 1) : IMAGE_PAGE_SIZE
})

type ImageCheckRow = {
  imageId: number
  imageName: string
  thumbUrl: string | null
  volumeFractionLabel: string
  particleCountLabel: string
  meanSizeLabel: string
  unitLabel: string
  statusLabel: string
  statusType: 'success' | 'warning'
}

const paginatedImageRows = computed(() => {
  const rows = imageRows.value
  if (!rows.length) return []
  const start = (imagePage.value - 1) * pageSize.value
  return rows.slice(start, start + pageSize.value)
})

const imageCheckRows = computed<ImageCheckRow[]>(() => {
  if (paginatedImageRows.value.length) {
    return paginatedImageRows.value.map((row) => ({
      imageId: row.imageId,
      imageName: row.imageName,
      thumbUrl: inputUrlByImageId.value.get(row.imageId) ?? null,
      volumeFractionLabel: Number.isFinite(row.volumeFraction) ? percent(row.volumeFraction) : '--',
      particleCountLabel: Number.isFinite(row.particleCount) ? formatNumber(row.particleCount, 0) : '--',
      meanSizeLabel:
        row.particleCount > 0 && Number.isFinite(row.meanSize)
          ? withUnit(row.meanSize, row.calibrated && row.appliedUmPerPx ? row.sizeUnit : 'px')
          : '--',
      unitLabel: row.calibrated && row.appliedUmPerPx ? row.sizeUnit : 'px',
      statusLabel: row.calibrated && row.appliedUmPerPx ? '已标定' : '按像素统计',
      statusType: row.calibrated && row.appliedUmPerPx ? 'success' : 'warning',
    }))
  }
  return []
})

const formulaRows = computed(() => {
  return [
    metricDefinitions.vf,
    metricDefinitions.particle_count,
    metricDefinitions.area,
    metricDefinitions.size,
    metricDefinitions.channel_width,
  ]
})

const openFormulaDialog = () => {
  formulaDialogVisible.value = true
}

const toggleShowAll = () => {
  showAllImages.value = !showAllImages.value
  imagePage.value = 1
}

onMounted(async () => {
  await loadPayload()
  if (payload.value && !['completed', 'failed', 'partial_success'].includes(payload.value.run.status)) {
    startPolling()
  }
})

onBeforeUnmount(() => {
  stopPolling()
})
</script>

<template>
  <div v-if="statisticsReady" class="statistics-dashboard">
    <section class="overview-grid">
      <article v-for="card in overviewCards" :key="card.key" class="glass-card overview-card">
        <div class="overview-card__icon" :class="`is-${card.key}`" aria-hidden="true">
          <svg v-if="card.icon === 'image'" viewBox="0 0 28 28" fill="none">
            <rect x="4" y="6" width="20" height="16" rx="2" />
            <circle cx="11" cy="12" r="2" />
            <path d="M6 19l5-5 4 4 3-3 4 4" />
          </svg>
          <svg v-else-if="card.icon === 'particles'" viewBox="0 0 28 28" fill="none">
            <circle cx="8" cy="8" r="3" />
            <circle cx="20" cy="10" r="3" />
            <circle cx="10" cy="20" r="3" />
            <circle cx="20" cy="20" r="3" />
          </svg>
          <svg v-else-if="card.icon === 'target'" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="7" />
            <circle cx="14" cy="14" r="2" />
            <path d="M14 2v5M14 21v5M2 14h5M21 14h5" />
          </svg>
          <svg v-else viewBox="0 0 28 28" fill="none">
            <path d="M7 21L21 7" />
            <path d="M8 20l-2 5 5-2L25 9l-3-3L8 20Z" />
          </svg>
        </div>
        <div class="overview-card__copy">
          <span>{{ card.title }}</span>
          <strong>{{ card.value }}</strong>
        </div>
      </article>
    </section>

    <section v-if="statisticsScopeNotice" class="scope-note">
      <span class="scope-note__label">统计口径</span>
      <p>{{ statisticsScopeNotice }}</p>
    </section>

    <section v-if="activeModes.length > 1" class="mode-switch">
      <span class="mode-switch__label">分割模式</span>
      <div class="mode-switch__items">
        <button
          v-for="mode in activeModes"
          :key="mode"
          type="button"
          class="mode-switch__item"
          :class="{ 'is-active': selectedMode === mode }"
          @click="selectedMode = mode"
        >
          {{ modeLabel(mode) }}
        </button>
      </div>
    </section>

    <section id="analysis-section" class="dashboard-main">
      <StatisticsMetricSelector
        v-model="selectedMetric"
        :axis="channelAxis"
        @update:axis="channelAxis = $event"
      />

      <StatisticsAnalysisPanel
        :title="currentMetricTitle"
        :subtitle="currentMetricSubtitle"
        :kind="chartKind"
        :metric-label="currentMetricLabel"
        :unit-label="currentUnit"
        :values="distributionValues"
        :image-series="imageSeries"
        :image-metric-label="imageMetricLabel"
      />

      <StatisticsDescriptionPanel :rows="descriptionRows" />
    </section>

    <StatisticsImageCheckTable
      id="image-check-section"
      :rows="imageCheckRows"
      :current-page="imagePage"
      :page-size="pageSize"
      :total="imageRows.length"
      :view-all-label="showAllImages ? '收起列表' : '查看全部'"
      @update:currentPage="imagePage = $event"
      @view-all="toggleShowAll"
    />

    <section id="export-section" class="export-section-stack">
      <StatsSidebar :exports="payload?.exports ?? []" />
      <section class="glass-card formula-entry-card">
        <div class="formula-entry-card__copy">
          <h3 class="section-title">统计公式说明</h3>
          <p class="section-subtitle">查看各统计量的口径与公式，不参与文件导出流程。</p>
        </div>
        <el-button plain @click="openFormulaDialog">查看说明</el-button>
      </section>
    </section>

    <el-dialog v-model="formulaDialogVisible" title="统计公式说明" width="860px">
      <div class="formula-dialog">
        <article v-for="item in formulaRows" :key="item.label" class="formula-item">
          <header class="formula-item__head">
            <h3>{{ item.title }}</h3>
            <p>{{ item.description }}</p>
          </header>

          <section class="formula-board">
            <div class="formula-board__expression" v-html="item.formulaSvg" />
          </section>
        </article>
      </div>
    </el-dialog>
  </div>

  <div v-else-if="payload" class="statistics-dashboard statistics-dashboard--waiting">
    <section class="glass-card waiting-card">
      <div>
        <h2 class="section-title">统计结果暂未开放</h2>
        <p class="section-subtitle">{{ statisticsPendingCopy }}</p>
      </div>
      <el-button type="primary" @click="router.push(`/runs/${payload.run.id}`)">返回结果页</el-button>
    </section>
  </div>

  <div v-else class="statistics-dashboard">
    <section class="glass-card waiting-card">
      <el-skeleton :rows="10" animated />
    </section>
  </div>
</template>

<style scoped>
.statistics-dashboard {
  --stats-deep: #0c2850;
  --stats-deep-strong: #13386b;
  --stats-accent: #2f6bff;
  --stats-accent-soft: rgba(47, 107, 255, 0.1);
  --stats-surface: rgba(255, 255, 255, 0.9);
  --stats-surface-strong: rgba(255, 255, 255, 0.96);
  --stats-surface-muted: #f4f7fb;
  --stats-border: rgba(148, 163, 184, 0.22);
  --stats-border-strong: rgba(47, 107, 255, 0.18);
  --stats-shadow: 0 16px 30px rgba(15, 23, 42, 0.08);
  --stats-shadow-strong: 0 18px 34px rgba(11, 45, 92, 0.22);
  --stats-radius-xl: 24px;
  --stats-radius-lg: 20px;
  --stats-radius-md: 16px;
  --stats-gap: 18px;
  display: grid;
  gap: var(--stats-gap);
  min-width: 0;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}

.overview-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 18px 20px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(246, 249, 253, 0.96));
  border: 1px solid var(--stats-border);
  border-radius: var(--stats-radius-lg);
  box-shadow: var(--stats-shadow);
}

.overview-card__icon {
  width: 52px;
  height: 52px;
  display: grid;
  place-items: center;
  flex: 0 0 52px;
  border-radius: 16px;
  color: var(--stats-accent);
  background: linear-gradient(180deg, #eef4ff, #e6eefc);
  border: 1px solid rgba(47, 107, 255, 0.12);
}

.overview-card__icon svg {
  width: 28px;
  height: 28px;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.overview-card__copy {
  display: grid;
  gap: 4px;
}

.overview-card__copy span {
  color: var(--muted);
  font-size: 13px;
}

.overview-card__copy strong {
  font-size: 22px;
  line-height: 1.2;
  letter-spacing: -0.02em;
}

.mode-switch {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 2px;
}

.scope-note {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 14px 18px;
  border-radius: 18px;
  border: 1px solid rgba(47, 107, 255, 0.12);
  background: linear-gradient(180deg, rgba(235, 242, 255, 0.78), rgba(246, 249, 253, 0.94));
}

.scope-note__label {
  flex: 0 0 auto;
  min-height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  background: rgba(47, 107, 255, 0.1);
  color: var(--stats-deep);
  font-size: 12px;
  font-weight: 800;
  line-height: 28px;
}

.scope-note p {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.75;
}

.mode-switch__label {
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
}

.mode-switch__items {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.mode-switch__item {
  min-height: 36px;
  padding: 0 14px;
  border: 1px solid var(--stats-border);
  border-radius: 999px;
  background: var(--stats-surface);
  color: var(--ink);
  font-weight: 700;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    background 0.2s ease,
    box-shadow 0.2s ease;
}

.mode-switch__item:hover,
.mode-switch__item:focus-visible {
  transform: translateY(-1px);
  border-color: var(--stats-border-strong);
  background: var(--stats-surface-strong);
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
  outline: none;
}

.mode-switch__item.is-active {
  border-color: rgba(12, 40, 80, 0.12);
  background: linear-gradient(135deg, var(--stats-deep-strong), var(--stats-deep));
  color: #ffffff;
  box-shadow: 0 12px 24px rgba(12, 40, 80, 0.16);
}

.dashboard-main {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr) 330px;
  gap: var(--stats-gap);
  align-items: start;
}

.export-section-stack {
  display: grid;
  gap: var(--stats-gap);
}

.formula-entry-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 20px;
}

.formula-entry-card__copy {
  display: grid;
  gap: 4px;
}

.formula-dialog {
  display: grid;
  gap: 18px;
  max-height: 72vh;
  overflow: auto;
  padding-right: 4px;
}

.formula-item {
  display: grid;
  gap: 16px;
  padding: 24px 26px;
  border-radius: var(--stats-radius-md);
  background: linear-gradient(180deg, #f9fbff, #f5f8fc);
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.formula-item__head {
  display: grid;
  gap: 8px;
}

.formula-item__head h3 {
  margin: 0;
  color: #364152;
  font-size: 20px;
  font-weight: 900;
  letter-spacing: -0.02em;
}

.formula-item__head p {
  margin: 0;
  color: #6b7280;
  font-size: 15px;
  line-height: 1.75;
}

.formula-board {
  display: grid;
  place-items: center;
  min-height: 112px;
  padding: 10px 8px 2px;
  border-radius: 0;
  background: transparent;
  border: 0;
  box-shadow: none;
}

.formula-board__expression {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
}

.formula-board__expression :deep(.formula-svg) {
  width: min(100%, 760px);
  height: auto;
  overflow: visible;
}

.formula-board__expression :deep(.formula-svg__text) {
  fill: var(--stats-deep);
  font-family: 'Cambria Math', 'STIX Two Math', 'Times New Roman', 'Noto Serif SC', 'SimSun', serif;
  dominant-baseline: middle;
}

.formula-board__expression :deep(.formula-svg__main) {
  font-size: 44px;
}

.formula-board__expression :deep(.formula-svg__cn) {
  font-size: 34px;
}

.formula-board__expression :deep(.formula-svg__cn--small) {
  font-size: 28px;
}

.formula-board__expression :deep(.formula-svg__sub) {
  font-size: 24px;
}

.formula-board__expression :deep(.formula-svg__super) {
  font-size: 24px;
}

.formula-board__expression :deep(.formula-svg__var) {
  font-style: italic;
}

.formula-board__expression :deep(.formula-svg__line) {
  stroke: var(--stats-deep);
  stroke-width: 2.4;
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.waiting-card {
  display: grid;
  gap: 16px;
  padding: 28px;
  border-radius: var(--stats-radius-lg);
}

.statistics-dashboard--waiting {
  min-height: 360px;
}

@media (max-width: 1360px) {
  .overview-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .dashboard-main {
    grid-template-columns: 240px minmax(0, 1fr);
  }
}

@media (max-width: 1080px) {
  .dashboard-main {
    grid-template-columns: 1fr;
  }

  .formula-entry-card {
    flex-direction: column;
    align-items: flex-start;
  }

  .formula-board {
    min-height: 124px;
    padding: 20px 24px;
  }

  .formula-board__expression {
    width: 100%;
  }
}

@media (max-width: 720px) {
  .overview-grid {
    grid-template-columns: 1fr;
  }

  .formula-board {
    min-height: 112px;
    padding: 18px 18px;
  }

  .formula-board__expression {
    width: 100%;
  }

  .formula-board__expression :deep(.formula-svg__main) {
    font-size: 36px;
  }

  .formula-board__expression :deep(.formula-svg__cn) {
    font-size: 32px;
  }

  .formula-board__expression :deep(.formula-svg__sub),
  .formula-board__expression :deep(.formula-svg__super) {
    font-size: 20px;
  }
}
</style>
