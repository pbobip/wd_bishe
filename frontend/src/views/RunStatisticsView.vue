<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '../api'
import ChartCanvas from '../components/ChartCanvas.vue'
import type { RunResultsPayload } from '../types'
import {
  buildModeStatistics,
  formatNumber,
  getAvailableModes,
  metricDefinitions,
  type ConfidenceInterval95,
  type HistogramBucket,
  type StatisticMetricKey,
} from '../utils/runStatistics'

type HistogramMode = 'probability' | 'count'
type ChannelAxis = 'x' | 'y'
type DetailTab = 'images' | 'particles'

const route = useRoute()
const router = useRouter()

const runId = computed(() => Number(route.params.id))
const payload = ref<RunResultsPayload | null>(null)
const selectedMode = ref('')
const selectedMetric = ref<StatisticMetricKey>('vf')
const histogramMode = ref<HistogramMode>('probability')
const channelAxis = ref<ChannelAxis>('x')
const detailTab = ref<DetailTab>('images')
const imageSearch = ref('')
const particleSearch = ref('')
const particleFilteredOnly = ref(false)
const imagePage = ref(1)
const particlePage = ref(1)
const infoPanels = ref<string[]>([])

const IMAGE_PAGE_SIZE = 16
const PARTICLE_PAGE_SIZE = 120

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

const modeLabel = (mode: string) => {
  if (mode === 'traditional') return '传统分割'
  if (mode === 'dl') return '深度学习'
  if (mode === 'compare') return '结果对比'
  return mode
}

const percent = (value: number, digits = 2) => `${formatNumber(value * 100, digits)}%`
const withUnit = (value: number, unit: string, digits = 2) => `${formatNumber(value, digits)} ${unit}`.trim()

const formatCiText = (ci: ConfidenceInterval95, options?: { percentage?: boolean }) => {
  if (!ci.available || ci.lower == null || ci.upper == null) {
    return ci.reason ?? '样本不足'
  }
  if (options?.percentage) {
    return `${percent(ci.lower)} ~ ${percent(ci.upper)}`
  }
  return `${formatNumber(ci.lower, 2)} ~ ${formatNumber(ci.upper, 2)} ${ci.unit ?? ''}`.trim()
}

const formatValue = (value: number, unit: string, options?: { percentage?: boolean; digits?: number }) => {
  if (options?.percentage) return percent(value, options.digits ?? 2)
  return withUnit(value, unit, options?.digits ?? 2)
}

const activeModes = computed(() => getAvailableModes(payload.value))
const analysis = computed(() => buildModeStatistics(payload.value, selectedMode.value))
const batch = computed(() => analysis.value.batch)
const modeOptions = computed(() => activeModes.value.map((mode) => ({ label: modeLabel(mode), value: mode })))
const activeModeLabel = computed(() => modeLabel(selectedMode.value))
const metricDefinition = computed(() => metricDefinitions[selectedMetric.value])
const isDistributionMetric = computed(() => ['area', 'size', 'channel_width'].includes(selectedMetric.value))

const metricOptions = [
  { label: 'Vf', value: 'vf' as const },
  { label: '颗粒数', value: 'particle_count' as const },
  { label: '面积', value: 'area' as const },
  { label: '尺寸', value: 'size' as const },
  { label: '通道宽度', value: 'channel_width' as const },
]

const currentCi = computed(() => {
  if (selectedMetric.value === 'vf') return batch.value.volumeFractionCi95
  if (selectedMetric.value === 'particle_count') return batch.value.particleCountCi95
  if (selectedMetric.value === 'area') return batch.value.imageMeanAreaCi95
  if (selectedMetric.value === 'size') return batch.value.imageMeanSizeCi95
  return channelAxis.value === 'x' ? batch.value.imageMeanChannelWidthXCi95 : batch.value.imageMeanChannelWidthYCi95
})

const currentUnit = computed(() => {
  if (selectedMetric.value === 'vf') return '%'
  if (selectedMetric.value === 'particle_count') return '个'
  if (selectedMetric.value === 'area') return batch.value.areaUnit
  if (selectedMetric.value === 'size') return batch.value.sizeUnit
  return batch.value.channelWidthUnit
})

const currentMetricLabel = computed(() => {
  if (selectedMetric.value === 'channel_width') {
    return channelAxis.value === 'x' ? '水平 γ 通道宽度' : '垂直 γ 通道宽度'
  }
  return metricDefinition.value.title
})

const calibrationProbe = computed<Record<string, unknown> | null>(() => {
  const summary = payload.value?.run.summary as Record<string, any> | undefined
  const topLevel = summary?.calibration_probe
  if (topLevel && typeof topLevel === 'object') return topLevel
  const modeSummary = summary?.batch?.[selectedMode.value]
  if (modeSummary?.calibration_probe && typeof modeSummary.calibration_probe === 'object') {
    return modeSummary.calibration_probe
  }
  return null
})

const calibrationItems = computed(() => {
  const probe = calibrationProbe.value
  if (!probe) return []
  return [
    { label: '标定值', value: payload.value?.run.config?.input_config?.um_per_px ? `${payload.value?.run.config?.input_config?.um_per_px} um/px` : '未填写' },
    { label: '底栏识别', value: probe.footer_detected ? '已识别' : '未识别' },
    { label: '比例尺', value: probe.ocr_scale_bar_um ? `${probe.ocr_scale_bar_um} μm` : '未识别' },
    { label: '比例尺像素', value: probe.scale_bar_pixels ? `${probe.scale_bar_pixels} px` : '未识别' },
    { label: 'FoV', value: probe.ocr_fov_um ? `${probe.ocr_fov_um} μm` : '未识别' },
    { label: 'Mag', value: probe.ocr_magnification_text ? String(probe.ocr_magnification_text) : '未识别' },
    { label: 'WD', value: probe.ocr_wd_mm ? `${probe.ocr_wd_mm} mm` : '未识别' },
  ]
})

const imageRows = computed(() => analysis.value.imageRows)
const activeParticleRows = computed(() => (detailTab.value === 'particles' ? analysis.value.particleRows : []))

const filteredImageRows = computed(() => {
  const keyword = imageSearch.value.trim().toLowerCase()
  if (!keyword) return imageRows.value
  return imageRows.value.filter((row) => row.imageName.toLowerCase().includes(keyword))
})

const filteredParticleRows = computed(() => {
  const keyword = particleSearch.value.trim().toLowerCase()
  return activeParticleRows.value.filter((row) => {
    if (particleFilteredOnly.value && !row.filtered) return false
    if (!keyword) return true
    return [row.imageName, String(row.label), row.filterReason].some((item) => item.toLowerCase().includes(keyword))
  })
})

const paginatedImageRows = computed(() => {
  const start = (imagePage.value - 1) * IMAGE_PAGE_SIZE
  return filteredImageRows.value.slice(start, start + IMAGE_PAGE_SIZE)
})

const paginatedParticleRows = computed(() => {
  const start = (particlePage.value - 1) * PARTICLE_PAGE_SIZE
  return filteredParticleRows.value.slice(start, start + PARTICLE_PAGE_SIZE)
})

const metricCards = computed(() => {
  if (selectedMetric.value === 'vf') {
    return [
      { label: '平均 Vf', value: percent(batch.value.avgVolumeFraction), note: `${batch.value.imageCount} 张图像` },
      { label: '最低 Vf', value: percent(batch.value.minVolumeFraction), note: '图像级最小值' },
      { label: '最高 Vf', value: percent(batch.value.maxVolumeFraction), note: '图像级最大值' },
      { label: '95% CI', value: formatCiText(batch.value.volumeFractionCi95, { percentage: true }), note: `n = ${batch.value.volumeFractionCi95.n}` },
    ]
  }

  if (selectedMetric.value === 'particle_count') {
    return [
      { label: '总颗粒数', value: String(batch.value.totalParticleCount), note: '对象总量' },
      { label: '图均颗粒数', value: formatNumber(batch.value.avgParticleCount, 1), note: `${batch.value.imageCount} 张图像` },
      { label: '图像数', value: String(batch.value.imageCount), note: '有效样本' },
      { label: '95% CI', value: formatCiText(batch.value.particleCountCi95), note: `n = ${batch.value.particleCountCi95.n}` },
    ]
  }

  if (selectedMetric.value === 'area') {
    return [
      { label: '平均面积', value: withUnit(batch.value.meanArea, batch.value.areaUnit), note: '对象级均值' },
      { label: '中位面积', value: withUnit(batch.value.medianArea, batch.value.areaUnit), note: '对象级中位数' },
      { label: '面积标准差', value: withUnit(batch.value.stdArea, batch.value.areaUnit), note: '对象级波动' },
      { label: '图均面积 95% CI', value: formatCiText(batch.value.imageMeanAreaCi95), note: `n = ${batch.value.imageMeanAreaCi95.n}` },
    ]
  }

  if (selectedMetric.value === 'size') {
    return [
      { label: '平均尺寸', value: withUnit(batch.value.meanSize, batch.value.sizeUnit), note: '对象级均值' },
      { label: '中位尺寸', value: withUnit(batch.value.medianSize, batch.value.sizeUnit), note: '对象级中位数' },
      { label: '尺寸标准差', value: withUnit(batch.value.stdSize, batch.value.sizeUnit), note: '对象级波动' },
      { label: '图均尺寸 95% CI', value: formatCiText(batch.value.imageMeanSizeCi95), note: `n = ${batch.value.imageMeanSizeCi95.n}` },
    ]
  }

  const isX = channelAxis.value === 'x'
  const ci = isX ? batch.value.imageMeanChannelWidthXCi95 : batch.value.imageMeanChannelWidthYCi95
  return [
    { label: `${isX ? '水平' : '垂直'}平均 W`, value: withUnit(isX ? batch.value.meanChannelWidthX : batch.value.meanChannelWidthY, batch.value.channelWidthUnit), note: '对象级均值' },
    { label: `${isX ? '水平' : '垂直'}中位 W`, value: withUnit(isX ? batch.value.medianChannelWidthX : batch.value.medianChannelWidthY, batch.value.channelWidthUnit), note: '对象级中位数' },
    { label: `${isX ? '水平' : '垂直'}标准差`, value: withUnit(isX ? batch.value.stdChannelWidthX : batch.value.stdChannelWidthY, batch.value.channelWidthUnit), note: '对象级波动' },
    { label: `${isX ? '水平' : '垂直'} 95% CI`, value: formatCiText(ci), note: `n = ${ci.n}` },
  ]
})

const ciCards = computed(() => {
  const ci = currentCi.value
  return [
    { label: '图像级样本数', value: String(ci.n), note: '95% CI 基于图像级样本而不是颗粒总数' },
    {
      label: '区间范围',
      value: selectedMetric.value === 'vf' ? formatCiText(ci, { percentage: true }) : formatCiText(ci),
      note: ci.available ? 'Student t 双侧 95% 置信区间' : ci.reason ?? '样本不足',
    },
    {
      label: 'CI 半宽',
      value:
        ci.available && ci.halfWidth != null
          ? formatValue(ci.halfWidth, currentUnit.value, { percentage: selectedMetric.value === 'vf' })
          : ci.reason ?? '不可估计',
      note: ci.available ? '表示均值估计的波动范围' : '需至少两张有效图像',
    },
  ]
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

const boxplotValues = computed(() => {
  if (selectedMetric.value === 'vf') return imageRows.value.map((row) => row.volumeFraction * 100)
  if (selectedMetric.value === 'particle_count') return imageRows.value.map((row) => row.particleCount)
  if (selectedMetric.value === 'area') return batch.value.areas
  if (selectedMetric.value === 'size') return batch.value.sizes
  return channelAxis.value === 'x' ? batch.value.channelWidthsX : batch.value.channelWidthsY
})

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

const buildBarChartOption = (labels: string[], values: number[], unit: string, options?: { percentage?: boolean; color?: string }) => ({
  animation: false,
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'shadow' },
    formatter: (params: any) => {
      const first = Array.isArray(params) ? params[0] : params
      if (!first) return ''
      const value = Number(first.value)
      return `${first.axisValue}<br/>${options?.percentage ? `${formatNumber(value, 2)}%` : formatValue(value, unit)}`
    },
  },
  grid: { left: 52, right: 24, top: 28, bottom: 72, containLabel: true },
  xAxis: {
    type: 'category',
    data: labels,
    axisLabel: {
      interval: 0,
      rotate: labels.length > 5 ? 24 : 0,
      color: '#6b7077',
    },
    axisLine: { lineStyle: { color: 'rgba(31, 40, 48, 0.16)' } },
  },
  yAxis: {
    type: 'value',
    axisLabel: {
      color: '#6b7077',
      formatter: (value: number) => (options?.percentage ? `${formatNumber(value, 1)}%` : `${formatNumber(value, 1)} ${unit}`.trim()),
    },
    splitLine: { lineStyle: { color: 'rgba(31, 40, 48, 0.08)' } },
  },
  series: [
    {
      type: 'bar',
      data: values,
      barMaxWidth: 34,
      itemStyle: {
        borderRadius: [10, 10, 4, 4],
        color: options?.color ?? '#176087',
      },
    },
  ],
})

const buildHistogramOption = (buckets: HistogramBucket[], unit: string, mode: HistogramMode, color: string) =>
  buildBarChartOption(
    buckets.map((bucket) => bucket.label),
    buckets.map((bucket) => (mode === 'probability' ? bucket.probability * 100 : bucket.count)),
    mode === 'probability' ? '%' : unit,
    { percentage: mode === 'probability', color },
  )

const buildBoxplotOption = (values: number[], unit: string, options?: { percentage?: boolean; label?: string }) => {
  if (values.length < 2) return null
  const clean = values.filter((item) => Number.isFinite(item))
  if (clean.length < 2) return null
  const sorted = [...clean].sort((left, right) => left - right)
  const data = [[sorted[0], quantile(sorted, 0.25), quantile(sorted, 0.5), quantile(sorted, 0.75), sorted[sorted.length - 1]]]
  return {
    animation: false,
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        const value = params?.data as number[] | undefined
        if (!value) return ''
        const formatter = (item: number) => (options?.percentage ? `${formatNumber(item, 2)}%` : `${formatNumber(item, 2)} ${unit}`.trim())
        return [
          options?.label ?? currentMetricLabel.value,
          `最小值：${formatter(value[0])}`,
          `Q1：${formatter(value[1])}`,
          `中位数：${formatter(value[2])}`,
          `Q3：${formatter(value[3])}`,
          `最大值：${formatter(value[4])}`,
        ].join('<br/>')
      },
    },
    grid: { left: 58, right: 24, top: 24, bottom: 48, containLabel: true },
    xAxis: {
      type: 'category',
      data: [options?.label ?? currentMetricLabel.value],
      axisLabel: { color: '#6b7077' },
      axisLine: { lineStyle: { color: 'rgba(31, 40, 48, 0.16)' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        color: '#6b7077',
        formatter: (value: number) => (options?.percentage ? `${formatNumber(value, 1)}%` : `${formatNumber(value, 1)} ${unit}`.trim()),
      },
      splitLine: { lineStyle: { color: 'rgba(31, 40, 48, 0.08)' } },
    },
    series: [
      {
        type: 'boxplot',
        data,
        itemStyle: {
          color: 'rgba(23, 96, 135, 0.14)',
          borderColor: '#176087',
          borderWidth: 2,
        },
      },
    ],
  }
}

const primaryChartOption = computed(() => {
  if (selectedMetric.value === 'vf') {
    return buildBarChartOption(imageSeries.value.map((item) => item.label), imageSeries.value.map((item) => item.value), '%', { percentage: true, color: '#176087' })
  }
  if (selectedMetric.value === 'particle_count') {
    return buildBarChartOption(imageSeries.value.map((item) => item.label), imageSeries.value.map((item) => item.value), '个', { color: '#b85a2b' })
  }
  if (selectedMetric.value === 'area') {
    return buildHistogramOption(batch.value.areaHistogram, batch.value.areaUnit, histogramMode.value, '#b85a2b')
  }
  if (selectedMetric.value === 'size') {
    return buildHistogramOption(batch.value.sizeHistogram, batch.value.sizeUnit, histogramMode.value, '#176087')
  }
  return buildHistogramOption(
    channelAxis.value === 'x' ? batch.value.channelWidthXHistogram : batch.value.channelWidthYHistogram,
    batch.value.channelWidthUnit,
    histogramMode.value,
    channelAxis.value === 'x' ? '#176087' : '#b85a2b',
  )
})

const boxplotOption = computed(() =>
  buildBoxplotOption(boxplotValues.value, currentUnit.value, {
    percentage: selectedMetric.value === 'vf',
    label: selectedMetric.value === 'channel_width' ? `${channelAxis.value === 'x' ? '水平' : '垂直'} W` : metricDefinition.value.label,
  }),
)

const primaryChartTitle = computed(() => {
  if (selectedMetric.value === 'vf') return '各图 Vf 柱状图'
  if (selectedMetric.value === 'particle_count') return '各图颗粒数柱状图'
  if (selectedMetric.value === 'area') return histogramMode.value === 'probability' ? '面积概率分布图' : '面积频数分布图'
  if (selectedMetric.value === 'size') return histogramMode.value === 'probability' ? '尺寸概率分布图' : '尺寸频数分布图'
  const axisLabel = channelAxis.value === 'x' ? '水平 γ 通道宽度' : '垂直 γ 通道宽度'
  return histogramMode.value === 'probability' ? `${axisLabel}概率分布图` : `${axisLabel}频数分布图`
})

const secondaryChartTitle = computed(() => {
  if (selectedMetric.value === 'vf') return 'Vf 箱线图'
  if (selectedMetric.value === 'particle_count') return '颗粒数箱线图'
  if (selectedMetric.value === 'area') return '面积箱线图'
  if (selectedMetric.value === 'size') return '尺寸箱线图'
  return `${channelAxis.value === 'x' ? '水平' : '垂直'} W 箱线图`
})

watch(activeModes, (modes) => {
  if (!modes.length) {
    selectedMode.value = ''
    return
  }
  if (!selectedMode.value || !modes.includes(selectedMode.value)) {
    selectedMode.value = modes[0]
  }
})

watch(selectedMetric, () => {
  imagePage.value = 1
  particlePage.value = 1
})

watch([imageSearch, selectedMode], () => {
  imagePage.value = 1
})

watch([particleSearch, particleFilteredOnly, selectedMode], () => {
  particlePage.value = 1
})

onMounted(async () => {
  try {
    await loadPayload()
    const firstMode = activeModes.value[0]
    if (firstMode) selectedMode.value = firstMode
    if (!payload.value || !['completed', 'failed', 'partial_success'].includes(payload.value.run.status)) {
      startPolling()
    }
  } catch {
    ElMessage.error('统计分析数据加载失败')
  }
})

onBeforeUnmount(() => {
  stopPolling()
})
</script>

<template>
  <div v-if="payload" class="statistics-shell">
    <section class="glass-card page-toolbar">
      <div class="toolbar-head">
        <div>
          <h2 class="section-title">统计分析</h2>
          <p class="section-subtitle">{{ payload.run.name }} · {{ activeModeLabel }}</p>
        </div>
        <div class="toolbar-actions">
          <span class="status-chip">{{ payload.run.status }}</span>
          <el-button plain @click="router.push(`/runs/${payload.run.id}`)">返回结果详情</el-button>
        </div>
      </div>

      <div class="control-toolbar">
        <div class="control-group">
          <span class="toolbar-label">模式</span>
          <div class="chip-row">
            <button
              v-for="option in modeOptions"
              :key="option.value"
              type="button"
              class="chip-button"
              :class="{ 'is-active': selectedMode === option.value }"
              @click="selectedMode = option.value"
            >
              {{ option.label }}
            </button>
          </div>
        </div>

        <div class="control-group control-group--wide">
          <span class="toolbar-label">统计量</span>
          <div class="chip-row">
            <button
              v-for="option in metricOptions"
              :key="option.value"
              type="button"
              class="chip-button"
              :class="{ 'is-active': selectedMetric === option.value }"
              @click="selectedMetric = option.value"
            >
              {{ option.label }}
            </button>
          </div>
        </div>

        <div v-if="selectedMetric === 'channel_width'" class="control-group">
          <span class="toolbar-label">方向</span>
          <div class="chip-row">
            <button type="button" class="chip-button" :class="{ 'is-active': channelAxis === 'x' }" @click="channelAxis = 'x'">水平</button>
            <button type="button" class="chip-button" :class="{ 'is-active': channelAxis === 'y' }" @click="channelAxis = 'y'">垂直</button>
          </div>
        </div>

        <div v-if="isDistributionMetric" class="control-group">
          <span class="toolbar-label">图表</span>
          <div class="chip-row">
            <button type="button" class="chip-button" :class="{ 'is-active': histogramMode === 'probability' }" @click="histogramMode = 'probability'">概率图</button>
            <button type="button" class="chip-button" :class="{ 'is-active': histogramMode === 'count' }" @click="histogramMode = 'count'">频数图</button>
          </div>
        </div>
      </div>
    </section>

    <section class="summary-grid">
      <article v-for="(card, index) in metricCards" :key="card.label" class="glass-card summary-card" :class="{ 'summary-card--primary': index === 0 }">
        <span>{{ card.label }}</span>
        <strong>{{ card.value }}</strong>
        <p>{{ card.note }}</p>
      </article>
    </section>

    <section class="glass-card chart-card">
      <div class="panel-head">
        <div>
          <h3 class="section-title">{{ primaryChartTitle }}</h3>
          <p class="section-subtitle">{{ currentMetricLabel }}</p>
        </div>
        <div class="meta-chips">
          <span class="meta-chip">图像 {{ batch.imageCount }} 张</span>
          <span class="meta-chip">对象 {{ batch.totalParticleCount }} 个</span>
        </div>
      </div>

      <ChartCanvas :option="primaryChartOption" height="360px" />
    </section>

    <section class="analysis-grid">
      <section class="glass-card ci-card">
        <div class="panel-head">
          <div>
            <h3 class="section-title">95% CI</h3>
            <p class="section-subtitle">按图像级样本估计均值区间，便于答辩展示统计稳定性。</p>
          </div>
        </div>

        <div class="ci-grid">
          <article v-for="card in ciCards" :key="card.label" class="ci-item">
            <span>{{ card.label }}</span>
            <strong>{{ card.value }}</strong>
            <p>{{ card.note }}</p>
          </article>
        </div>
      </section>

      <section v-if="boxplotOption" class="glass-card chart-card chart-card--secondary">
        <div class="panel-head">
          <div>
            <h3 class="section-title">{{ secondaryChartTitle }}</h3>
            <p class="section-subtitle">用于查看离散程度、中位数和异常波动。</p>
          </div>
        </div>
        <ChartCanvas :option="boxplotOption" height="280px" />
      </section>
    </section>

    <section class="glass-card detail-card">
      <div class="panel-head">
        <div>
          <h3 class="section-title">统计明细</h3>
          <p class="section-subtitle">图像级用于横向比较，颗粒级用于对象追踪与筛选。</p>
        </div>
      </div>

      <el-tabs v-model="detailTab">
        <el-tab-pane label="图像级统计" name="images">
          <div class="table-toolbar">
            <el-input v-model="imageSearch" clearable placeholder="按图像名称搜索" />
            <div class="table-meta">共 {{ filteredImageRows.length }} 行</div>
          </div>

          <div class="table-shell">
            <el-table :data="paginatedImageRows" stripe height="380">
              <el-table-column prop="imageName" label="图像名称" min-width="160" />
              <el-table-column label="Vf" width="110">
                <template #default="{ row }">{{ percent(row.volumeFraction) }}</template>
              </el-table-column>
              <el-table-column prop="particleCount" label="颗粒数" width="92" />
              <el-table-column label="平均面积" min-width="132">
                <template #default="{ row }">{{ withUnit(row.meanArea, row.areaUnit) }}</template>
              </el-table-column>
              <el-table-column label="平均尺寸" min-width="128">
                <template #default="{ row }">{{ withUnit(row.meanSize, row.sizeUnit) }}</template>
              </el-table-column>
              <el-table-column label="中位面积" min-width="128">
                <template #default="{ row }">{{ withUnit(row.medianArea, row.areaUnit) }}</template>
              </el-table-column>
              <el-table-column label="中位尺寸" min-width="128">
                <template #default="{ row }">{{ withUnit(row.medianSize, row.sizeUnit) }}</template>
              </el-table-column>
              <el-table-column label="水平 W" min-width="120">
                <template #default="{ row }">{{ withUnit(row.meanChannelWidthX, row.channelWidthUnit) }}</template>
              </el-table-column>
              <el-table-column label="垂直 W" min-width="120">
                <template #default="{ row }">{{ withUnit(row.meanChannelWidthY, row.channelWidthUnit) }}</template>
              </el-table-column>
            </el-table>
          </div>

          <div class="pagination-bar">
            <el-pagination
              v-model:current-page="imagePage"
              :page-size="IMAGE_PAGE_SIZE"
              layout="prev, pager, next"
              :total="filteredImageRows.length"
              small
            />
          </div>
        </el-tab-pane>

        <el-tab-pane label="颗粒级统计" name="particles" lazy>
          <div class="table-toolbar">
            <el-input v-model="particleSearch" clearable placeholder="按图像名、编号或过滤原因搜索" />
            <el-switch v-model="particleFilteredOnly" inline-prompt active-text="仅看过滤对象" inactive-text="全部对象" />
            <div class="table-meta">共 {{ filteredParticleRows.length }} 行</div>
          </div>

          <div class="table-shell">
            <el-table :data="paginatedParticleRows" stripe height="420">
              <el-table-column prop="imageName" label="图像名称" min-width="150" />
              <el-table-column prop="label" label="编号" width="82" />
              <el-table-column label="面积" min-width="120">
                <template #default="{ row }">{{ withUnit(row.areaValue, batch.areaUnit) }}</template>
              </el-table-column>
              <el-table-column label="尺寸" min-width="120">
                <template #default="{ row }">{{ withUnit(row.sizeValue, batch.sizeUnit) }}</template>
              </el-table-column>
              <el-table-column label="周长" min-width="120">
                <template #default="{ row }">{{ withUnit(row.perimeter, batch.sizeUnit) }}</template>
              </el-table-column>
              <el-table-column label="Major" min-width="120">
                <template #default="{ row }">{{ withUnit(row.major, batch.sizeUnit) }}</template>
              </el-table-column>
              <el-table-column label="Minor" min-width="120">
                <template #default="{ row }">{{ withUnit(row.minor, batch.sizeUnit) }}</template>
              </el-table-column>
              <el-table-column label="Feret" min-width="120">
                <template #default="{ row }">{{ withUnit(row.feret, batch.sizeUnit) }}</template>
              </el-table-column>
              <el-table-column label="MinFeret" min-width="120">
                <template #default="{ row }">{{ withUnit(row.minFeret, batch.sizeUnit) }}</template>
              </el-table-column>
              <el-table-column label="长宽比" min-width="100">
                <template #default="{ row }">{{ formatNumber(row.aspectRatio, 2) }}</template>
              </el-table-column>
              <el-table-column label="圆度" min-width="96">
                <template #default="{ row }">{{ formatNumber(row.circularity, 3) }}</template>
              </el-table-column>
              <el-table-column label="Roundness" min-width="110">
                <template #default="{ row }">{{ formatNumber(row.roundness, 3) }}</template>
              </el-table-column>
              <el-table-column label="Solidity" min-width="100">
                <template #default="{ row }">{{ formatNumber(row.solidity, 3) }}</template>
              </el-table-column>
              <el-table-column label="被过滤" width="88">
                <template #default="{ row }">{{ row.filtered ? '是' : '否' }}</template>
              </el-table-column>
              <el-table-column prop="filterReason" label="过滤原因" min-width="160" />
            </el-table>
          </div>

          <div class="pagination-bar">
            <el-pagination
              v-model:current-page="particlePage"
              :page-size="PARTICLE_PAGE_SIZE"
              layout="prev, pager, next"
              :total="filteredParticleRows.length"
              small
            />
          </div>
        </el-tab-pane>
      </el-tabs>
    </section>

    <section class="glass-card info-card">
      <el-collapse v-model="infoPanels">
        <el-collapse-item name="calibration" title="采集与标定信息">
          <div class="info-grid">
            <article v-for="item in calibrationItems" :key="item.label" class="info-item">
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
            </article>
          </div>
        </el-collapse-item>
        <el-collapse-item name="formula" title="公式与统计口径">
          <div class="formula-stack">
            <div class="formula-copy">
              <h3 class="section-title">{{ metricDefinition.title }}</h3>
              <p class="section-subtitle">{{ metricDefinition.description }}</p>
            </div>
            <code class="formula-block">{{ metricDefinition.formula }}</code>
          </div>
        </el-collapse-item>
      </el-collapse>
    </section>
  </div>

  <div v-else class="statistics-shell">
    <section class="glass-card loading-card">
      <el-skeleton :rows="10" animated />
    </section>
  </div>
</template>

<style scoped>
.statistics-shell {
  display: grid;
  gap: 16px;
  min-width: 0;
  width: 100%;
}

.page-toolbar,
.chart-card,
.ci-card,
.detail-card,
.info-card,
.loading-card,
.summary-card {
  padding: 18px;
}

.toolbar-head,
.panel-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 14px;
}

.toolbar-actions,
.meta-chips {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.control-toolbar {
  display: grid;
  gap: 14px;
  margin-top: 16px;
}

.control-group {
  display: grid;
  gap: 8px;
}

.control-group--wide {
  min-width: 0;
}

.toolbar-label {
  font-size: 12px;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.chip-button {
  min-width: 0;
  padding: 10px 16px;
  border-radius: 999px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.74);
  color: var(--muted);
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease;
}

.chip-button:hover {
  transform: translateY(-1px);
  border-color: rgba(23, 96, 135, 0.16);
}

.chip-button.is-active {
  color: #fff;
  border-color: transparent;
  background: linear-gradient(135deg, #176087, #2f8bc0);
  box-shadow: 0 12px 22px rgba(23, 96, 135, 0.22);
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
  gap: 12px;
}

.summary-card {
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.66);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.summary-card--primary {
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.12), rgba(184, 90, 43, 0.08));
  border-color: rgba(23, 96, 135, 0.12);
}

.summary-card span,
.ci-item span,
.info-item span {
  display: block;
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 12px;
}

.summary-card strong,
.ci-item strong,
.info-item strong {
  display: block;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.summary-card p,
.ci-item p {
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.55;
}

.meta-chip {
  display: inline-flex;
  align-items: center;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
  gap: 16px;
  min-width: 0;
}

.ci-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 180px), 1fr));
  gap: 12px;
}

.ci-item,
.info-item {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.66);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.table-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}

.table-toolbar :deep(.el-input) {
  flex: 1 1 240px;
  min-width: 220px;
}

.table-meta {
  color: var(--muted);
  font-size: 12px;
  margin-left: auto;
}

.table-shell {
  width: 100%;
  min-width: 0;
  overflow-x: auto;
  overflow-y: hidden;
  padding-bottom: 4px;
  -webkit-overflow-scrolling: touch;
  scrollbar-gutter: stable both-edges;
}

.pagination-bar {
  display: flex;
  justify-content: flex-end;
  margin-top: 14px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 148px), 1fr));
  gap: 12px;
}

.formula-stack {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(260px, 0.8fr);
  gap: 16px;
  align-items: start;
}

.formula-copy {
  display: grid;
  gap: 8px;
}

.formula-block {
  display: grid;
  place-items: center;
  min-height: 100%;
  padding: 18px 20px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.08), rgba(184, 90, 43, 0.06));
  border: 1px solid rgba(23, 96, 135, 0.1);
  color: var(--accent);
  font-size: 18px;
  font-weight: 700;
  line-height: 1.7;
  text-align: center;
  white-space: normal;
  overflow-wrap: anywhere;
}

.info-item {
  min-height: 92px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.formula-copy .section-title {
  margin-bottom: 4px;
}

.formula-copy .section-subtitle {
  line-height: 1.7;
}

:deep(.info-card .el-collapse) {
  display: grid;
  gap: 12px;
  border: 0;
  background: transparent;
}

:deep(.info-card .el-collapse-item) {
  border: 1px solid rgba(31, 40, 48, 0.08);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.56);
  overflow: hidden;
}

:deep(.info-card .el-collapse-item__header) {
  min-height: 56px;
  padding: 0 18px;
  border-bottom: 0;
  background: transparent;
  color: var(--ink);
  font-size: 15px;
  font-weight: 700;
}

:deep(.info-card .el-collapse-item__wrap) {
  border-bottom: 0;
  background: transparent;
}

:deep(.info-card .el-collapse-item__content) {
  padding: 18px;
}

.loading-card {
  padding: 24px;
}

@media (max-width: 920px) {
  .toolbar-head,
  .panel-head {
    flex-direction: column;
    align-items: stretch;
  }

  .table-meta {
    margin-left: 0;
  }
}

@media (max-width: 640px) {
  .page-toolbar,
  .chart-card,
  .ci-card,
  .detail-card,
  .info-card,
  .summary-card {
    padding: 16px;
  }

  .chip-row {
    gap: 8px;
  }

  .chip-button {
    width: 100%;
    justify-content: center;
  }

  .table-toolbar {
    align-items: stretch;
  }

  .formula-stack {
    grid-template-columns: 1fr;
  }
}
</style>
