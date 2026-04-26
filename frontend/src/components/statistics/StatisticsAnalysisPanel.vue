<script setup lang="ts">
import { computed, ref } from 'vue'

import ChartCanvas from '../ChartCanvas.vue'

type ChartKind = 'distribution' | 'by_image'

type RankedImagePoint = {
  label: string
  value: number
  rank: number
}

type HistogramBin = {
  start: number
  end: number
  center: number
  count: number
  ratio: number
}

const props = defineProps<{
  title: string
  subtitle: string
  kind: ChartKind
  metricLabel: string
  unitLabel: string
  values: number[]
  imageSeries: Array<{ label: string; value: number }>
  imageMetricLabel: string
}>()

const upperChartRef = ref<InstanceType<typeof ChartCanvas> | null>(null)

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

const normalizeUnitLabel = (unit: string) => {
  if (unit === 'um') return 'μm'
  if (unit === 'um^2') return 'μm²'
  if (unit === 'px^2') return 'px²'
  return unit
}

const normalizedUnitLabel = computed(() => normalizeUnitLabel(props.unitLabel))

const resolveAxisDigits = (min: number, max: number) => {
  const span = Math.abs(max - min)
  if (span <= 1) return 2
  if (span <= 10) return 1
  return 0
}

const resolveBinDigits = (step: number, min: number, max: number) => {
  const spanDigits = resolveAxisDigits(min, max)
  if (step < 0.1) return Math.max(spanDigits, 3)
  if (step < 1) return Math.max(spanDigits, 2)
  if (step < 10) return Math.max(spanDigits, 1)
  return spanDigits
}

const formatAxisTick = (value: number, digits: number) => {
  if (!Number.isFinite(value)) return ''
  return value
    .toFixed(digits)
    .replace(/\.0+$/, '')
    .replace(/(\.\d*?)0+$/, '$1')
}

const formatMetricValue = (value: number, unit: string, digits = 2) => {
  if (!Number.isFinite(value)) return '--'
  if (unit === '%') return `${value.toFixed(digits)}%`
  if (unit === '个') return `${Math.round(value)}`
  return `${value.toFixed(digits)} ${normalizeUnitLabel(unit)}`.trim()
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

const formatMetricInterval = (start: number, end: number, unit: string, digits = 2) => {
  return `${formatAxisTick(start, digits)} - ${formatAxisTick(end, digits)} ${normalizeUnitLabel(unit)}`.trim()
}

const buildEmptyOption = (message: string) => ({
  animation: false,
  xAxis: { type: 'value', show: false },
  yAxis: { type: 'value', show: false },
  series: [],
  graphic: [
    {
      type: 'group',
      left: 'center',
      top: 'middle',
      children: [
        {
          type: 'text',
          style: {
            text: message,
            fill: '#6b7077',
            fontSize: 14,
            fontWeight: 600,
          },
        },
      ],
    },
  ],
})

const distributionValues = computed(() =>
  props.values
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item) && item >= 0)
    .sort((left, right) => left - right),
)

const hasDistributionData = computed(() => distributionValues.value.length > 0)
const hasByImageData = computed(() => rankedImageSeries.value.length > 0)
const hasChartData = computed(() =>
  props.kind === 'distribution' ? hasDistributionData.value : hasByImageData.value,
)

const emptyState = computed(() => {
  if (props.kind === 'distribution') {
    return {
      title: '当前指标暂无可绘制样本',
      description: '还没有足够的连续测量值用于生成分布曲线与箱线摘要，请先确认统计结果是否已产出。',
    }
  }
  return {
    title: '当前模式暂无图片级汇总',
    description: '按图片统计尚未形成有效序列，因此暂时无法展示排序曲线，请先检查任务结果是否完整。',
  }
})

const rankedImageSeries = computed<RankedImagePoint[]>(() =>
  props.imageSeries
    .map((item) => ({ label: String(item.label), value: Number(item.value) }))
    .filter((item) => item.label && Number.isFinite(item.value))
    .sort((left, right) => right.value - left.value || left.label.localeCompare(right.label, 'zh-CN'))
    .map((item, index) => ({
      ...item,
      rank: index + 1,
    })),
)

const imageRankLabelStep = computed(() => {
  const count = rankedImageSeries.value.length
  if (count <= 12) return 1
  return Math.ceil(count / 12)
})

const byImageStats = computed(() => {
  const values = rankedImageSeries.value.map((item) => item.value)
  if (!values.length) {
    return {
      min: 0,
      max: 0,
      mean: 0,
      median: 0,
    }
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  return {
    min,
    max,
    mean,
    median: quantile(values, 0.5),
  }
})

const buildHistogramBins = (values: number[]) => {
  if (!values.length) {
    return {
      bins: [] as HistogramBin[],
      step: 0,
      digits: 0,
    }
  }

  const min = values[0]
  const max = values[values.length - 1]
  if (min === max) {
    const step = Math.max(Math.abs(min) * 0.1, 0.1)
    return {
      bins: [
        {
          start: Math.max(0, min - step / 2),
          end: min + step / 2,
          center: min,
          count: values.length,
          ratio: 1,
        },
      ],
      step,
      digits: resolveBinDigits(step, min, max || min + step),
    }
  }

  const binCount = clamp(Math.round(Math.log2(values.length) + 1), 8, 18)
  const step = Math.max((max - min) / binCount, 1e-6)
  const bins = Array.from({ length: binCount }, (_, index) => {
    const start = min + step * index
    const end = index === binCount - 1 ? max + step * 1e-3 : start + step
    return {
      start,
      end,
      center: start + (end - start) / 2,
      count: 0,
      ratio: 0,
    }
  })

  for (const value of values) {
    const rawIndex = Math.floor((value - min) / step)
    const targetIndex = Math.min(Math.max(rawIndex, 0), bins.length - 1)
    bins[targetIndex].count += 1
  }

  for (const bin of bins) {
    bin.ratio = bin.count / values.length
  }

  return {
    bins,
    step,
    digits: resolveBinDigits(step, min, max),
  }
}

const histogram = computed(() => buildHistogramBins(distributionValues.value))

const boxSummaryStats = computed(() => {
  const values = distributionValues.value
  if (!values.length) {
    return null
  }
  const min = values[0]
  const q1 = quantile(values, 0.25)
  const median = quantile(values, 0.5)
  const q3 = quantile(values, 0.75)
  const max = values[values.length - 1]
  const iqr = q3 - q1
  const upperFence = q3 + 1.5 * iqr
  return {
    min,
    q1,
    median,
    q3,
    max,
    outliers: values.filter((value) => value > upperFence).slice(0, 8),
  }
})

const distributionOption = computed(() => {
  if (!distributionValues.value.length || !histogram.value.bins.length) {
    return buildEmptyOption('暂无连续指标样本')
  }

  const bins = histogram.value.bins
  const axisDigits = histogram.value.digits
  const categoryLabels = bins.map((bin) => formatAxisTick(bin.center, axisDigits))

  return {
    animation: false,
    grid: {
      left: 104,
      right: 36,
      top: 52,
      bottom: 58,
      containLabel: true,
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
        shadowStyle: {
          color: 'rgba(31, 40, 48, 0.04)',
        },
      },
      formatter: (params: any[]) => {
        const primary = Array.isArray(params) ? params[0] : params
        const bin = bins[primary?.dataIndex ?? 0]
        if (!bin) return ''
        return [
          `区间：${formatMetricInterval(bin.start, bin.end, props.unitLabel, axisDigits)}`,
          `颗粒数：${bin.count} 个`,
          `占比：${(bin.ratio * 100).toFixed(1)}%`,
        ].join('<br/>')
      },
    },
    xAxis: {
      type: 'category',
      data: categoryLabels,
      name: `${props.metricLabel}（${normalizedUnitLabel.value}）`,
      nameLocation: 'middle',
      nameGap: 36,
      axisLabel: {
        color: '#6b7077',
        hideOverlap: true,
        margin: 12,
        formatter: (value: string, index: number) => {
          const labelStep = categoryLabels.length <= 10 ? 1 : Math.ceil(categoryLabels.length / 10)
          if (index % labelStep !== 0 && index !== categoryLabels.length - 1) return ''
          return value
        },
      },
      nameTextStyle: {
        color: '#5b6573',
        fontSize: 12,
        fontWeight: 700,
        padding: [12, 0, 0, 0],
      },
      axisLine: { lineStyle: { color: 'rgba(31, 40, 48, 0.16)' } },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: '颗粒数（个）',
      nameLocation: 'middle',
      nameRotate: 90,
      nameGap: 72,
      axisLabel: {
        show: true,
        color: '#6b7077',
        margin: 12,
        formatter: (value: number) => `${Math.round(value)}`,
      },
      nameTextStyle: {
        color: '#5b6573',
        fontSize: 12,
        fontWeight: 700,
      },
      axisLine: { show: false },
      splitLine: {
        lineStyle: {
          color: 'rgba(31, 40, 48, 0.07)',
        },
      },
    },
    series: [
      {
        name: 'histogram',
        type: 'bar',
        data: bins.map((bin) => bin.count),
        barGap: '-100%',
        barCategoryGap: '0%',
        itemStyle: {
          color: 'rgba(32, 84, 147, 0.16)',
          borderColor: 'rgba(22, 50, 79, 0.14)',
          borderWidth: 1,
        },
        emphasis: {
          itemStyle: {
            color: 'rgba(32, 84, 147, 0.22)',
          },
        },
        z: 1,
      },
    ],
  }
})

const boxplotOption = computed(() => {
  const stats = boxSummaryStats.value
  if (!stats) return null
  const yMin = Math.max(0, stats.min - (stats.max - stats.min) * 0.06)
  const yMax = stats.max + (stats.max - stats.min || 1) * 0.08
  return {
    animation: false,
    grid: {
      left: 36,
      right: 36,
      top: 18,
      bottom: 18,
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: ['箱线摘要'],
      axisTick: { show: false },
      axisLabel: { show: false },
      axisLine: { show: false },
    },
    yAxis: {
      type: 'value',
      min: yMin,
      max: yMax,
      axisTick: { show: false },
      axisLabel: { show: false },
      axisLine: { show: false },
      splitLine: { show: false },
    },
    series: [
      {
        type: 'boxplot',
        data: [[stats.min, stats.q1, stats.median, stats.q3, stats.max]],
        itemStyle: {
          color: 'rgba(22, 50, 79, 0.08)',
          borderColor: '#5d6a79',
          borderWidth: 1.4,
        },
      },
      {
        type: 'scatter',
        data: stats.outliers.map((value) => [0, value]),
        symbolSize: 7,
        itemStyle: {
          color: '#a3adb8',
          borderColor: '#ffffff',
          borderWidth: 1,
        },
        tooltip: { show: false },
      },
    ],
  }
})

const boxSummaryCaption = computed(() => {
  const stats = boxSummaryStats.value
  if (!stats) return ''
  return `Q1 ${formatMetricValue(stats.q1, props.unitLabel)} · 中位数 ${formatMetricValue(stats.median, props.unitLabel)} · Q3 ${formatMetricValue(stats.q3, props.unitLabel)}`
})

const chartAxisSummary = computed(() => {
  if (props.kind === 'distribution') {
    return {
      x: `横轴：${props.metricLabel}（${normalizedUnitLabel.value}）`,
      y: '纵轴：落入各区间的颗粒数（个）',
    }
  }

  return {
    x: '横轴：按当前指标降序排列的图片位次',
    y: `纵轴：${props.imageMetricLabel}`,
  }
})

const imageSummaryOption = computed(() => {
  const sorted = rankedImageSeries.value
  if (!sorted.length) {
    return buildEmptyOption('暂无图片级汇总样本')
  }

  const rankLabels = sorted.map((item) => String(item.rank))
  const stats = byImageStats.value
  const pointData = sorted.map((item, index) => ({
    value: item.value,
    itemStyle: {
      color: index < 3 ? '#0b2d5c' : '#1f6b94',
      borderColor: '#ffffff',
      borderWidth: 1.5,
    },
  }))

  return {
    animation: false,
    grid: {
      left: 86,
      right: 34,
      top: 42,
      bottom: 64,
      containLabel: true,
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
        shadowStyle: {
          color: 'rgba(31, 40, 48, 0.04)',
        },
      },
      formatter: (params: any[]) => {
        const primary = Array.isArray(params)
          ? params.find((item) => item.seriesName === 'point') ?? params.find((item) => item.seriesName === 'stem')
          : params
        if (!primary) return ''
        const row = sorted[primary.dataIndex]
        if (!row) return ''
        return [
          `排序位次：#${row.rank}`,
          `图片：${row.label}`,
          `${props.imageMetricLabel}：${formatMetricValue(row.value, props.unitLabel)}`,
        ].join('<br/>')
      },
    },
    xAxis: {
      type: 'category',
      data: rankLabels,
      name: '按值降序的图片位次',
      nameLocation: 'middle',
      nameGap: 40,
      axisLabel: {
        color: '#6b7077',
        hideOverlap: true,
        margin: 12,
        formatter: (value: string, index: number) => {
          if (index % imageRankLabelStep.value !== 0 && index !== rankLabels.length - 1) {
            return ''
          }
          return `#${value}`
        },
      },
      nameTextStyle: {
        color: '#5b6573',
        fontSize: 12,
        fontWeight: 700,
        padding: [14, 0, 0, 0],
      },
      axisLine: { lineStyle: { color: 'rgba(31, 40, 48, 0.16)' } },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: '',
      min: Math.min(0, stats.min * 0.96),
      max: stats.max * 1.08,
      axisLabel: {
        color: '#6b7077',
        margin: 12,
        formatter: (value: number) => {
          if (props.unitLabel === '个') return `${Math.round(value)}`
          return props.unitLabel === '%' ? `${value.toFixed(1)}%` : `${value.toFixed(1)}`
        },
      },
      splitLine: {
        lineStyle: {
          color: 'rgba(31, 40, 48, 0.08)',
        },
      },
    },
    series: [
      {
        name: 'stem',
        type: 'bar',
        data: sorted.map((item) => item.value),
        barMaxWidth: 10,
        itemStyle: {
          color: 'rgba(22, 50, 79, 0.2)',
          borderRadius: [6, 6, 0, 0],
        },
        emphasis: {
          disabled: true,
        },
        z: 1,
      },
      {
        name: 'median',
        type: 'line',
        data: sorted.map(() => stats.median),
        symbol: 'none',
        silent: true,
        tooltip: { show: false },
        lineStyle: {
          color: 'rgba(93, 106, 121, 0.9)',
          width: 1.3,
          type: 'dashed',
        },
        z: 2,
      },
      {
        name: 'point',
        type: 'scatter',
        data: pointData,
        symbol: 'circle',
        symbolSize: 10,
        z: 4,
      },
    ],
  }
})

const exportChart = () => upperChartRef.value?.exportDataUrl('png') ?? ''

defineExpose({
  exportChart,
})
</script>

<template>
  <section class="glass-card analysis-panel">
    <header class="analysis-panel__head">
      <div class="analysis-panel__copy">
        <span class="analysis-panel__eyebrow">
          {{ kind === 'distribution' ? '分布分析' : '图片汇总' }}
        </span>
        <h3 class="section-title">{{ title }}</h3>
        <p class="section-subtitle">{{ subtitle }}</p>
      </div>
    </header>

    <div class="analysis-panel__chart-shell" :class="{ 'analysis-panel__chart-shell--distribution': kind === 'distribution' }">
      <div v-if="hasChartData" class="analysis-panel__axis-summary" aria-label="图表轴说明">
        <span>{{ chartAxisSummary.x }}</span>
        <span>{{ chartAxisSummary.y }}</span>
      </div>
      <ChartCanvas
        v-if="hasChartData"
        ref="upperChartRef"
        :option="kind === 'distribution' ? distributionOption : imageSummaryOption"
        height="360px"
      />
      <div v-else class="analysis-panel__empty-state" role="status" aria-live="polite">
        <div class="analysis-panel__empty-icon" aria-hidden="true">
          <svg viewBox="0 0 48 48" fill="none">
            <rect x="8" y="10" width="32" height="24" rx="6" />
            <path d="M15 28l6-7 5 5 7-9" />
            <path d="M15 36h18" />
          </svg>
        </div>
        <strong>{{ emptyState.title }}</strong>
        <p>{{ emptyState.description }}</p>
      </div>
    </div>

    <div v-if="kind === 'distribution' && boxplotOption && hasChartData" class="box-summary">
      <ChartCanvas :option="boxplotOption" height="92px" />
      <div class="box-summary__caption">{{ boxSummaryCaption }}</div>
    </div>
  </section>
</template>

<style scoped>
.analysis-panel {
  display: grid;
  gap: 12px;
  padding: 18px;
}

.analysis-panel__head {
  display: block;
}

.analysis-panel__copy {
  display: grid;
  gap: 3px;
}

.analysis-panel__eyebrow {
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.section-title {
  margin-bottom: 0;
}

.analysis-panel__chart-shell {
  padding: 8px 8px 0;
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(248, 250, 252, 0.92));
  min-height: 346px;
}

.analysis-panel__chart-shell--distribution {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(247, 249, 252, 0.96));
}

.analysis-panel__axis-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 10px 14px 2px;
  color: #66707d;
  font-size: 12px;
  line-height: 1.5;
}

.analysis-panel__axis-summary span:last-child {
  text-align: right;
}

.analysis-panel__empty-state {
  min-height: 336px;
  display: grid;
  place-content: center;
  justify-items: center;
  gap: 10px;
  padding: 28px 32px;
  text-align: center;
}

.analysis-panel__empty-icon {
  width: 56px;
  height: 56px;
  display: grid;
  place-items: center;
  border-radius: 18px;
  background: rgba(22, 50, 79, 0.06);
  border: 1px solid rgba(22, 50, 79, 0.08);
  color: #516173;
}

.analysis-panel__empty-icon svg {
  width: 28px;
  height: 28px;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.analysis-panel__empty-state strong {
  color: var(--ink);
  font-size: 16px;
  font-weight: 700;
  line-height: 1.4;
}

.analysis-panel__empty-state p {
  max-width: 340px;
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.7;
}

.box-summary {
  display: grid;
  gap: 6px;
  padding: 6px 8px 0;
  border-top: 1px solid rgba(31, 40, 48, 0.06);
}

.box-summary__caption {
  color: var(--muted);
  font-size: 12px;
  text-align: center;
  letter-spacing: 0.01em;
  line-height: 1.5;
}

@media (max-width: 960px) {
  .analysis-panel__axis-summary {
    flex-direction: column;
    align-items: flex-start;
  }

  .analysis-panel__axis-summary span:last-child {
    text-align: left;
  }
}
</style>
