<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { BarChart, BoxplotChart, LineChart, ScatterChart } from 'echarts/charts'
import { DatasetComponent, GraphicComponent, GridComponent, MarkAreaComponent, TooltipComponent, TransformComponent } from 'echarts/components'
import { use, init, type EChartsType, type EChartsCoreOption } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'

const props = defineProps<{
  option: EChartsCoreOption
  height?: string
}>()

use([
  BarChart,
  BoxplotChart,
  LineChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  DatasetComponent,
  TransformComponent,
  GraphicComponent,
  MarkAreaComponent,
  CanvasRenderer,
])

const chartRef = ref<HTMLDivElement | null>(null)
let chart: EChartsType | null = null
let resizeObserver: ResizeObserver | null = null
let optionFrame = 0
let resizeFrame = 0
let visibilityFrame = 0
let lastWidth = 0
let lastHeight = 0

const ensureChart = () => {
  if (!chartRef.value) return
  if (!chart) {
    if (chartRef.value.clientWidth <= 0 || chartRef.value.clientHeight <= 0) return
    chart = init(chartRef.value, undefined, {
      renderer: 'canvas',
      useDirtyRect: true,
      devicePixelRatio: Math.min(window.devicePixelRatio || 1, 1.5),
    })
  }
}

const applyOption = () => {
  ensureChart()
  if (!chart) return
  chart.setOption(
    {
      animation: false,
      stateAnimation: { duration: 0 },
      ...props.option,
    },
    { notMerge: true, lazyUpdate: true, silent: true },
  )
}

const queueOptionUpdate = () => {
  if (optionFrame) {
    cancelAnimationFrame(optionFrame)
  }
  optionFrame = requestAnimationFrame(() => {
    optionFrame = 0
    applyOption()
  })
}

const queueResize = () => {
  if (resizeFrame) {
    cancelAnimationFrame(resizeFrame)
  }
  resizeFrame = requestAnimationFrame(() => {
    resizeFrame = 0
    if (!chartRef.value || !chart) return
    const nextWidth = chartRef.value.clientWidth
    const nextHeight = chartRef.value.clientHeight
    if (nextWidth <= 0 || nextHeight <= 0) {
      return
    }
    if (nextWidth === lastWidth && nextHeight === lastHeight) {
      return
    }
    lastWidth = nextWidth
    lastHeight = nextHeight
    chart.resize()
  })
}

const handleVisibilityChange = () => {
  if (visibilityFrame) {
    cancelAnimationFrame(visibilityFrame)
  }
  visibilityFrame = requestAnimationFrame(() => {
    visibilityFrame = 0
    queueResize()
  })
}

const exportDataUrl = (type: 'png' | 'jpeg' = 'png') => {
  ensureChart()
  if (chart) {
    chart.resize()
  }
  if (!chart) return ''
  return chart.getDataURL({
    type,
    pixelRatio: 2,
    backgroundColor: '#ffffff',
  })
}

defineExpose({
  getChartInstance: () => chart,
  exportDataUrl,
})

onMounted(() => {
  void nextTick().then(() => {
    ensureChart()
    lastWidth = chartRef.value?.clientWidth ?? 0
    lastHeight = chartRef.value?.clientHeight ?? 0
    queueOptionUpdate()
    queueResize()
    if (chartRef.value) {
      resizeObserver = new ResizeObserver(() => queueResize())
      resizeObserver.observe(chartRef.value)
    }
  })
  document.addEventListener('visibilitychange', handleVisibilityChange)
})

watch(
  () => props.option,
  () => queueOptionUpdate(),
)

onBeforeUnmount(() => {
  if (optionFrame) {
    cancelAnimationFrame(optionFrame)
  }
  if (resizeFrame) {
    cancelAnimationFrame(resizeFrame)
  }
  if (visibilityFrame) {
    cancelAnimationFrame(visibilityFrame)
  }
  document.removeEventListener('visibilitychange', handleVisibilityChange)
  resizeObserver?.disconnect()
  chart?.dispose()
})
</script>

<template>
  <div class="chart-canvas-shell">
    <div ref="chartRef" class="chart-canvas" :style="{ width: '100%', height: height ?? '220px' }"></div>
  </div>
</template>

<style scoped>
.chart-canvas-shell {
  width: 100%;
  min-width: 0;
  border-radius: 14px;
}

.chart-canvas {
  display: block;
  min-width: 0;
  contain: layout paint size;
}
</style>
