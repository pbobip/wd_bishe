<script setup lang="ts">
import { computed } from 'vue'

import type { RunRecord } from '../types'

const props = defineProps<{
  run: RunRecord | null | undefined
}>()

const snapshot = computed(() => {
  const summary = (props.run?.summary ?? {}) as Record<string, unknown>
  const candidates = [
    summary.calibration_probe,
    summary.calibration_prompt,
    summary.calibration,
  ]
  for (const candidate of candidates) {
    if (candidate && typeof candidate === 'object') {
      return candidate as Record<string, unknown>
    }
  }
  return null
})

const snapshotRows = computed(() => {
  if (!snapshot.value) return []
  const readNumber = (value: unknown, suffix = ''): string => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return `${value}${suffix}`
    }
    return '--'
  }
  const readText = (value: unknown, suffix = ''): string => {
    if (typeof value === 'string' && value.trim()) {
      return `${value.trim()}${suffix}`
    }
    return '--'
  }
  return [
    { label: '比例尺', value: readNumber(snapshot.value.ocr_scale_bar_um, ' μm') },
    { label: '比例尺像素', value: readNumber(snapshot.value.scale_bar_pixels, ' px') },
    { label: 'FoV', value: readNumber(snapshot.value.ocr_fov_um, ' μm') },
    { label: '放大倍数', value: readText(snapshot.value.ocr_magnification_text) },
    { label: 'WD', value: readNumber(snapshot.value.ocr_wd_mm, ' mm') },
    { label: '探测器', value: readText(snapshot.value.ocr_detector) },
    { label: '扫描模式', value: readText(snapshot.value.ocr_scan_mode) },
    { label: '真空模式', value: readText(snapshot.value.ocr_vacuum_mode) },
    { label: '底部栏检测', value: snapshot.value.footer_detected === false ? '未检测' : '已检测' },
    { label: '分析宽度', value: readNumber(snapshot.value.analysis_width_px, ' px') },
    { label: '分析高度', value: readNumber(snapshot.value.analysis_height_px, ' px') },
  ]
})

</script>

<template>
  <section class="glass-card calibration-banner">
    <details class="banner-details" :open="!snapshotRows.length">
      <summary>查看采集与 OCR 详情</summary>
      <div class="banner-body">
        <div v-if="snapshotRows.length" class="banner-grid">
          <div v-for="row in snapshotRows" :key="row.label" class="banner-metric">
            <span>{{ row.label }}</span>
            <strong>{{ row.value }}</strong>
          </div>
        </div>
        <p v-else class="banner-empty-text">当前结果未解析到采集或 OCR 信息。</p>
      </div>
    </details>
  </section>
</template>

<style scoped>
.calibration-banner {
  padding: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.58);
  display: grid;
  gap: 0;
  box-shadow: 0 10px 24px rgba(44, 32, 20, 0.05);
}

.banner-details {
  padding: 14px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.84), rgba(247, 242, 235, 0.78));
  border: 1px solid rgba(31, 40, 48, 0.06);
  transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}

.banner-details[open] {
  border-color: rgba(23, 96, 135, 0.12);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.78);
}

.banner-details summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  font-weight: 700;
  color: rgba(31, 40, 48, 0.84);
  list-style: none;
  line-height: 1.4;
}

.banner-details summary::after {
  content: '+';
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 18px;
  font-weight: 500;
  flex: 0 0 auto;
}

.banner-details[open] summary::after {
  content: '-';
}

.banner-body {
  display: grid;
  gap: 12px;
  padding-top: 14px;
  margin-top: 14px;
  border-top: 1px solid rgba(31, 40, 48, 0.06);
}

.banner-details summary::-webkit-details-marker {
  display: none;
}

.banner-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
  gap: 10px;
}

.banner-metric {
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.banner-metric span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 6px;
}

.banner-metric strong {
  font-size: 16px;
}

.banner-empty-text {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

@media (max-width: 900px) {
  .banner-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .banner-grid {
    grid-template-columns: 1fr;
  }
}
</style>
