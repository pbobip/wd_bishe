<script setup lang="ts">
import { computed } from 'vue'

import type { RunRecord } from '../types'
import { getCalibrationHint, getCalibrationState } from '../utils/calibration'

const props = defineProps<{
  run: RunRecord | null | undefined
}>()

const state = computed(() => getCalibrationState(props.run))

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

const hint = computed(() => getCalibrationHint(props.run))

const summaryLine = computed(() => {
  if (!snapshot.value) return ''
  const segments = [
    snapshot.value.ocr_scale_bar_um ? `比例尺 ${snapshot.value.ocr_scale_bar_um} μm` : null,
    snapshot.value.ocr_fov_um ? `FoV ${snapshot.value.ocr_fov_um} μm` : null,
    snapshot.value.ocr_magnification_text ? `Mag ${snapshot.value.ocr_magnification_text}` : null,
    snapshot.value.ocr_wd_mm ? `WD ${snapshot.value.ocr_wd_mm} mm` : null,
  ].filter((item): item is string => Boolean(item))
  return segments.length ? `OCR 标定结果：${segments.join(' · ')}` : ''
})
</script>

<template>
  <section class="glass-card calibration-banner" :class="{ 'is-uncalibrated': !state.calibrated }">
    <div class="banner-header">
      <div>
        <span class="status-chip">{{ state.label }}</span>
        <h3 class="section-title">标定状态</h3>
      </div>
      <p class="banner-detail">{{ state.detail }}</p>
    </div>

    <div v-if="summaryLine" class="banner-summary">
      <strong>OCR 标定信息</strong>
      <span>{{ summaryLine }}</span>
    </div>

    <p v-if="hint" class="banner-hint">{{ hint }}</p>

    <details v-if="snapshotRows.length" class="banner-details">
      <summary>查看采集与 OCR 详情</summary>
      <div class="banner-grid">
        <div v-for="row in snapshotRows" :key="row.label" class="banner-metric">
          <span>{{ row.label }}</span>
          <strong>{{ row.value }}</strong>
        </div>
      </div>
    </details>
  </section>
</template>

<style scoped>
.calibration-banner {
  padding: 16px 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.08), rgba(184, 90, 43, 0.08));
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.calibration-banner.is-uncalibrated {
  border-color: rgba(184, 90, 43, 0.2);
}

.banner-header {
  display: grid;
  gap: 8px;
  align-items: start;
}

.banner-detail,
.banner-hint {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.banner-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 12px;
  align-items: center;
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.banner-summary strong {
  font-size: 14px;
}

.banner-summary span {
  color: var(--muted);
  line-height: 1.6;
}

.banner-hint {
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.58);
}

.banner-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.banner-metric {
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.62);
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

.banner-details {
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.56);
}

.banner-details summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--muted);
  list-style: none;
}

.banner-details summary::-webkit-details-marker {
  display: none;
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
