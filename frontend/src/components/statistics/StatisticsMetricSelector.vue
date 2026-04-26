<script setup lang="ts">
import type { StatisticMetricKey } from '../../utils/runStatistics'

type DisplayMetricKey = StatisticMetricKey | 'channel_width_x' | 'channel_width_y'

defineProps<{
  modelValue: StatisticMetricKey
  axis: 'x' | 'y'
}>()

const emit = defineEmits<{
  'update:modelValue': [value: StatisticMetricKey]
  'update:axis': [value: 'x' | 'y']
}>()

const metricOptions: Array<{ key: DisplayMetricKey; label: string; icon: 'vf' | 'count' | 'area' | 'size' | 'wx' | 'wy' }> = [
  { key: 'vf', label: 'Vf', icon: 'vf' },
  { key: 'particle_count', label: '颗粒数', icon: 'count' },
  { key: 'area', label: '面积', icon: 'area' },
  { key: 'size', label: '尺寸', icon: 'size' },
  { key: 'channel_width_x', label: '水平 W', icon: 'wx' },
  { key: 'channel_width_y', label: '垂直 W', icon: 'wy' },
]

const selectMetric = (key: DisplayMetricKey) => {
  if (key === 'channel_width_x' || key === 'channel_width_y') {
    emit('update:axis', key === 'channel_width_x' ? 'x' : 'y')
    emit('update:modelValue', 'channel_width')
    return
  }
  emit('update:modelValue', key)
}

const isActive = (current: StatisticMetricKey, axis: 'x' | 'y', key: DisplayMetricKey) => {
  if (key === 'channel_width_x') {
    return current === 'channel_width' && axis === 'x'
  }
  if (key === 'channel_width_y') {
    return current === 'channel_width' && axis === 'y'
  }
  return current === key
}
</script>

<template>
  <section class="glass-card metric-panel">
    <header class="metric-panel__head">
      <h3 class="section-title">指标选择</h3>
      <p class="section-subtitle">切换统计口径后，主图和描述统计同步更新。</p>
    </header>

    <div class="metric-list" role="listbox" aria-label="统计指标选择">
      <button
        v-for="option in metricOptions"
        :key="option.key"
        type="button"
        class="metric-item"
        :class="{ 'is-active': isActive(modelValue, axis, option.key) }"
        :aria-selected="isActive(modelValue, axis, option.key)"
        @click="selectMetric(option.key)"
      >
        <span class="metric-item__icon" aria-hidden="true">
          <svg v-if="option.icon === 'vf'" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="8.5" />
            <path d="M12 4v8h6" />
          </svg>
          <svg v-else-if="option.icon === 'count'" viewBox="0 0 24 24" fill="none">
            <circle cx="7" cy="7" r="2.5" />
            <circle cx="17" cy="7" r="2.5" />
            <circle cx="7" cy="17" r="2.5" />
            <circle cx="17" cy="17" r="2.5" />
          </svg>
          <svg v-else-if="option.icon === 'area'" viewBox="0 0 24 24" fill="none">
            <rect x="4.5" y="4.5" width="15" height="15" rx="2" />
            <path d="M4.5 10h15M10 4.5v15" />
          </svg>
          <svg v-else-if="option.icon === 'size'" viewBox="0 0 24 24" fill="none">
            <path d="M5 19L19 5" />
            <path d="M7 5h12v12" />
          </svg>
          <svg v-else-if="option.icon === 'wx'" viewBox="0 0 24 24" fill="none">
            <path d="M4 12h16" />
            <path d="M7 9l-3 3 3 3" />
            <path d="M17 9l3 3-3 3" />
          </svg>
          <svg v-else viewBox="0 0 24 24" fill="none">
            <path d="M12 4v16" />
            <path d="M9 7l3-3 3 3" />
            <path d="M9 17l3 3 3-3" />
          </svg>
        </span>
        <span class="metric-item__label">{{ option.label }}</span>
      </button>
    </div>

    <div class="metric-notes" aria-label="指标分组说明">
      <article class="metric-note">
        <span class="metric-note__kicker">按图片汇总</span>
        <strong>Vf、颗粒数</strong>
      </article>
      <article class="metric-note">
        <span class="metric-note__kicker">按颗粒测量</span>
        <strong>面积、尺寸</strong>
      </article>
      <article class="metric-note">
        <span class="metric-note__kicker">按通道测量</span>
        <strong>水平 W、垂直 W</strong>
      </article>
    </div>
  </section>
</template>

<style scoped>
.metric-panel {
  --panel-radius: var(--stats-radius-lg, var(--radius-card, 20px));
  --panel-radius-sm: 14px;
  --panel-radius-xs: 12px;
  --panel-border: var(--stats-border, var(--border, rgba(31, 40, 48, 0.1)));
  --panel-border-strong: var(--stats-border-strong, rgba(23, 96, 135, 0.16));
  --panel-shadow: var(--stats-shadow, 0 14px 28px rgba(15, 23, 42, 0.06));
  --panel-surface: var(--surface, rgba(255, 249, 240, 0.92));
  --panel-surface-strong: var(--surface-strong, #fffdf8);
  --panel-muted: rgba(31, 40, 48, 0.04);
  --panel-accent: var(--stats-accent, var(--accent, #176087));
  --panel-deep: var(--stats-deep, #0b2d5c);
  display: grid;
  gap: 12px;
  padding: 16px;
  border-radius: var(--panel-radius);
  border: 1px solid var(--panel-border);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.95));
  box-shadow: var(--panel-shadow);
}

.metric-panel__head {
  display: grid;
  gap: 3px;
}

.section-title {
  margin-bottom: 0;
}

.metric-list {
  display: grid;
  gap: 8px;
}

.metric-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-height: 44px;
  padding: 0 13px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  border-radius: var(--panel-radius-sm);
  background: rgba(255, 255, 255, 0.82);
  color: var(--ink);
  cursor: pointer;
  transition:
    border-color 0.2s ease,
    background 0.2s ease,
    box-shadow 0.2s ease,
    transform 0.18s ease;
}

.metric-item:hover {
  transform: translateY(-1px);
  border-color: var(--panel-border-strong);
  box-shadow: 0 8px 16px rgba(15, 23, 42, 0.05);
}

.metric-item:focus-visible {
  outline: none;
  box-shadow: inset 0 0 0 2px rgba(23, 96, 135, 0.18);
}

.metric-item.is-active {
  border-color: transparent;
  background: linear-gradient(135deg, var(--panel-accent), var(--panel-deep));
  color: #ffffff;
  box-shadow: 0 12px 22px rgba(23, 96, 135, 0.18);
}

.metric-item.is-active:focus-visible {
  box-shadow:
    inset 0 0 0 2px rgba(255, 255, 255, 0.18),
    0 12px 22px rgba(23, 96, 135, 0.18);
}

.metric-item__icon {
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  flex: 0 0 28px;
  border-radius: 10px;
  background: rgba(31, 40, 48, 0.05);
}

.metric-item.is-active .metric-item__icon {
  background: rgba(255, 255, 255, 0.12);
}

.metric-item__icon svg {
  width: 18px;
  height: 18px;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.metric-item__label {
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.01em;
}

.metric-notes {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  padding-top: 2px;
}

.metric-note {
  display: grid;
  gap: 3px;
  padding: 10px 12px;
  border-radius: var(--panel-radius-xs);
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.7);
}

.metric-note__kicker {
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.metric-note strong {
  color: var(--ink);
  font-size: 13px;
  line-height: 1.35;
}

@media (max-width: 960px) {
  .metric-notes {
    grid-template-columns: 1fr;
  }
}
</style>
