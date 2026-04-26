<script setup lang="ts">
defineProps<{
  rows: Array<{ label: string; value: string }>
}>()
</script>

<template>
  <section class="glass-card description-panel">
    <header class="description-panel__head">
      <h3 class="section-title">描述统计</h3>
      <p class="section-subtitle">口径：当前指标全部颗粒测量值的核心统计量。</p>
    </header>

    <div class="description-table" role="table" aria-label="描述统计">
      <div class="description-table__head" role="rowgroup">
        <div class="description-table__head-row" role="row">
          <span class="description-table__head-label" role="columnheader">统计量</span>
          <span class="description-table__head-value" role="columnheader">数值</span>
        </div>
      </div>
      <div v-for="row in rows" :key="row.label" class="description-row" role="row">
        <span class="description-label" role="cell">{{ row.label }}</span>
        <strong class="description-value" role="cell">{{ row.value }}</strong>
      </div>
    </div>
  </section>
</template>

<style scoped>
.description-panel {
  --panel-radius: var(--stats-radius-lg, var(--radius-card, 20px));
  --panel-radius-sm: 14px;
  --panel-border: var(--stats-border, var(--border, rgba(31, 40, 48, 0.1)));
  --panel-shadow: var(--stats-shadow, 0 14px 28px rgba(15, 23, 42, 0.06));
  --panel-deep: var(--stats-deep, var(--accent, #176087));
  display: grid;
  gap: 12px;
  padding: 16px;
  border-radius: var(--panel-radius);
  border: 1px solid var(--panel-border);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(248, 250, 252, 0.95));
  box-shadow: var(--panel-shadow);
}

.description-panel__head {
  display: grid;
  gap: 3px;
}

.section-title {
  margin-bottom: 0;
}

.description-table {
  display: grid;
  border: 1px solid rgba(31, 40, 48, 0.08);
  border-radius: var(--panel-radius-sm);
  overflow: hidden;
  background: rgba(255, 255, 255, 0.9);
}

.description-table__head {
  border-bottom: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(31, 40, 48, 0.04);
}

.description-table__head-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  min-height: 34px;
}

.description-table__head-label,
.description-table__head-value {
  padding: 0 14px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.description-table__head-value {
  min-width: 118px;
  border-left: 1px solid rgba(31, 40, 48, 0.08);
  text-align: right;
}

.description-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  min-height: 46px;
  border-bottom: 1px solid rgba(31, 40, 48, 0.06);
}

.description-row:last-child {
  border-bottom: 0;
}

.description-label,
.description-value {
  padding: 0 14px;
}

.description-label {
  color: var(--ink);
  font-size: 14px;
  line-height: 1.45;
}

.description-value {
  min-width: 118px;
  border-left: 1px solid rgba(31, 40, 48, 0.06);
  color: var(--panel-deep);
  font-size: 15px;
  font-weight: 700;
  line-height: 1.2;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
</style>
