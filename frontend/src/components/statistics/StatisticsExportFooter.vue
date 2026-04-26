<script setup lang="ts">
defineProps<{
  exportCsvDisabled?: boolean
  exportChartDisabled?: boolean
}>()

const emit = defineEmits<{
  exportCsv: []
  exportChart: []
  openFormula: []
}>()
</script>

<template>
  <section class="glass-card export-footer">
    <header class="export-footer__head">
      <div class="export-footer__copy">
        <h3 class="section-title">导出与公式</h3>
        <p class="section-subtitle">导出统计结果、当前图表，并查看统计口径说明。</p>
      </div>
      <div class="export-footer__status">
        <span class="export-footer__status-label">输出</span>
        <strong>统计文件与图表</strong>
      </div>
    </header>

    <div class="export-footer__actions">
      <el-button class="export-footer__button" plain :disabled="exportCsvDisabled" @click="emit('exportCsv')">导出 CSV</el-button>
      <el-button class="export-footer__button" plain :disabled="exportChartDisabled" @click="emit('exportChart')">导出图表</el-button>
      <el-button class="export-footer__button export-footer__button--primary" @click="emit('openFormula')">统计公式说明</el-button>
    </div>
  </section>
</template>

<style scoped>
.export-footer {
  --panel-radius: var(--stats-radius-lg, var(--radius-card, 20px));
  --panel-radius-sm: 12px;
  --panel-border: var(--stats-border, var(--border, rgba(31, 40, 48, 0.1)));
  --panel-shadow: var(--stats-shadow, 0 14px 28px rgba(15, 23, 42, 0.06));
  --panel-accent: var(--stats-accent, var(--accent, #176087));
  display: grid;
  gap: 12px;
  padding: 16px;
  border-radius: var(--panel-radius);
  border: 1px solid var(--panel-border);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(247, 250, 252, 0.95));
  box-shadow: var(--panel-shadow);
}

.export-footer__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.export-footer__copy {
  display: grid;
  gap: 3px;
}

.export-footer__status {
  display: grid;
  gap: 3px;
  min-width: 136px;
  padding: 10px 12px;
  border-radius: var(--panel-radius-sm);
  border: 1px solid rgba(23, 96, 135, 0.12);
  background: rgba(23, 96, 135, 0.05);
  text-align: right;
}

.export-footer__status-label {
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.export-footer__status strong {
  color: var(--ink);
  font-size: 13px;
  line-height: 1.35;
}

.export-footer__actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.export-footer__actions :deep(.el-button.export-footer__button) {
  min-height: 38px;
  margin-left: 0;
  padding-inline: 14px;
  border-radius: var(--panel-radius-sm);
  border-color: rgba(31, 40, 48, 0.1);
  background: rgba(255, 255, 255, 0.82);
  color: var(--ink);
  box-shadow: none;
  font-weight: 700;
}

.export-footer__actions :deep(.el-button.export-footer__button:hover),
.export-footer__actions :deep(.el-button.export-footer__button:focus-visible) {
  border-color: rgba(23, 96, 135, 0.16);
  background: rgba(255, 255, 255, 0.94);
  color: var(--ink);
}

.export-footer__actions :deep(.el-button.export-footer__button.is-disabled) {
  opacity: 0.45;
  background: rgba(255, 255, 255, 0.72);
  color: var(--muted);
}

.export-footer__actions :deep(.el-button.export-footer__button--primary) {
  border-color: transparent;
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.96), rgba(11, 45, 92, 0.96));
  color: #ffffff;
  box-shadow: 0 10px 18px rgba(23, 96, 135, 0.14);
}

.export-footer__actions :deep(.el-button.export-footer__button--primary:hover),
.export-footer__actions :deep(.el-button.export-footer__button--primary:focus-visible) {
  border-color: transparent;
  background: linear-gradient(135deg, rgba(23, 96, 135, 1), rgba(11, 45, 92, 1));
  color: #ffffff;
}

@media (max-width: 900px) {
  .export-footer__head {
    flex-direction: column;
    align-items: stretch;
  }

  .export-footer__status {
    min-width: 0;
    text-align: left;
  }

  .export-footer__actions {
    justify-content: flex-start;
  }
}
</style>
