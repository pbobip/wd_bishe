<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { api } from '../api'
import type { RunRecord } from '../types'

const router = useRouter()
const runs = ref<RunRecord[]>([])
const selectedRunId = ref<number | null>(null)

const loadRuns = async () => {
  const response = await api.get<RunRecord[]>('/runs')
  runs.value = response.data
  if (!resultRuns.value.length) {
    selectedRunId.value = null
    return
  }
  if (!resultRuns.value.some((run) => run.id === selectedRunId.value)) {
    selectedRunId.value = resultRuns.value[0].id
  }
}

const resultRuns = computed(() =>
  runs.value.filter((run) => ['completed', 'partial_success', 'failed'].includes(run.status)),
)

const latestRun = computed(() => resultRuns.value[0] ?? null)
const selectedRun = computed(() => resultRuns.value.find((run) => run.id === selectedRunId.value) ?? latestRun.value)

const runMetricCards = computed(() => {
  if (!selectedRun.value) return []
  return [
    {
      label: '状态',
      value: selectedRun.value.status,
    },
    {
      label: '模式',
      value: selectedRun.value.segmentation_mode,
    },
    {
      label: '输入方式',
      value: selectedRun.value.input_mode,
    },
    {
      label: 'Vf',
      value: selectedRun.value.summary?.vf ?? selectedRun.value.summary?.avg_vf ?? '--',
    },
    {
      label: '颗粒数',
      value: selectedRun.value.summary?.particle_count ?? selectedRun.value.summary?.avg_particle_count ?? '--',
    },
    {
      label: '导出包',
      value: selectedRun.value.export_bundle_path ? '已生成' : '未生成',
    },
  ]
})

onMounted(() => {
  loadRuns().catch(() => ElMessage.error('结果任务加载失败'))
})
</script>

<template>
  <div class="results-layout">
    <div class="results-main-column">
      <section class="glass-card results-list">
        <div class="list-header">
          <div>
            <h2 class="section-title">结果任务列表</h2>
            <p class="section-subtitle">左侧选任务，右侧看摘要和结果入口。</p>
          </div>
          <el-button type="primary" plain @click="router.push('/tasks/create')">新建任务</el-button>
        </div>

        <el-empty v-if="!resultRuns.length" description="还没有可查看的结果任务" />

        <div v-else class="result-run-list">
          <button
            v-for="run in resultRuns"
            :key="run.id"
            type="button"
            class="result-run-item"
            :class="{ 'is-active': run.id === selectedRun?.id }"
            @click="selectedRunId = run.id"
          >
            <div class="result-run-head">
              <strong>{{ run.name }}</strong>
              <span class="status-chip">{{ run.status }}</span>
            </div>
            <div class="result-run-meta">
              <span>{{ run.segmentation_mode }}</span>
              <span>{{ run.input_mode }}</span>
              <span>Vf {{ run.summary?.vf ?? run.summary?.avg_vf ?? '--' }}</span>
            </div>
            <div class="result-run-foot">
              <span>{{ run.created_at }}</span>
              <span>{{ run.summary?.particle_count ?? run.summary?.avg_particle_count ?? '--' }} 颗粒</span>
            </div>
          </button>
        </div>
      </section>
    </div>

    <aside class="glass-card results-focus">
      <div class="focus-header">
        <div>
          <h2 class="section-title">当前任务摘要</h2>
          <p class="section-subtitle">这里展示当前选中结果任务的摘要和入口。</p>
        </div>
        <span class="status-chip">{{ selectedRun?.status ?? '暂无结果' }}</span>
      </div>

      <template v-if="selectedRun">
        <div class="focus-actions">
          <el-button type="primary" size="large" @click="router.push(`/runs/${selectedRun.id}`)">
            打开结果详情
          </el-button>
        </div>

        <div class="focus-card">
          <strong class="focus-name">{{ selectedRun.name }}</strong>
          <p>更新时间：{{ selectedRun.updated_at }}</p>
          <p>创建时间：{{ selectedRun.created_at }}</p>
          <p>当前任务已纳入结果中心，可继续查看图像级结果和导出文件。</p>
        </div>

        <div class="focus-summary">
          <div v-for="metric in runMetricCards" :key="metric.label">
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
          </div>
        </div>
      </template>

      <el-empty v-else description="等待首个结果任务生成" />
    </aside>
  </div>
</template>

<style scoped>
.results-layout {
  display: flex;
  gap: 16px;
  align-items: start;
}

.results-main-column {
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.results-list,
.results-focus {
  padding: 18px;
}

.results-focus {
  width: 304px;
  flex: 0 0 304px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  position: sticky;
  top: 0;
}

.overview-header,
.list-header,
.focus-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 18px;
}

.list-header > div,
.focus-header > div {
  flex: 1 1 240px;
  min-width: 0;
}

.list-header .el-button,
.focus-header .status-chip {
  flex: 0 0 auto;
}

.focus-card {
  padding: 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.focus-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.result-run-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.result-run-item {
  width: 100%;
  padding: 16px;
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.58);
  text-align: left;
  cursor: pointer;
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.result-run-item:hover {
  transform: translateY(-1px);
  border-color: rgba(23, 96, 135, 0.16);
  box-shadow: 0 12px 24px rgba(23, 96, 135, 0.08);
}

.result-run-item.is-active {
  border-color: rgba(23, 96, 135, 0.22);
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.08), rgba(184, 90, 43, 0.08));
}

.result-run-head,
.result-run-foot {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.result-run-head {
  margin-bottom: 10px;
}

.result-run-head strong {
  font-size: 16px;
  flex: 1 1 220px;
  min-width: 0;
}

.result-run-meta,
.result-run-foot {
  color: var(--muted);
  font-size: 12px;
}

.result-run-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 10px;
}

.focus-name {
  display: block;
  margin-bottom: 10px;
  font-size: 18px;
}

.focus-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.focus-summary div {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.52);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.focus-summary span {
  display: block;
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 6px;
}

.focus-summary strong {
  font-size: 20px;
}

.focus-actions {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.focus-note {
  padding: 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.52);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.focus-note .metric-label {
  display: block;
  margin-bottom: 10px;
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.focus-note p {
  margin: 10px 0 0;
  color: var(--muted);
  line-height: 1.7;
}

@media (max-width: 1100px) {
  .results-layout {
    flex-direction: column;
  }

  .results-focus {
    width: 100%;
    flex: 1 1 auto;
    position: static;
  }
}
</style>
