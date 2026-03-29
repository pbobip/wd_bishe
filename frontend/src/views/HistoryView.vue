<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { api } from '../api'
import type { RunRecord } from '../types'

const router = useRouter()
const runs = ref<RunRecord[]>([])

const loadRuns = async () => {
  const response = await api.get<RunRecord[]>('/runs')
  runs.value = response.data
}

const completedCount = computed(() => runs.value.filter((run) => run.status === 'completed').length)
const runningCount = computed(() => runs.value.filter((run) => ['pending', 'running'].includes(run.status)).length)
const failedCount = computed(() => runs.value.filter((run) => run.status === 'failed').length)

onMounted(() => {
  loadRuns().catch(() => ElMessage.error('历史任务加载失败'))
})
</script>

<template>
  <div class="history-layout">
    <section class="glass-card history-summary">
      <article class="summary-card">
        <span>全部任务</span>
        <strong>{{ runs.length }}</strong>
      </article>
      <article class="summary-card">
        <span>已完成</span>
        <strong>{{ completedCount }}</strong>
      </article>
      <article class="summary-card">
        <span>运行中</span>
        <strong>{{ runningCount }}</strong>
      </article>
      <article class="summary-card">
        <span>失败</span>
        <strong>{{ failedCount }}</strong>
      </article>
    </section>

    <section class="glass-card history-panel">
      <div class="history-header">
        <div>
          <h2 class="section-title">历史记录</h2>
          <p class="section-subtitle">查看每次任务的状态、模式、进度与结果入口</p>
        </div>
        <el-button @click="loadRuns">刷新</el-button>
      </div>

      <el-table :data="runs" stripe>
        <el-table-column prop="name" label="任务名称" min-width="180" />
        <el-table-column prop="input_mode" label="输入方式" width="110" />
        <el-table-column prop="segmentation_mode" label="模式" width="120" />
        <el-table-column prop="status" label="状态" width="120" />
        <el-table-column label="进度" width="180">
          <template #default="{ row }">
            <el-progress :percentage="Math.round(row.progress * 100)" />
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" min-width="180" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button type="primary" link @click="router.push(`/runs/${row.id}`)">查看</el-button>
          </template>
        </el-table-column>
      </el-table>
    </section>
  </div>
</template>

<style scoped>
.history-layout {
  display: grid;
  gap: 18px;
}

.history-summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  padding: 20px;
}

.summary-card {
  padding: 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.summary-card span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 10px;
}

.summary-card strong {
  font-size: 28px;
}

.history-panel {
  padding: 20px;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
</style>
