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

const resultRuns = computed(() =>
  runs.value.filter((run) => ['completed', 'partial_success', 'failed'].includes(run.status)),
)

onMounted(() => {
  loadRuns().catch(() => ElMessage.error('结果任务加载失败'))
})
</script>

<template>
  <div class="results-layout">
    <section class="glass-card results-list">
      <div class="list-header">
        <div>
          <h2 class="section-title">结果任务列表</h2>
          <p class="section-subtitle">点击任务直接进入结果详情。</p>
        </div>
      </div>

      <el-empty v-if="!resultRuns.length" description="还没有可查看的结果任务" />

      <div v-else class="result-run-list">
        <button
          v-for="run in resultRuns"
          :key="run.id"
          type="button"
          class="result-run-item"
          @click="router.push(`/runs/${run.id}`)"
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
</template>

<style scoped>
.results-layout {
  display: grid;
}

.results-list {
  padding: 18px;
}

.overview-header,
.list-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 18px;
}

.list-header > div {
  flex: 1 1 240px;
  min-width: 0;
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
</style>
