<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '../api'
import CalibrationStatusBanner from '../components/CalibrationStatusBanner.vue'
import PreviewWorkspace from '../components/PreviewWorkspace.vue'
import StatsSidebar from '../components/StatsSidebar.vue'
import type { RunResultsPayload } from '../types'

const route = useRoute()
const router = useRouter()
const runId = computed(() => Number(route.params.id))
const payload = ref<RunResultsPayload | null>(null)
let timer: number | null = null

const loadPayload = async () => {
  const response = await api.get<RunResultsPayload>(`/runs/${runId.value}/results`)
  payload.value = response.data
}

const startPolling = () => {
  timer = window.setInterval(async () => {
    await loadPayload()
    const status = payload.value?.run.status
    if (status && ['completed', 'failed', 'partial_success'].includes(status)) {
      if (timer) window.clearInterval(timer)
      timer = null
    }
  }, 2500)
}

const progressPercent = computed(() => Math.round((payload.value?.run.progress ?? 0) * 100))

onMounted(async () => {
  try {
    await loadPayload()
    if (!payload.value || !['completed', 'failed', 'partial_success'].includes(payload.value.run.status)) {
      startPolling()
    }
  } catch {
    ElMessage.error('任务详情加载失败')
  }
})

onBeforeUnmount(() => {
  if (timer) window.clearInterval(timer)
})
</script>

<template>
  <div v-if="payload" class="detail-page">
    <section class="glass-card detail-toolbar">
      <div>
        <h2 class="section-title">结果核查工作台</h2>
        <p class="section-subtitle">先核对图像层与分割结果，再进入统计分析区查看图表和明细。</p>
      </div>

      <div class="detail-toolbar-actions">
        <el-button type="primary" @click="router.push(`/runs/${payload.run.id}/statistics`)">进入统计分析区</el-button>
        <el-button plain @click="router.push('/results')">返回结果主界面</el-button>
      </div>
    </section>

    <section class="detail-banner">
      <CalibrationStatusBanner :run="payload.run" />
    </section>

    <div class="detail-workbench">
      <main class="detail-main">
        <PreviewWorkspace :images="payload.images" :mode="payload.run.segmentation_mode" />

        <section class="glass-card step-card">
          <div class="panel-heading">
            <div>
              <h3 class="section-title">执行轨迹</h3>
              <p class="section-subtitle">保留任务步骤和处理进度，方便对当前结果做方法追溯。</p>
            </div>
            <div class="panel-meta">
              <span class="status-chip">{{ payload.run.status }}</span>
              <span class="meta-chip">进度 {{ progressPercent }}%</span>
            </div>
          </div>

          <div class="step-timeline-shell">
            <el-timeline>
              <el-timeline-item
                v-for="step in payload.steps"
                :key="step.id"
                :type="step.status === 'completed' ? 'success' : step.status === 'failed' ? 'danger' : 'primary'"
                :hollow="step.status === 'pending'"
              >
                <strong>{{ step.step_key }}</strong>
                <p>{{ step.message || '等待执行' }}</p>
              </el-timeline-item>
            </el-timeline>
          </div>
        </section>
      </main>

      <aside class="detail-side">
        <section class="glass-card overview-card">
          <div class="overview-head">
            <div>
              <span class="panel-eyebrow">任务</span>
              <h3 class="section-title">{{ payload.run.name }}</h3>
            </div>
            <span class="status-chip">{{ payload.run.status }}</span>
          </div>

          <el-progress :percentage="progressPercent" />

          <div class="overview-grid">
            <article class="overview-item">
              <span>输入方式</span>
              <strong>{{ payload.run.input_mode }}</strong>
            </article>
            <article class="overview-item">
              <span>分割模式</span>
              <strong>{{ payload.run.segmentation_mode }}</strong>
            </article>
            <article class="overview-item">
              <span>图像数</span>
              <strong>{{ payload.images.length }}</strong>
            </article>
            <article class="overview-item">
              <span>创建时间</span>
              <strong>{{ payload.run.created_at }}</strong>
            </article>
          </div>

          <div v-if="payload.run.error_message" class="error-block">
            <span>错误信息</span>
            <p>{{ payload.run.error_message }}</p>
          </div>
        </section>

        <StatsSidebar :run="payload.run" :summary="payload.run.summary" :exports="payload.exports" />
      </aside>
    </div>
  </div>
</template>

<style scoped>
.detail-page {
  display: grid;
  gap: 16px;
  min-width: 0;
}

.detail-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  padding: 18px 20px;
}

.detail-toolbar-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.detail-banner {
  min-width: 0;
}

.detail-workbench {
  display: grid;
  grid-template-columns: minmax(0, 1.56fr) minmax(300px, 360px);
  gap: 16px;
  min-width: 0;
  align-items: start;
}

.detail-main,
.detail-side {
  display: grid;
  gap: 16px;
  min-width: 0;
}

.step-card,
.overview-card {
  padding: 18px;
}

.panel-heading,
.overview-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 14px;
  margin-bottom: 14px;
}

.panel-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.meta-chip {
  display: inline-flex;
  align-items: center;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}

.panel-eyebrow {
  display: inline-flex;
  align-items: center;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 11px;
  font-weight: 700;
}

.step-timeline-shell {
  max-height: 380px;
  overflow: auto;
  padding-right: 4px;
}

.overview-card :deep(.el-progress) {
  margin-bottom: 16px;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 16px;
}

.overview-item {
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.64);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.overview-item span {
  display: block;
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 12px;
}

.overview-item strong {
  display: block;
  line-height: 1.55;
  overflow-wrap: anywhere;
}

.error-block {
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(184, 90, 43, 0.08);
}

.error-block span {
  display: block;
  margin-bottom: 6px;
  color: #8d4c27;
  font-size: 12px;
  font-weight: 700;
}

.error-block p {
  margin: 0;
  color: #8d4c27;
  line-height: 1.6;
}

@media (max-width: 1280px) {
  .detail-workbench {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 920px) {
  .detail-toolbar,
  .panel-heading,
  .overview-head {
    flex-direction: column;
    align-items: stretch;
  }

  .detail-toolbar-actions {
    justify-content: flex-start;
  }
}

@media (max-width: 640px) {
  .overview-grid {
    grid-template-columns: 1fr;
  }
}
</style>
