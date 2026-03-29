<script setup lang="ts">
import { ElMessage, ElMessageBox } from 'element-plus'
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'

import { api } from '../api'
import type { Project, RunRecord } from '../types'

const router = useRouter()
const projects = ref<Project[]>([])
const runs = ref<RunRecord[]>([])
const selectedProjectId = ref<number | null>(null)
const creating = ref(false)
const deleting = ref(false)

const form = reactive({
  name: '',
  description: '',
})

const loadData = async () => {
  const [projectRes, runRes] = await Promise.all([api.get<Project[]>('/projects'), api.get<RunRecord[]>('/runs')])
  projects.value = projectRes.data
  runs.value = runRes.data
}

const selectFallbackProject = () => {
  if (!projects.value.length) {
    selectedProjectId.value = null
    return
  }
  if (!projects.value.some((project) => project.id === selectedProjectId.value)) {
    selectedProjectId.value = projects.value[0].id
  }
}

const projectCards = computed(() =>
  projects.value.map((project) => {
    const projectRuns = runs.value.filter((run) => run.project_id === project.id)
    const completed = projectRuns.filter((run) => run.status === 'completed').length
    const latestRun = projectRuns[0] ?? null
    return {
      ...project,
      runCount: projectRuns.length,
      completed,
      latestRun,
    }
  }),
)

const totalTaskCount = computed(() => projectCards.value.reduce((sum, project) => sum + project.runCount, 0))
const totalCompletedCount = computed(() => projectCards.value.reduce((sum, project) => sum + project.completed, 0))
const overviewCards = computed(() => [
  { label: '项目总数', value: projects.value.length, note: '当前已建立的实验项目' },
  { label: '任务总数', value: totalTaskCount.value, note: '项目下累计运行任务' },
  { label: '已完成', value: totalCompletedCount.value, note: '已成功完成的任务数' },
])

const selectedProject = computed(
  () => projectCards.value.find((project) => project.id === selectedProjectId.value) ?? null,
)

const deleteProject = async () => {
  if (!selectedProject.value) {
    return
  }

  try {
    await ElMessageBox.confirm(
      `删除后将同时移除“${selectedProject.value.name}”下的任务记录、统计结果和导出文件，此操作不可恢复，是否继续？`,
      '删除项目提醒',
      {
        confirmButtonText: '确认删除',
        cancelButtonText: '取消',
        type: 'warning',
        confirmButtonClass: 'el-button--danger',
      },
    )

    deleting.value = true
    await api.delete(`/projects/${selectedProject.value.id}`)
    await loadData()
    selectFallbackProject()
    ElMessage.success('项目已删除')
  } catch (error: any) {
    if (error === 'cancel' || error === 'close') {
      return
    }
    ElMessage.error(error?.response?.data?.detail ?? error?.message ?? '项目删除失败')
  } finally {
    deleting.value = false
  }
}

const createProject = async () => {
  if (!form.name.trim()) {
    ElMessage.warning('请先填写项目名称')
    return
  }
  creating.value = true
  try {
    const response = await api.post<Project>('/projects', {
      name: form.name.trim(),
      description: form.description.trim() || null,
    })
    form.name = ''
    form.description = ''
    await loadData()
    selectedProjectId.value = response.data.id
    ElMessage.success('项目已创建')
  } catch (error: any) {
    ElMessage.error(error?.response?.data?.detail ?? error?.message ?? '项目创建失败')
  } finally {
    creating.value = false
  }
}

watch(projects, selectFallbackProject)

onMounted(() => {
  loadData()
    .then(selectFallbackProject)
    .catch(() => ElMessage.error('项目数据加载失败'))
})
</script>

<template>
  <div class="project-layout">
    <section class="glass-card project-create-card">
      <div class="section-head">
        <div>
          <h2 class="section-title">新建项目</h2>
          <p class="section-subtitle">这里是建立项目本身，不是创建任务；项目建好后再进入任务创建。</p>
        </div>
      </div>

      <el-form label-position="top" class="project-create-form">
        <div class="project-create-grid">
          <el-form-item label="项目名称">
            <el-input v-model="form.name" placeholder="例如：镍基单晶 γ′ 相统计 - 退火批次 A" />
          </el-form-item>
          <el-form-item label="项目说明" class="create-span-2">
            <el-input
              v-model="form.description"
              type="textarea"
              :rows="4"
              resize="none"
              placeholder="可填写样品来源、处理条件、实验目的或答辩说明。"
            />
          </el-form-item>
        </div>
        <el-button type="primary" size="large" :loading="creating" @click="createProject">创建项目</el-button>
      </el-form>
    </section>

    <div class="project-top-row">
      <section class="glass-card project-overview-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">项目总览列表</h2>
            <p class="section-subtitle">左侧查看项目总览和项目列表，右侧同步查看当前项目。</p>
          </div>
          <el-button @click="loadData">刷新</el-button>
        </div>

        <div class="overview-grid">
          <article v-for="card in overviewCards" :key="card.label" class="overview-item">
            <span>{{ card.label }}</span>
            <strong>{{ card.value }}</strong>
            <p>{{ card.note }}</p>
          </article>
        </div>

        <div class="project-list-section">
          <div class="section-head list-inner-head">
            <div>
              <h3 class="section-title">项目列表</h3>
              <p class="section-subtitle">点击列表项后，右侧同步显示项目详情和快捷操作。</p>
            </div>
          </div>

          <div v-if="projectCards.length" class="project-list">
            <button
              v-for="project in projectCards"
              :key="project.id"
              type="button"
              class="project-item"
              :class="{ 'is-active': project.id === selectedProjectId }"
              @click="selectedProjectId = project.id"
            >
              <div class="project-item-head">
                <strong>{{ project.name }}</strong>
                <span class="status-chip">{{ project.runCount }} 个任务</span>
              </div>
              <p>{{ project.description || '暂无项目说明，可用于区分不同样品或实验阶段。' }}</p>
              <div class="project-item-meta">
                <span>完成 {{ project.completed }}</span>
                <span>{{ project.latestRun ? `最近任务：${project.latestRun.name}` : '暂无任务' }}</span>
              </div>
            </button>
          </div>

          <el-empty v-else description="还没有项目，先在下方创建一个项目。" />
        </div>
      </section>

      <aside class="glass-card project-detail-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">当前项目</h2>
            <p class="section-subtitle">先选项目，再从这里进入该项目下的任务创建流程。</p>
          </div>
        </div>

        <template v-if="selectedProject">
          <div class="project-focus">
            <strong class="focus-title">{{ selectedProject.name }}</strong>
            <p>{{ selectedProject.description || '当前项目还没有补充说明。' }}</p>
          </div>

          <div class="focus-stats">
            <div>
              <span>任务总数</span>
              <strong>{{ selectedProject.runCount }}</strong>
            </div>
            <div>
              <span>完成任务</span>
              <strong>{{ selectedProject.completed }}</strong>
            </div>
            <div class="focus-span-2">
              <span>最近任务</span>
              <strong>{{ selectedProject.latestRun?.name ?? '暂无' }}</strong>
            </div>
          </div>

          <div class="focus-actions">
            <div class="focus-action-unit focus-action-unit-primary">
              <el-button
                class="focus-action-primary"
                type="primary"
                size="large"
                @click="router.push({ path: '/tasks/create', query: { project_id: selectedProject.id } })"
              >
                进入该项目创建任务
              </el-button>
            </div>

            <div class="focus-action-unit focus-action-unit-danger">
              <el-button
                class="focus-action-danger"
                plain
                type="danger"
                size="large"
                :loading="deleting"
                @click="deleteProject"
              >
                删除当前项目
              </el-button>
            </div>
          </div>
        </template>

        <el-empty v-else description="从左侧列表选择一个项目后在这里查看详情" />
      </aside>
    </div>
  </div>
</template>

<style scoped>
.project-layout {
  display: grid;
  gap: 18px;
}

.project-top-row {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(360px, 0.72fr);
  gap: 18px;
  align-items: stretch;
}

.project-overview-card,
.project-detail-card,
.project-create-card {
  padding: 22px;
}

.project-overview-card,
.project-detail-card {
  height: 100%;
}

.section-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 14px;
  margin-bottom: 18px;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}

.overview-item {
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.58);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.overview-item span {
  display: block;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 12px;
}

.overview-item strong {
  display: block;
  margin-bottom: 8px;
  font-size: 30px;
  line-height: 1;
}

.overview-item p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.project-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 420px;
  overflow: auto;
  padding-right: 4px;
}

.project-list-section {
  padding-top: 18px;
  border-top: 1px solid rgba(31, 40, 48, 0.08);
}

.list-inner-head {
  margin-bottom: 14px;
}

.project-item {
  width: 100%;
  padding: 16px;
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.58);
  text-align: left;
  cursor: pointer;
  transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
}

.project-item:hover {
  transform: translateY(-1px);
  border-color: rgba(23, 96, 135, 0.16);
  box-shadow: 0 12px 24px rgba(23, 96, 135, 0.08);
}

.project-item.is-active {
  border-color: rgba(23, 96, 135, 0.22);
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.08), rgba(184, 90, 43, 0.08));
}

.project-item-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.project-item-head strong {
  font-size: 16px;
}

.project-item p {
  margin: 0 0 12px;
  color: var(--muted);
  line-height: 1.6;
}

.project-item-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  color: var(--muted);
  font-size: 12px;
}

.project-focus {
  padding: 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.58);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.focus-title {
  display: block;
  margin-bottom: 10px;
  font-size: 20px;
}

.project-focus p {
  margin: 0;
  color: var(--muted);
  line-height: 1.7;
}

.focus-stats {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.focus-span-2 {
  grid-column: 1 / -1;
}

.focus-stats div {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.52);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.focus-stats span {
  display: block;
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 12px;
}

.focus-stats strong {
  font-size: 22px;
}

.focus-actions {
  display: grid;
  gap: 12px;
  margin-top: auto;
  padding-top: 20px;
  border-top: 1px solid rgba(31, 40, 48, 0.08);
}

.focus-action-unit {
  padding: 12px;
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.54);
}

.focus-action-unit-primary {
  background: linear-gradient(180deg, rgba(76, 154, 240, 0.08), rgba(255, 255, 255, 0.64));
  border-color: rgba(62, 134, 220, 0.14);
}

.focus-action-unit-danger {
  background: linear-gradient(180deg, rgba(255, 94, 94, 0.05), rgba(255, 255, 255, 0.62));
  border-color: rgba(239, 93, 93, 0.14);
}

.focus-actions :deep(.el-button) {
  width: 100%;
  min-height: 48px;
  margin: 0;
  border-radius: 14px;
  font-weight: 700;
  font-size: 15px;
  letter-spacing: 0.01em;
  transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease, border-color 0.18s ease,
    color 0.18s ease;
}

.focus-actions :deep(.el-button:hover) {
  transform: translateY(-1px);
}

.focus-action-primary:deep(.el-button),
.focus-actions :deep(.focus-action-primary.el-button) {
  border-color: transparent;
  background: linear-gradient(135deg, #4c9af0, #3e86dc);
  box-shadow: 0 12px 24px rgba(62, 134, 220, 0.18);
}

.focus-actions :deep(.focus-action-primary.el-button:hover) {
  box-shadow: 0 16px 28px rgba(62, 134, 220, 0.24);
}

.focus-actions :deep(.focus-action-primary.el-button:active) {
  transform: translateY(0);
  box-shadow: 0 8px 16px rgba(62, 134, 220, 0.18);
}

.focus-actions :deep(.focus-action-danger.el-button) {
  background: rgba(255, 94, 94, 0.06);
  border-color: rgba(255, 94, 94, 0.28);
  color: #ef5d5d;
}

.focus-actions :deep(.focus-action-danger.el-button:hover) {
  background: rgba(255, 94, 94, 0.1);
  border-color: rgba(255, 94, 94, 0.38);
  color: #e24949;
  box-shadow: 0 10px 22px rgba(239, 93, 93, 0.08);
}

.focus-actions :deep(.focus-action-danger.el-button:active) {
  transform: translateY(0);
  box-shadow: none;
}

.project-create-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.project-create-grid {
  display: grid;
  grid-template-columns: minmax(360px, 0.9fr) minmax(0, 1.1fr);
  gap: 16px;
}

.create-span-2 {
  grid-column: 2;
}
</style>
