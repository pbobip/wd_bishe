<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onBeforeUnmount, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { useTaskWorkflow } from '../composables/useTaskWorkflow'

const router = useRouter()
const route = useRoute()
const workflow = useTaskWorkflow()

const launchSummaryCards = computed(() => [
  {
    label: '已加载图像',
    value: `${workflow.activePayload.value?.images.length ?? workflow.selectedFiles.value.length} 张`,
    note: '图像已在任务创建页完成登记',
  },
  {
    label: '已处理',
    value: workflow.activeStepProgressText.value || '等待执行',
    note: '逐图处理进度',
  },
  {
    label: '执行状态',
    value: workflow.runStatusLabel.value,
    note: workflow.activePayload.value?.run.status ? '后端运行状态' : '等待启动',
  },
  {
    label: '当前任务',
    value: workflow.activeRunId.value ? `#${workflow.activeRunId.value}` : '未创建',
    note: workflow.activePayload.value?.run.name ?? workflow.form.name,
  },
])

const currentImageCard = computed(() => {
  const images = workflow.activePayload.value?.images ?? []
  const currentName = workflow.activeRunningStep.value?.details?.current_image_name
  const matchedImage =
    images.find((image) => image.image_name === currentName) ??
    images[0] ??
    null

  return {
    name: matchedImage?.image_name ?? currentName ?? '等待当前图像',
    url: matchedImage?.input_url ?? null,
    message: workflow.activeRunningStep.value?.details?.message || workflow.launchMessage.value,
    status: workflow.activeRunningStep.value?.status ?? workflow.launchPhase.value,
  }
})

const failureNotice = computed(() => {
  const runError = workflow.activePayload.value?.run.error_message
  if (workflow.launchError.value) return workflow.launchError.value
  if (runError) return runError
  if (workflow.activePayload.value?.run.status === 'failed') return '任务执行失败，请查看步骤状态与错误日志。'
  return ''
})

const canConfigureTraditionalPostprocess = computed(() => workflow.form.segmentation_mode !== 'dl')

const handleLaunch = async () => {
  if (!workflow.canLaunch.value) {
    if (!workflow.workflowReady.value) {
      ElMessage.warning('请先完成任务创建配置')
      return
    }
    if (!workflow.ensurePreprocessReady()) {
      return
    }
    ElMessage.warning('当前任务暂时无法启动')
    return
  }
  if (workflow.calibrationNeedsWarning.value) {
    await workflow.openCalibrationDialog()
    return
  }
  await workflow.launchCurrentTask()
}

const continueWithoutCalibration = async () => {
  workflow.closeCalibrationDialog()
  await workflow.launchCurrentTask()
}

const applyCalibrationAndLaunch = async () => {
  if (workflow.calibrationComputedValue.value === null) {
    ElMessage.warning('请先填写有效的比例尺 μm 数值和像素长度')
    return
  }
  workflow.applyCalibrationValue(workflow.calibrationComputedValue.value)
  workflow.closeCalibrationDialog()
  await workflow.launchCurrentTask()
}

const applySuggestedCalibration = async (value: number) => {
  workflow.applyCalibrationValue(value)
  workflow.closeCalibrationDialog()
  await workflow.launchCurrentTask()
}

onMounted(async () => {
  try {
    await workflow.initialize(Number(route.query.project_id || 0) || null)
    const restored = await workflow.restoreDraftRunFromSession()
    if (!workflow.activeRunId.value && !restored) {
      router.replace('/tasks/create')
      ElMessage.warning('请先在任务创建页完成图像登记，再进入后处理页')
      return
    }
    if (workflow.activeRunId.value) {
      if (!restored) {
        await workflow.loadActivePayload()
      } else {
        ElMessage.success('已恢复草稿任务，可继续在后处理页配置并运行')
      }
      if (workflow.launchPhase.value === 'running') {
        workflow.startPolling()
      }
    }
  } catch {
    ElMessage.error('后处理页初始化失败')
  }
})

onBeforeUnmount(() => {
  workflow.stopPolling()
})
</script>

<template>
  <div class="postprocess-layout">
    <section class="postprocess-main">
      <section class="glass-card postprocess-config-card">
        <div class="section-head">
          <div>
            <h2 class="section-title">后处理</h2>
            <p class="section-subtitle">在这里配置传统后处理、统计导出，并正式启动任务运行。</p>
          </div>
          <div class="head-actions">
            <el-button @click="router.push('/tasks/create')">返回任务创建</el-button>
            <el-button type="primary" size="large" :loading="workflow.actionLoading.value" @click="handleLaunch">
              开始运行
            </el-button>
          </div>
        </div>

        <el-form label-position="top" class="postprocess-form">
          <section class="form-section">
            <div class="section-inline-head">
              <h3>后处理方案</h3>
              <span>这里配置填孔、watershed、平滑与形状过滤，运行页再负责执行与进度。</span>
            </div>

            <div v-if="canConfigureTraditionalPostprocess" class="form-grid two-columns">
              <div class="switch-card">
                <div>
                  <strong>填孔</strong>
                  <p>补全连通域内部空洞，增强颗粒区域完整性。</p>
                </div>
                <el-switch v-model="workflow.form.postprocess.fill_holes" />
              </div>

              <div class="switch-card">
                <div>
                  <strong>Watershed</strong>
                  <p>用于分离相互粘连的颗粒或岛状区域。</p>
                </div>
                <el-switch v-model="workflow.form.postprocess.watershed" />
              </div>

              <div class="switch-card form-span-2">
                <div>
                  <strong>平滑</strong>
                  <p>对边界进行轻量平滑，降低毛刺和锯齿干扰。</p>
                </div>
                <el-switch v-model="workflow.form.postprocess.smoothing.enabled" />
              </div>

              <template v-if="workflow.form.postprocess.smoothing.enabled">
                <el-form-item label="平滑方式">
                  <el-select v-model="workflow.form.postprocess.smoothing.method">
                    <el-option label="均值平滑" value="mean" />
                    <el-option label="高斯平滑" value="gaussian" />
                    <el-option label="中值平滑" value="median" />
                  </el-select>
                </el-form-item>

                <el-form-item label="平滑核大小">
                  <el-input-number v-model="workflow.form.postprocess.smoothing.kernel" :min="1" :step="2" />
                </el-form-item>
              </template>

              <div class="switch-card form-span-2">
                <div>
                  <strong>形状过滤</strong>
                  <p>按面积、圆度、实心度等规则过滤异常连通域。</p>
                </div>
                <el-switch v-model="workflow.form.postprocess.shape_filter.enabled" />
              </div>

              <template v-if="workflow.form.postprocess.shape_filter.enabled">
                <el-form-item label="最小面积">
                  <el-input-number v-model="workflow.form.postprocess.shape_filter.min_area" :min="1" />
                </el-form-item>
                <el-form-item label="最大面积">
                  <el-input-number
                    v-model="workflow.form.postprocess.shape_filter.max_area"
                    :min="1"
                    :step="10"
                    :precision="0"
                    placeholder="可留空"
                  />
                </el-form-item>
                <el-form-item label="最小实心度">
                  <el-input-number v-model="workflow.form.postprocess.shape_filter.min_solidity" :min="0" :max="1" :step="0.05" :precision="2" />
                </el-form-item>
                <el-form-item label="最小圆度">
                  <el-input-number v-model="workflow.form.postprocess.shape_filter.min_circularity" :min="0" :max="1" :step="0.05" :precision="2" />
                </el-form-item>
                <el-form-item label="最小 roundness">
                  <el-input-number v-model="workflow.form.postprocess.shape_filter.min_roundness" :min="0" :max="1" :step="0.05" :precision="2" />
                </el-form-item>
                <el-form-item class="form-span-2" label="最大长宽比">
                  <el-input-number
                    v-model="workflow.form.postprocess.shape_filter.max_aspect_ratio"
                    :min="1"
                    :step="0.1"
                    :precision="2"
                    placeholder="可留空"
                  />
                </el-form-item>
              </template>

              <div class="switch-card form-span-2">
                <div>
                  <strong>触边颗粒剔除</strong>
                  <p>去掉贴边连通域，减少 ROI 边缘误判。</p>
                </div>
                <el-switch v-model="workflow.form.postprocess.remove_border" />
              </div>
            </div>

            <div v-else class="placeholder-card">
              <strong>当前模式不使用传统后处理</strong>
              <p>深度学习模式将直接使用模型输出的掩码进入统计与导出环节。</p>
            </div>
          </section>

          <section class="form-section">
            <div class="section-inline-head">
              <h3>统计与测量项</h3>
              <span>这里控制是否统计，以及本次要输出哪些指标与导出内容。</span>
            </div>

            <div class="form-grid two-columns">
              <div class="switch-card form-span-2">
                <div>
                  <strong>启用统计</strong>
                  <p>{{ workflow.measurementSummary.value }}</p>
                </div>
                <el-switch v-model="workflow.form.stats.enabled" />
              </div>

              <template v-if="workflow.form.stats.enabled">
                <div class="form-span-2 measurement-panel">
                  <div class="measurement-panel-head">
                    <strong>测量项配置</strong>
                    <span>建议至少保留 Vf、颗粒数、面积、尺寸和通道宽度。</span>
                  </div>
                  <div class="measurement-grid">
                    <div v-for="option in workflow.measurementOptions" :key="option.value" class="measurement-item">
                      <div>
                        <strong>{{ option.label }}</strong>
                        <p>当前{{ workflow.form.stats.measurements[option.value] ? '启用' : '关闭' }}</p>
                      </div>
                      <el-switch v-model="workflow.form.stats.measurements[option.value]" />
                    </div>
                  </div>
                  <div class="field-helper">
                    <span class="field-helper-status">当前已选 {{ workflow.measurementCount.value }} 项。</span>
                  </div>
                </div>

                <div class="switch-card">
                  <div>
                    <strong>导出 CSV</strong>
                    <p>输出图像级与颗粒级 CSV 统计表。</p>
                  </div>
                  <el-switch v-model="workflow.form.stats.export_csv" />
                </div>

                <div class="switch-card">
                  <div>
                    <strong>导出 XLSX</strong>
                    <p>输出图像级与颗粒级 Excel 统计表。</p>
                  </div>
                  <el-switch v-model="workflow.form.stats.export_xlsx" />
                </div>

                <div class="switch-card">
                  <div>
                    <strong>导出统计表</strong>
                    <p>保留图像级、颗粒级和批量摘要表。</p>
                  </div>
                  <el-switch v-model="workflow.form.export.include_tables" />
                </div>

                <div class="switch-card">
                  <div>
                    <strong>导出统计图</strong>
                    <p>保存 Vf、面积、尺寸和通道宽度图表。</p>
                  </div>
                  <el-switch v-model="workflow.form.export.include_charts" />
                </div>
              </template>

              <div class="switch-card">
                <div>
                  <strong>导出掩码</strong>
                  <p>保存分割掩码结果用于复核与论文插图。</p>
                </div>
                <el-switch v-model="workflow.form.export.include_masks" />
              </div>

              <div class="switch-card">
                <div>
                  <strong>导出叠加图</strong>
                  <p>保存原图叠加结果和边界结果。</p>
                </div>
                <el-switch v-model="workflow.form.export.include_overlays" />
              </div>

              <div class="switch-card form-span-2">
                <div>
                  <strong>保存配置快照</strong>
                  <p>保存本次运行的完整配置，便于答辩回溯。</p>
                </div>
                <el-switch v-model="workflow.form.export.include_config_snapshot" />
              </div>
            </div>
          </section>
        </el-form>
      </section>
    </section>

    <aside class="postprocess-side">
      <section class="glass-card runtime-card">
        <div class="section-head compact">
          <div>
            <h2 class="section-title">运行状态</h2>
            <p class="section-subtitle">这里聚焦任务执行、逐图进度、当前图像与失败提示。</p>
          </div>
        </div>

        <div class="run-summary-grid">
          <article v-for="card in launchSummaryCards" :key="card.label" class="summary-card">
            <span>{{ card.label }}</span>
            <strong>{{ card.value }}</strong>
            <p>{{ card.note }}</p>
          </article>
        </div>

        <div class="progress-stack">
          <div class="progress-card">
            <div class="progress-head">
              <strong>执行阶段</strong>
              <span>{{ workflow.activeStepProgressText.value }}</span>
            </div>
            <el-progress :percentage="workflow.executionProgress.value" />
            <small>{{ workflow.launchMessage.value }}</small>
          </div>
        </div>

        <section class="steps-card">
          <div class="section-head compact">
            <div>
              <h2 class="section-title">步骤状态</h2>
              <p class="section-subtitle">这里显示草稿状态与后端执行步骤。</p>
            </div>
          </div>

          <el-timeline>
            <el-timeline-item
              v-for="step in workflow.workflowSteps.value"
              :key="step.key"
              :type="step.status === 'completed' ? 'success' : step.status === 'failed' ? 'danger' : step.status === 'running' ? 'primary' : 'info'"
              :hollow="step.status === 'pending'"
            >
              <strong>{{ step.key }}</strong>
              <p>{{ step.message }}</p>
            </el-timeline-item>
          </el-timeline>
        </section>

        <div class="current-image-card">
          <div class="section-head compact">
            <div>
              <h2 class="section-title">当前图像</h2>
              <p class="section-subtitle">这里显示正在处理的图像、当前步骤和进度提示。</p>
            </div>
          </div>
          <div class="current-image-preview">
            <template v-if="currentImageCard.url">
              <img :src="currentImageCard.url" :alt="currentImageCard.name" />
            </template>
            <template v-else>
              <div class="current-image-placeholder">
                <strong>{{ currentImageCard.name }}</strong>
                <p>{{ currentImageCard.message }}</p>
              </div>
            </template>
          </div>
          <div class="current-image-meta">
            <span>{{ currentImageCard.status }}</span>
            <strong>{{ currentImageCard.name }}</strong>
            <p>{{ currentImageCard.message }}</p>
          </div>
        </div>

        <el-alert
          v-if="failureNotice"
          class="failure-alert"
          :title="failureNotice"
          type="error"
          :closable="false"
          show-icon
        />
      </section>
    </aside>

    <el-dialog
      v-model="workflow.calibrationDialogVisible.value"
      title="未填写标定值"
      width="720px"
      top="8vh"
      destroy-on-close
      append-to-body
      class="calibration-dialog"
    >
      <div class="calibration-dialog-content">
        <p class="calibration-warning">
          当前未填写 <code>um_per_px</code>。任务仍可继续，但面积、尺寸和后续通道宽度只能以像素单位输出，无法得到真实物理量。
        </p>

        <div v-if="workflow.calibrationProbeLoading.value" class="probe-inline-state">
          正在识别底栏和比例尺，请稍候…
        </div>

        <div v-else-if="workflow.calibrationProbe.value" class="calibration-probe-grid">
          <div class="probe-card">
            <span>识别结果</span>
            <strong>
              {{
                workflow.calibrationProbe.value.scale_bar_detected
                  ? '已识别比例尺'
                  : workflow.calibrationProbe.value.footer_detected
                    ? '已检测到底栏'
                    : '未检测到底栏'
              }}
            </strong>
            <p>{{ workflow.calibrationProbeSummary.value }}</p>
          </div>
          <div class="probe-card">
            <span>比例尺</span>
            <strong>
              {{
                workflow.calibrationProbe.value.ocr_scale_bar_um && workflow.calibrationProbe.value.scale_bar_pixels
                  ? `${workflow.calibrationProbe.value.ocr_scale_bar_um} μm / ${workflow.calibrationProbe.value.scale_bar_pixels} px`
                  : '请手动确认'
              }}
            </strong>
            <p v-if="workflow.calibrationProbe.value.ocr_fov_um">FoV {{ workflow.calibrationProbe.value.ocr_fov_um }} μm</p>
            <p v-else>未识别到可直接换算的 FoV。</p>
          </div>
          <div class="probe-card">
            <span>采集信息</span>
            <strong>{{ workflow.calibrationProbe.value.ocr_magnification_text || 'Mag 未识别' }}</strong>
            <p>{{ workflow.calibrationProbe.value.ocr_wd_mm ? `WD ${workflow.calibrationProbe.value.ocr_wd_mm} mm` : 'WD 未识别' }}</p>
          </div>
        </div>

        <div class="calibration-form">
          <el-form label-position="top">
            <el-form-item label="比例尺对应的 μm 数值">
              <el-input-number v-model="workflow.calibrationScaleBarUm.value" :min="0.0001" :step="0.1" :precision="4" />
            </el-form-item>
            <el-form-item label="比例尺像素长度">
              <el-input-number v-model="workflow.calibrationScaleBarPixels.value" :min="1" :step="1" :precision="0" />
            </el-form-item>
          </el-form>
          <div class="calibration-result">
            <span>自动换算结果</span>
            <strong>{{ workflow.calibrationComputedValue.value !== null ? `${workflow.calibrationComputedValue.value.toFixed(6)} um/px` : '--' }}</strong>
          </div>
        </div>

        <div v-if="workflow.calibrationCandidates.value.length" class="candidate-row">
          <span>快捷候选</span>
          <div class="candidate-actions">
            <el-button
              v-if="workflow.calibrationProbe.value?.suggested_um_per_px"
              size="small"
              type="primary"
              plain
              @click="applySuggestedCalibration(workflow.calibrationProbe.value.suggested_um_per_px)"
            >
              OCR 建议值 → {{ workflow.calibrationProbe.value.suggested_um_per_px.toFixed(6) }} um/px
            </el-button>
            <el-button
              v-for="candidate in workflow.calibrationCandidates.value"
              :key="candidate.scale_um"
              size="small"
              plain
              @click="applySuggestedCalibration(candidate.um_per_px)"
            >
              {{ candidate.scale_um }} μm → {{ candidate.um_per_px.toFixed(6) }} um/px
            </el-button>
          </div>
        </div>
      </div>

      <template #footer>
        <div class="dialog-actions">
          <el-button @click="workflow.closeCalibrationDialog()">返回填写</el-button>
          <el-button plain @click="continueWithoutCalibration">继续以像素单位运行</el-button>
          <el-button type="primary" :disabled="workflow.calibrationComputedValue.value === null" @click="applyCalibrationAndLaunch">
            使用换算结果并继续
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.postprocess-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.75fr);
  gap: 18px;
  align-items: start;
}

.postprocess-main {
  min-width: 0;
  display: grid;
  gap: 18px;
}

.postprocess-side {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.postprocess-config-card,
.runtime-card {
  padding: 22px;
}

.section-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 18px;
}

.section-head.compact {
  margin-bottom: 14px;
}

.head-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.postprocess-form {
  display: grid;
  gap: 16px;
}

.form-section {
  padding: 18px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.46);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.section-inline-head {
  margin-bottom: 14px;
}

.section-inline-head h3 {
  margin: 0 0 4px;
  font-size: 18px;
}

.section-inline-head span {
  color: var(--muted);
  font-size: 13px;
}

.form-grid {
  display: grid;
  gap: 14px 16px;
}

.two-columns {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.form-span-2 {
  grid-column: 1 / -1;
}

.switch-card,
.placeholder-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.74);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.switch-card strong,
.placeholder-card strong {
  display: block;
  margin-bottom: 4px;
}

.switch-card p,
.placeholder-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.placeholder-card {
  flex-direction: column;
  align-items: flex-start;
}

.run-summary-grid {
  display: grid;
  gap: 12px;
}

.run-summary-grid {
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}

.summary-card,
.progress-card {
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.summary-card span {
  display: block;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 12px;
}

.summary-card strong {
  display: block;
  font-size: 20px;
  margin-bottom: 8px;
}

.summary-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.progress-stack {
  display: grid;
  gap: 12px;
  margin-top: 18px;
}

.progress-card small {
  display: block;
  margin-top: 10px;
  color: var(--muted);
  line-height: 1.5;
}

.progress-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.progress-head span {
  color: var(--muted);
}

.grid-banner {
  margin-top: 16px;
}

.steps-card {
  margin-top: 16px;
}

.calibration-dialog-content {
  display: grid;
  gap: 16px;
}

.calibration-warning {
  margin: 0;
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(184, 90, 43, 0.08);
  color: #8d4c27;
  line-height: 1.7;
}

.probe-inline-state {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--muted);
}

.calibration-probe-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.probe-card {
  padding: 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.74);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.probe-card span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}

.probe-card strong {
  font-size: 18px;
}

.probe-card p {
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.6;
}

.calibration-form {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 220px;
  gap: 16px;
  align-items: end;
}

.calibration-result {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.74);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.calibration-result span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}

.candidate-row {
  display: grid;
  gap: 10px;
}

.candidate-row > span {
  color: var(--muted);
  font-size: 12px;
}

.candidate-actions,
.dialog-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.measurement-panel {
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.66);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.measurement-panel-head {
  display: grid;
  gap: 4px;
  margin-bottom: 12px;
}

.measurement-panel-head strong {
  font-size: 16px;
}

.measurement-panel-head span {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
}

.measurement-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.measurement-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.76);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.measurement-item strong {
  display: block;
  margin-bottom: 4px;
}

.measurement-item p {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
}

.current-image-card {
  display: grid;
  gap: 12px;
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid rgba(31, 40, 48, 0.06);
}

.current-image-preview {
  min-height: 180px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.66);
  border: 1px solid rgba(31, 40, 48, 0.06);
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.current-image-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.current-image-placeholder {
  padding: 18px;
  text-align: center;
}

.current-image-placeholder strong,
.current-image-meta strong {
  display: block;
  margin-bottom: 8px;
  font-size: 16px;
}

.current-image-placeholder p,
.current-image-meta p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.current-image-meta {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.current-image-meta span {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 0 10px;
  margin-bottom: 10px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}

.failure-alert {
  margin-top: 4px;
}

@media (max-width: 1280px) {
  .postprocess-layout {
    grid-template-columns: 1fr;
  }

  .run-summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .section-head,
  .progress-head {
    flex-direction: column;
  }

  .two-columns,
  .run-summary-grid,
  .calibration-probe-grid,
  .calibration-form {
    grid-template-columns: 1fr;
  }

  .form-span-2 {
    grid-column: auto;
  }

  .switch-card {
    flex-direction: column;
    align-items: flex-start;
  }

  .measurement-grid {
    grid-template-columns: 1fr;
  }
}
</style>
