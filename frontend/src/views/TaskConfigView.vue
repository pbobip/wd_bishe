<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '../api'
import { useTaskWorkflow } from '../composables/useTaskWorkflow'

const router = useRouter()
const route = useRoute()
const workflow = useTaskWorkflow()
const fileInputRef = ref<HTMLInputElement | null>(null)
const folderInputRef = ref<HTMLInputElement | null>(null)
const preprocessPreviewVisible = ref(false)
const preprocessPreviewLoading = ref(false)
const preprocessPreviewIndex = ref(0)
const preprocessPreviewRequestSeq = ref(0)
const preprocessPreview = ref<{
  source_label: string
  footer_detected: boolean
  original_preview_url: string
  processed_preview_url: string
  message: string
  file_name: string
} | null>(null)

const selectedPreviewNames = computed(() => workflow.selectedFiles.value.slice(0, 6).map((file) => file.name))
const remainingCount = computed(() => Math.max(0, workflow.selectedFiles.value.length - selectedPreviewNames.value.length))
const previewFileOptions = computed(() =>
  workflow.selectedFiles.value.map((file, index) => ({
    label: `${index + 1}. ${file.name}`,
    value: index,
  })),
)
const currentPreviewFile = computed(() => workflow.selectedFiles.value[preprocessPreviewIndex.value] ?? null)

const syncFiles = async (event: Event, source: 'files' | 'folder') => {
  const target = event.target as HTMLInputElement
  const files = target.files ? Array.from(target.files) : []
  await workflow.setSelectedFiles(files, source)
  target.value = ''
}

const goToPostprocess = async () => {
  if (!workflow.workflowReady.value) {
    ElMessage.warning('请先填写任务名称并导入图像')
    return
  }
  if (!workflow.ensurePreprocessReady()) {
    return
  }
  const success = await workflow.prepareCurrentTask()
  if (success) {
    router.push('/postprocess')
  }
}

const loadPreprocessPreview = async (index = preprocessPreviewIndex.value) => {
  const targetFile = workflow.selectedFiles.value[index]
  if (!targetFile) {
    preprocessPreview.value = null
    return
  }

  preprocessPreviewIndex.value = index
  const requestSeq = ++preprocessPreviewRequestSeq.value
  preprocessPreviewLoading.value = true
  preprocessPreview.value = null

  try {
    const payload = new FormData()
    payload.append('file', targetFile)
    payload.append('preprocess', JSON.stringify(workflow.form.preprocess))
    payload.append('auto_crop_sem_region', String(workflow.form.input_config.auto_crop_sem_region))
    const { data } = await api.post('/preprocess/preview', payload, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    if (requestSeq !== preprocessPreviewRequestSeq.value) return
    preprocessPreview.value = {
      ...data,
      file_name: targetFile.name,
    }
  } catch (error: any) {
    if (requestSeq !== preprocessPreviewRequestSeq.value) return
    ElMessage.error(error?.response?.data?.detail ?? error?.message ?? '生成预处理预览失败')
    preprocessPreviewVisible.value = false
  } finally {
    if (requestSeq === preprocessPreviewRequestSeq.value) {
      preprocessPreviewLoading.value = false
    }
  }
}

const openPreprocessPreview = async () => {
  if (!workflow.selectedFiles.value.length) {
    ElMessage.warning('请先选择至少一张图像，再查看预处理预览')
    return
  }
  if (!workflow.ensurePreprocessReady()) {
    return
  }

  if (preprocessPreviewIndex.value >= workflow.selectedFiles.value.length) {
    preprocessPreviewIndex.value = 0
  }

  preprocessPreviewVisible.value = true
  await loadPreprocessPreview(preprocessPreviewIndex.value)
}

const changePreviewFile = async (index: number) => {
  if (index === preprocessPreviewIndex.value && preprocessPreview.value) {
    return
  }
  await loadPreprocessPreview(index)
}

const shiftPreviewFile = async (offset: number) => {
  const nextIndex = preprocessPreviewIndex.value + offset
  if (nextIndex < 0 || nextIndex >= workflow.selectedFiles.value.length) {
    return
  }
  await changePreviewFile(nextIndex)
}

watch(
  () => workflow.selectedFiles.value.length,
  (length) => {
    if (!length) {
      preprocessPreviewVisible.value = false
      preprocessPreview.value = null
      preprocessPreviewIndex.value = 0
      preprocessPreviewRequestSeq.value += 1
      preprocessPreviewLoading.value = false
      return
    }
    if (preprocessPreviewIndex.value >= length) {
      preprocessPreviewIndex.value = 0
    }
  },
)

onMounted(() => {
  workflow.initialize(Number(route.query.project_id || 0) || null).catch(() => {
    ElMessage.error('任务创建页初始化失败')
  })
})
</script>

<template>
  <div class="task-create-layout">
    <section class="glass-card task-create-main">
      <div class="section-head">
        <div>
          <h2 class="section-title">任务创建</h2>
          <p class="section-subtitle">导入图像，配置标定、预处理与分割方式，再进入后处理与运行。</p>
        </div>
          <div class="head-actions">
            <el-button class="back-project-button" @click="router.push('/projects')">
              <span class="back-project-button__icon" aria-hidden="true">←</span>
              <span>返回项目</span>
            </el-button>
            <el-button
              type="primary"
              size="large"
              :disabled="!workflow.workflowReady.value"
              :loading="workflow.actionLoading.value"
              @click="goToPostprocess"
            >
              下一步：后处理
            </el-button>
          </div>
      </div>

      <el-form label-position="top" class="task-create-form">
        <section class="form-section">
          <div class="section-inline-head">
            <h3>基础信息与导入</h3>
            <span>先确定项目、任务名和图像来源，输入方式由单图/多图/文件夹自动判定。</span>
          </div>

          <div class="form-grid two-columns">
            <el-form-item v-if="workflow.projects.value.length > 1" label="所属项目">
              <el-select v-model="workflow.form.project_id">
                <el-option
                  v-for="project in workflow.projects.value"
                  :key="project.id"
                  :label="project.name"
                  :value="project.id"
                />
              </el-select>
            </el-form-item>

            <el-form-item :class="{ 'form-span-2': workflow.projects.value.length <= 1 }" label="任务名称">
              <el-input v-model="workflow.form.name" />
            </el-form-item>

            <el-form-item class="form-span-2" label="图像导入">
              <div class="upload-actions">
                <el-button @click="fileInputRef?.click()">选择图片文件</el-button>
                <el-button plain @click="folderInputRef?.click()">选择文件夹</el-button>
              </div>

              <input
                ref="fileInputRef"
                type="file"
                multiple
                accept=".png,.jpg,.jpeg,.tif,.tiff"
                class="hidden-input"
                @change="(event) => syncFiles(event, 'files')"
              />
              <input
                ref="folderInputRef"
                type="file"
                multiple
                webkitdirectory
                directory
                accept=".png,.jpg,.jpeg,.tif,.tiff"
                class="hidden-input"
                @change="(event) => syncFiles(event, 'folder')"
              />

              <div v-if="workflow.selectedFiles.value.length" class="upload-status-bar">
                <div class="upload-tip">
                  <span class="upload-count-tag">已选 {{ workflow.selectedFiles.value.length }} 张图像</span>
                  <span class="upload-mode-tag">{{ workflow.inputModeLabel }}</span>
                </div>
                <el-button class="upload-clear-button" plain type="danger" @click="workflow.clearSelectedFiles()">
                  清空已选
                </el-button>
              </div>
            </el-form-item>

            <div class="selection-card form-span-2">
              <div class="selection-head">
                <strong>本次导入清单</strong>
                <span>{{ workflow.selectedFiles.value.length ? '已就绪，可进入后处理页' : '尚未选择图像' }}</span>
              </div>
              <el-empty v-if="!workflow.selectedFiles.value.length" description="支持单张、多张或整文件夹导入" />
              <div v-else class="selection-list">
                <div v-for="name in selectedPreviewNames" :key="name" class="selection-item">
                  <strong>{{ name }}</strong>
                </div>
                <div v-if="remainingCount" class="selection-item selection-item-muted">
                  还有 {{ remainingCount }} 张图像未展开
                </div>
              </div>
            </div>

            <div
              v-if="workflow.selectedFiles.value.length && ['creating', 'uploading', 'ready'].includes(workflow.launchPhase.value)"
              class="upload-progress-card form-span-2"
            >
              <div class="upload-progress-card__head">
                <div>
                  <strong>图像上传与登记</strong>
                  <p>任务创建页会先完成草稿任务创建和图片上传，后处理页只负责运行与统计。</p>
                </div>
                <span>{{ workflow.uploadProgressLabel.value }}</span>
              </div>
              <el-progress :percentage="Math.round(workflow.uploadState.percentage)" />
              <small>
                当前共 {{ Math.max(workflow.uploadState.totalChunks, workflow.selectedFiles.value.length ? Math.ceil(workflow.selectedFiles.value.length / 10) : 0) }} 批，
                已上传 {{ workflow.uploadState.uploadedBytes }} / {{ workflow.uploadState.totalBytes || 0 }} bytes。
              </small>
            </div>
          </div>
        </section>

        <section class="form-section">
          <div class="section-inline-head">
            <h3>标定与输入</h3>
            <span>这一块只负责单位换算和 SEM 有效分析区裁剪，不混入后处理运行逻辑。</span>
          </div>

          <div class="form-grid two-columns">
            <el-form-item class="form-span-2" label="比例尺标定（um_per_px）">
              <div class="calibration-field">
                <div class="calibration-input-row">
                  <el-input-number
                    v-model="workflow.form.input_config.um_per_px"
                    :min="0.0001"
                    :step="0.001"
                    :precision="4"
                  />
                  <span class="calibration-unit">单位：um / px</span>
                  <el-button
                    plain
                    :loading="workflow.calibrationProbeLoading.value"
                    :disabled="!workflow.selectedFiles.value.length"
                    @click="workflow.runCalibrationProbe(true)"
                  >
                    {{ workflow.calibrationActionLabel.value }}
                  </el-button>
                  <el-button
                    v-if="workflow.calibrationSuggestedValue.value"
                    type="primary"
                    plain
                    @click="workflow.applyCalibrationValue(workflow.calibrationSuggestedValue.value)"
                  >
                    使用建议值
                  </el-button>
                </div>
                <div class="field-helper">
                  <span>不填写时，面积、尺寸和通道宽度只会输出 px / px²。</span>
                  <span class="field-helper-status">{{ workflow.calibrationReminder }}</span>
                  <span v-if="workflow.calibrationProbeSummary.value" class="field-helper-hint">{{ workflow.calibrationProbeSummary.value }}</span>
                </div>
              </div>
            </el-form-item>

            <div class="switch-card">
              <div>
                <strong>自动裁分析区域</strong>
                <p>默认去掉底部信息栏与无效背景，避免影响后续统计。</p>
              </div>
              <el-switch v-model="workflow.form.input_config.auto_crop_sem_region" />
            </div>

            <div class="switch-card">
              <div>
                <strong>保存底部信息栏</strong>
                <p>{{ workflow.calibrationFooterHint.value }}</p>
              </div>
              <el-switch v-model="workflow.form.input_config.save_sem_footer" />
            </div>
          </div>
        </section>

        <section class="form-section">
          <div class="section-inline-head">
            <h3>分割方法</h3>
            <span>这里只配置分割路线和模型参数；后处理、统计和导出移到下一页。</span>
          </div>

          <div class="mode-selector">
            <div class="mode-hint-card">
              <strong>{{ workflow.currentModeLabel }}</strong>
              <p>{{ workflow.segmentationHint }}</p>
            </div>
            <el-segmented
              v-model="workflow.form.segmentation_mode"
              :options="workflow.segmentationOptions"
              class="mode-segmented"
            />
          </div>

          <div class="form-grid two-columns form-grid--after-mode">
            <template v-if="workflow.form.segmentation_mode !== 'dl'">
              <el-form-item class="form-span-2" label="传统路线">
                <div class="traditional-method-panel">
                  <el-segmented
                    v-model="workflow.form.traditional_seg.method"
                    :options="workflow.traditionalMethodOptions"
                    class="traditional-method-segmented"
                  />
                  <div class="traditional-method-hint">
                    <strong>{{ workflow.traditionalMethodOptions.find((item) => item.value === workflow.form.traditional_seg.method)?.label }}</strong>
                    <p>{{ workflow.traditionalMethodHint }}</p>
                  </div>
                </div>
              </el-form-item>

              <div class="traditional-config-shell form-span-2">
                <section class="traditional-config-card">
                  <div class="traditional-config-card__head">
                    <strong>公共参数</strong>
                    <span>不同路线共用的边界输出与观察入口，切换路线时保持稳定。</span>
                  </div>
                  <div class="traditional-config-grid">
                    <el-form-item label="边界图算子">
                      <el-select v-model="workflow.form.traditional_seg.edge_operator">
                        <el-option label="Canny" value="canny" />
                        <el-option label="Sobel" value="sobel" />
                        <el-option label="Laplacian" value="laplacian" />
                      </el-select>
                    </el-form-item>
                  </div>
                </section>

                <section class="traditional-config-card route-config-card">
                  <div class="traditional-config-card__head">
                    <strong>路线专属参数</strong>
                    <span>这里只切换当前路线的核心参数，外层布局保持不变。</span>
                  </div>

                  <Transition name="route-panel" mode="out-in">
                    <div :key="workflow.form.traditional_seg.method" class="route-config-surface">
                      <div v-if="workflow.form.traditional_seg.method === 'threshold'" class="route-config-grid">
                        <el-form-item label="阈值模式">
                          <el-select v-model="workflow.form.traditional_seg.threshold_mode">
                            <el-option label="Otsu" value="otsu" />
                            <el-option label="全局阈值" value="global" />
                            <el-option label="固定阈值" value="fixed" />
                          </el-select>
                        </el-form-item>

                        <el-form-item v-if="workflow.form.traditional_seg.threshold_mode === 'global'" label="全局阈值">
                          <el-input-number v-model="workflow.form.traditional_seg.global_threshold" :min="1" :max="254" />
                        </el-form-item>

                        <el-form-item v-else-if="workflow.form.traditional_seg.threshold_mode === 'fixed'" label="固定阈值">
                          <el-input-number v-model="workflow.form.traditional_seg.fixed_threshold" :min="1" :max="254" />
                        </el-form-item>

                        <div v-else class="route-inline-note route-span-2">
                          <strong>当前为 Otsu 自动阈值</strong>
                          <span>系统会自动估计最佳分割阈值，不需要再手动输入具体数值。</span>
                        </div>
                      </div>

                      <div v-else-if="workflow.form.traditional_seg.method === 'adaptive'" class="route-config-grid">
                        <el-form-item label="自适应模式">
                          <el-select v-model="workflow.form.traditional_seg.adaptive_method">
                            <el-option label="Gaussian" value="gaussian" />
                            <el-option label="Mean" value="mean" />
                          </el-select>
                        </el-form-item>

                        <el-form-item label="局部窗口大小">
                          <el-input-number v-model="workflow.form.traditional_seg.adaptive_block_size" :min="3" :step="2" />
                        </el-form-item>

                        <el-form-item class="route-span-2" label="自适应偏移 C">
                          <el-input-number v-model="workflow.form.traditional_seg.adaptive_c" :min="-50" :max="50" :step="1" />
                        </el-form-item>
                      </div>

                      <div v-else-if="workflow.form.traditional_seg.method === 'edge'" class="route-config-grid">
                        <el-form-item label="边缘预模糊核">
                          <el-input-number v-model="workflow.form.traditional_seg.edge_blur_kernel" :min="1" :step="2" />
                        </el-form-item>

                        <el-form-item label="边缘膨胀次数">
                          <el-input-number v-model="workflow.form.traditional_seg.edge_dilate_iterations" :min="0" :step="1" />
                        </el-form-item>

                        <template v-if="workflow.form.traditional_seg.edge_operator === 'canny'">
                          <el-form-item label="Canny 下阈值">
                            <el-input-number v-model="workflow.form.traditional_seg.edge_threshold1" :min="0" :max="255" />
                          </el-form-item>
                          <el-form-item label="Canny 上阈值">
                            <el-input-number v-model="workflow.form.traditional_seg.edge_threshold2" :min="0" :max="255" />
                          </el-form-item>
                        </template>

                        <div v-else class="route-static-note route-span-2">
                          <strong>{{ workflow.form.traditional_seg.edge_operator === 'sobel' ? 'Sobel' : 'Laplacian' }} 算子使用自动阈值</strong>
                          <p>当前路线会先计算边缘强度，再自动估计边缘输出的二值阈值。</p>
                        </div>
                      </div>

                      <div v-else class="route-config-grid">
                        <el-form-item label="聚类数">
                          <el-input-number v-model="workflow.form.traditional_seg.kmeans_clusters" :min="2" :max="6" :step="1" />
                        </el-form-item>

                        <el-form-item label="尝试次数">
                          <el-input-number v-model="workflow.form.traditional_seg.kmeans_attempts" :min="1" :max="20" :step="1" />
                        </el-form-item>

                        <el-form-item class="route-span-2" label="目标簇选择">
                          <el-select v-model="workflow.form.traditional_seg.cluster_target">
                            <el-option label="亮相簇" value="bright" />
                            <el-option label="暗相簇" value="dark" />
                            <el-option label="最大簇" value="largest" />
                          </el-select>
                        </el-form-item>
                      </div>
                    </div>
                  </Transition>
                </section>
              </div>
            </template>

            <template v-if="workflow.form.segmentation_mode !== 'traditional'">
              <el-form-item label="模型槽位">
                <el-select v-model="workflow.form.dl_model.model_slot">
                  <el-option label="MatSAM" value="matsam" />
                  <el-option label="SAM LoRA" value="sam_lora" />
                  <el-option label="ResNeXt50" value="resnext50" />
                  <el-option label="Custom 占位运行器" value="custom" />
                </el-select>
              </el-form-item>
              <el-form-item label="权重路径">
                <el-input v-model="workflow.form.dl_model.weight_path" placeholder="可留空，后续在模型运行器里补齐" />
              </el-form-item>
              <el-form-item label="运行设备">
                <el-select v-model="workflow.form.dl_model.device">
                  <el-option label="自动" value="auto" />
                  <el-option label="CUDA" value="cuda" />
                  <el-option label="CPU" value="cpu" />
                </el-select>
              </el-form-item>
              <el-form-item label="输入尺寸">
                <el-input-number v-model="workflow.form.dl_model.input_size" :min="256" :step="128" />
              </el-form-item>
              <el-form-item class="form-span-2" label="分割阈值">
                <el-slider v-model="workflow.form.dl_model.threshold" :min="0.1" :max="0.9" :step="0.05" />
              </el-form-item>
            </template>
          </div>
        </section>

        <section class="form-section">
          <div class="section-inline-head section-inline-head--actions section-inline-head--card-aligned">
            <div>
              <h3>预处理</h3>
              <span>输入规范化已在上方“标定与输入”完成，这里只配置背景校正、主去噪、主增强和可选增强。</span>
            </div>
            <el-button plain :disabled="!workflow.selectedFiles.value.length" :loading="preprocessPreviewLoading" @click="openPreprocessPreview">
              查看预处理预览
            </el-button>
          </div>

          <div class="form-grid two-columns">
            <div class="switch-card form-span-2">
              <div>
                <strong>预处理开关</strong>
                <p>{{ workflow.preprocessSummary }}</p>
              </div>
              <el-switch v-model="workflow.form.preprocess.enabled" active-text="启用" inactive-text="跳过" />
            </div>

            <template v-if="workflow.form.preprocess.enabled">
              <div class="preprocess-preset-bar form-span-2">
                <div>
                  <strong>推荐预设</strong>
                  <p>先选一个接近当前任务的默认流程，再按图像特性微调。</p>
                </div>
                <div class="preset-actions">
                  <el-button plain @click="workflow.applyPreprocessPreset('traditional')">传统分割推荐</el-button>
                  <el-button plain @click="workflow.applyPreprocessPreset('dl')">深度学习推荐</el-button>
                  <el-button plain @click="workflow.applyPreprocessPreset('compare')">结果对比推荐</el-button>
                </div>
              </div>

              <div class="preprocess-workflow-grid form-span-2">
                <section class="traditional-config-card">
                  <div class="traditional-config-card__head">
                    <strong>背景校正</strong>
                    <span>{{ workflow.preprocessBackgroundHint.value }}</span>
                  </div>
                  <el-segmented
                    v-model="workflow.form.preprocess.background.method"
                    :options="workflow.preprocessBackgroundOptions"
                    class="preprocess-segmented"
                  />
                  <Transition name="route-panel" mode="out-in">
                    <div :key="workflow.form.preprocess.background.method" class="route-config-surface preprocess-surface">
                      <div v-if="workflow.form.preprocess.background.method === 'none'" class="route-static-note">
                        <strong>当前不做背景校正</strong>
                        <p>适合原图亮度已经稳定的情况，后续直接进入去噪和增强。</p>
                      </div>
                      <div v-else class="route-config-grid preprocess-param-grid">
                        <el-form-item class="route-span-2" label="半径 / 窗口大小">
                          <el-input-number v-model="workflow.form.preprocess.background.radius" :min="3" :step="2" />
                        </el-form-item>
                      </div>
                    </div>
                  </Transition>
                </section>

                <section class="traditional-config-card">
                  <div class="traditional-config-card__head">
                    <strong>主去噪方法</strong>
                    <span>{{ workflow.preprocessDenoiseHint.value }}</span>
                  </div>
                  <el-segmented
                    v-model="workflow.form.preprocess.denoise.method"
                    :options="workflow.preprocessDenoiseOptions"
                    class="preprocess-segmented"
                  />
                  <Transition name="route-panel" mode="out-in">
                    <div :key="workflow.form.preprocess.denoise.method" class="route-config-surface preprocess-surface">
                      <div v-if="workflow.form.preprocess.denoise.method === 'none'" class="route-static-note">
                        <strong>当前跳过去噪</strong>
                        <p>适合原图质量较稳或希望深度学习直接接收原始灰度分布的场景。</p>
                      </div>
                      <div v-else class="route-config-grid preprocess-param-grid">
                        <el-form-item
                          v-if="workflow.form.preprocess.denoise.method === 'wavelet'"
                          class="route-span-2"
                          label="小波强度"
                        >
                          <el-slider v-model="workflow.form.preprocess.denoise.wavelet_strength" :min="0.02" :max="0.3" :step="0.01" />
                        </el-form-item>
                        <el-form-item
                          v-else-if="workflow.form.preprocess.denoise.method === 'gaussian'"
                          class="route-span-2"
                          label="高斯滤波核"
                        >
                          <el-input-number v-model="workflow.form.preprocess.denoise.gaussian_kernel" :min="1" :step="2" />
                        </el-form-item>
                        <el-form-item
                          v-else-if="workflow.form.preprocess.denoise.method === 'median'"
                          class="route-span-2"
                          label="中值滤波核"
                        >
                          <el-input-number v-model="workflow.form.preprocess.denoise.median_kernel" :min="1" :step="2" />
                        </el-form-item>
                        <template v-else-if="workflow.form.preprocess.denoise.method === 'bilateral'">
                          <el-form-item label="滤波直径">
                            <el-input-number v-model="workflow.form.preprocess.denoise.bilateral_diameter" :min="3" :step="2" />
                          </el-form-item>
                          <el-form-item label="灰度域 sigma">
                            <el-input-number v-model="workflow.form.preprocess.denoise.bilateral_sigma_color" :min="1" :step="1" />
                          </el-form-item>
                          <el-form-item class="route-span-2" label="空间域 sigma">
                            <el-input-number v-model="workflow.form.preprocess.denoise.bilateral_sigma_space" :min="1" :step="1" />
                          </el-form-item>
                        </template>
                        <el-form-item v-else class="route-span-2" label="均值滤波核">
                          <el-input-number v-model="workflow.form.preprocess.denoise.mean_kernel" :min="1" :step="2" />
                        </el-form-item>
                      </div>
                    </div>
                  </Transition>
                </section>

                <section class="traditional-config-card">
                  <div class="traditional-config-card__head">
                    <strong>主增强方法</strong>
                    <span>{{ workflow.preprocessEnhanceHint.value }}</span>
                  </div>
                  <el-segmented
                    v-model="workflow.form.preprocess.enhance.method"
                    :options="workflow.preprocessEnhanceOptions"
                    class="preprocess-segmented"
                  />
                  <Transition name="route-panel" mode="out-in">
                    <div :key="workflow.form.preprocess.enhance.method" class="route-config-surface preprocess-surface">
                      <div v-if="workflow.form.preprocess.enhance.method === 'none'" class="route-static-note">
                        <strong>当前不做主增强</strong>
                        <p>保留去噪后的自然灰度分布，减少对分割输入的额外干预。</p>
                      </div>
                      <div v-else class="route-config-grid preprocess-param-grid">
                        <template v-if="workflow.form.preprocess.enhance.method === 'clahe'">
                          <el-form-item label="CLAHE clip limit">
                            <el-input-number
                              v-model="workflow.form.preprocess.enhance.clahe_clip_limit"
                              :min="0.1"
                              :step="0.1"
                            />
                          </el-form-item>
                          <el-form-item label="Tile size">
                            <el-input-number v-model="workflow.form.preprocess.enhance.clahe_tile_size" :min="2" :step="1" />
                          </el-form-item>
                        </template>
                        <el-form-item v-else-if="workflow.form.preprocess.enhance.method === 'gamma'" class="route-span-2" label="Gamma 值">
                          <el-slider v-model="workflow.form.preprocess.enhance.gamma" :min="0.4" :max="2.5" :step="0.05" />
                        </el-form-item>
                        <div v-else class="route-static-note route-span-2">
                          <strong>当前为全局直方图均衡化</strong>
                          <p>系统会对整体灰度分布做全局拉伸，不需要额外参数。</p>
                        </div>
                      </div>
                    </div>
                  </Transition>
                </section>

                <section class="traditional-config-card traditional-config-card--extras">
                  <div class="traditional-config-card__head">
                    <strong>可选增强</strong>
                    <span>仅在主增强后轮廓偏软时，再叠加轻度锐化。</span>
                  </div>
                  <div class="extras-shell">
                    <div class="switch-card switch-card--inner">
                      <strong>轻度锐化（Unsharp）</strong>
                      <el-switch v-model="workflow.form.preprocess.extras.unsharp" />
                    </div>
                    <Transition name="route-panel" mode="out-in">
                      <div v-if="workflow.form.preprocess.extras.unsharp" key="unsharp" class="route-config-grid preprocess-param-grid">
                        <el-form-item label="锐化半径">
                          <el-input-number v-model="workflow.form.preprocess.extras.unsharp_radius" :min="1" :step="2" />
                        </el-form-item>
                        <el-form-item label="锐化强度">
                          <el-slider v-model="workflow.form.preprocess.extras.unsharp_amount" :min="0.1" :max="2.0" :step="0.1" />
                        </el-form-item>
                      </div>
                      <div v-else key="no-unsharp" class="route-static-note">
                        <strong>当前未启用额外增强</strong>
                        <p>建议先观察背景校正、主去噪和主增强效果，只有轮廓偏软时再打开锐化。</p>
                      </div>
                    </Transition>
                  </div>
                </section>
              </div>

              <div class="preprocess-preview-entry form-span-2">
                <div>
                  <strong>预处理前后预览</strong>
                  <p>批量任务按单张抽样核查，弹窗内可切换不同图像查看“分析区域 / 预处理后图”。</p>
                  <p class="preprocess-preview-entry__note">预览直接基于当前本地选中的图像即时生成，不依赖上传登记是否完成。</p>
                </div>
                <el-button type="primary" plain :disabled="!workflow.selectedFiles.value.length" :loading="preprocessPreviewLoading" @click="openPreprocessPreview">
                  查看预处理预览
                </el-button>
              </div>

              <p v-if="workflow.preprocessValidationMessage.value" class="field-validation form-span-2">
                {{ workflow.preprocessValidationMessage.value }}
              </p>
            </template>
          </div>
        </section>

      </el-form>
    </section>
    <el-dialog v-model="preprocessPreviewVisible" title="预处理预览" width="min(1080px, 92vw)" destroy-on-close>
      <div class="preview-dialog-body">
        <div class="preview-dialog-meta">
          <strong>{{ preprocessPreview?.file_name ?? currentPreviewFile?.name ?? '图像预览' }}</strong>
          <p>{{ preprocessPreview?.message ?? '批量任务按单张抽样查看，不会一次性生成整批预处理预览。' }}</p>
        </div>
        <div v-if="workflow.selectedFiles.value.length > 1" class="preview-dialog-toolbar">
          <el-button plain :disabled="preprocessPreviewLoading || preprocessPreviewIndex <= 0" @click="shiftPreviewFile(-1)">
            上一张
          </el-button>
          <el-select
            :model-value="preprocessPreviewIndex"
            class="preview-dialog-select"
            filterable
            placeholder="选择要预览的图像"
            @change="(value) => changePreviewFile(Number(value))"
          >
            <el-option
              v-for="option in previewFileOptions"
              :key="option.value"
              :label="option.label"
              :value="option.value"
            />
          </el-select>
          <el-button
            plain
            :disabled="preprocessPreviewLoading || preprocessPreviewIndex >= workflow.selectedFiles.value.length - 1"
            @click="shiftPreviewFile(1)"
          >
            下一张
          </el-button>
          <span class="preview-dialog-counter">
            第 {{ preprocessPreviewIndex + 1 }} / {{ workflow.selectedFiles.value.length }} 张
          </span>
        </div>
        <el-skeleton :loading="preprocessPreviewLoading" animated>
          <template #template>
            <div class="preview-dialog-grid">
              <div class="preview-panel preview-panel--skeleton" />
              <div class="preview-panel preview-panel--skeleton" />
            </div>
          </template>
          <div class="preview-dialog-grid" v-if="preprocessPreview">
            <div class="preview-panel">
              <div class="preview-panel__head">
                <strong>{{ preprocessPreview.source_label }}</strong>
                <span>{{ preprocessPreview.footer_detected ? '已自动裁出分析区域' : '未检测到底栏，使用原图' }}</span>
              </div>
              <img :src="preprocessPreview.original_preview_url" alt="预处理前图像" class="preview-panel__image" />
            </div>
            <div class="preview-panel">
              <div class="preview-panel__head">
                <strong>预处理后图像</strong>
                <span>按当前 5 层工作流生成</span>
              </div>
              <img :src="preprocessPreview.processed_preview_url" alt="预处理后图像" class="preview-panel__image" />
            </div>
          </div>
        </el-skeleton>
      </div>
      <template #footer>
        <el-button @click="preprocessPreviewVisible = false">关闭</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.task-create-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 18px;
  align-items: start;
}

.task-create-main {
  padding: 22px;
}

.section-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 18px;
}

.head-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  align-items: center;
  gap: 10px;
}

.head-actions :deep(.el-button) {
  min-height: 48px;
  border-radius: 14px;
}

.back-project-button {
  padding-inline: 18px;
  border-color: rgba(23, 96, 135, 0.12);
  background: rgba(255, 255, 255, 0.8);
  color: var(--ink);
  font-weight: 600;
  box-shadow: 0 10px 24px rgba(21, 40, 62, 0.06);
}

.back-project-button:hover {
  border-color: rgba(23, 96, 135, 0.22);
  background: rgba(23, 96, 135, 0.06);
  color: var(--accent);
}

.back-project-button__icon {
  display: inline-flex;
  align-items: center;
  margin-right: 6px;
  color: var(--accent);
  font-size: 14px;
  line-height: 1;
}

.task-create-form {
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

.section-inline-head--actions {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.section-inline-head--card-aligned {
  padding-left: 16px;
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

.form-grid--after-mode {
  margin-top: 16px;
}

.form-span-2 {
  grid-column: 1 / -1;
}

.upload-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.upload-status-bar {
  margin-top: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.upload-tip {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.upload-count-tag {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 0 12px;
  border-radius: 999px;
  background: rgba(31, 40, 48, 0.05);
  color: var(--ink);
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.upload-mode-tag {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 0 12px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-weight: 600;
  font-size: 12px;
  white-space: nowrap;
}

.upload-clear-button {
  flex: 0 0 auto;
}

:deep(.upload-clear-button.el-button) {
  min-height: 36px;
  padding-inline: 14px;
  border-radius: 12px;
  border-color: rgba(239, 93, 93, 0.2);
  background: rgba(255, 94, 94, 0.04);
  color: #de5a5a;
}

:deep(.upload-clear-button.el-button:hover) {
  border-color: rgba(239, 93, 93, 0.3);
  background: rgba(255, 94, 94, 0.08);
  color: #d84e4e;
}

.selection-card {
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.selection-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}

.selection-head span {
  color: var(--muted);
  font-size: 12px;
}

.selection-list {
  display: grid;
  gap: 10px;
}

.selection-item {
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.selection-item strong {
  font-size: 14px;
  word-break: break-all;
}

.selection-item-muted {
  color: var(--muted);
  text-align: center;
  font-size: 13px;
}

.upload-progress-card {
  display: grid;
  gap: 12px;
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.upload-progress-card__head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.upload-progress-card__head strong {
  display: block;
  margin-bottom: 4px;
  font-size: 15px;
}

.upload-progress-card__head p,
.upload-progress-card small {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.upload-progress-card__head span {
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.calibration-field {
  display: grid;
  gap: 12px;
}

.calibration-input-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}

.calibration-unit {
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
}

:deep(.calibration-input-row .el-input-number) {
  width: min(260px, 100%);
}

.field-helper {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(31, 40, 48, 0.06);
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
}

.field-helper-hint {
  color: var(--accent);
  font-weight: 600;
}

.field-helper-status {
  color: #a85b1f;
}

.switch-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.74);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.switch-card strong {
  display: block;
  margin-bottom: 4px;
}

.switch-card p {
  margin: 0;
  color: var(--muted);
}

.mode-selector {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.preprocess-preset-bar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.preprocess-preset-bar strong {
  display: block;
  margin-bottom: 4px;
  font-size: 15px;
}

.preprocess-preset-bar p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.preset-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.preprocess-workflow-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.preprocess-surface {
  min-height: 164px;
}

.preprocess-param-grid {
  align-content: start;
}

.extras-shell {
  display: grid;
  gap: 12px;
}

.switch-card--inner {
  min-height: 40px;
  padding: 2px 0;
  background: transparent;
  border: 0;
  align-items: center;
}

.switch-card--inner strong {
  margin: 0;
  font-size: 15px;
}

.preprocess-preview-entry {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.preprocess-preview-entry strong {
  display: block;
  margin-bottom: 4px;
  font-size: 15px;
}

.preprocess-preview-entry p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.preprocess-preview-entry__note {
  margin-top: 6px !important;
  color: var(--accent) !important;
  font-size: 12px;
  font-weight: 600;
}

.traditional-method-panel {
  display: grid;
  gap: 12px;
}

.traditional-config-shell {
  display: grid;
  grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
  gap: 16px;
  align-items: stretch;
}

.traditional-config-card {
  display: grid;
  gap: 12px;
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.traditional-config-card--extras {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}

.traditional-config-card--extras .extras-shell {
  display: flex;
  flex-direction: column;
}

.traditional-config-card--extras .switch-card--inner {
  padding: 0;
}

.traditional-config-card__head {
  display: grid;
  gap: 4px;
}

.traditional-config-card__head strong {
  font-size: 15px;
}

.traditional-config-card__head span {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
}

.traditional-config-grid,
.route-config-grid {
  display: grid;
  gap: 14px 16px;
}

.route-config-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.route-config-surface {
  min-height: 140px;
}

.route-span-2 {
  grid-column: 1 / -1;
}

.route-static-note {
  display: grid;
  gap: 6px;
  align-content: start;
  padding: 14px 16px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.06), rgba(184, 90, 43, 0.05));
  border: 1px solid rgba(23, 96, 135, 0.12);
}

.route-static-note strong {
  font-size: 14px;
}

.route-static-note p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.route-inline-note {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 42px;
  padding: 0 2px;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.6;
}

.route-inline-note strong {
  color: var(--ink);
  font-size: 14px;
  white-space: nowrap;
}

.route-inline-note span {
  min-width: 0;
}

.traditional-method-hint {
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.traditional-method-hint strong {
  display: block;
  margin-bottom: 6px;
  font-size: 15px;
}

.traditional-method-hint p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.mode-hint-card {
  padding: 14px 16px;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.09), rgba(184, 90, 43, 0.08));
  border: 1px solid rgba(23, 96, 135, 0.14);
}

.mode-hint-card strong {
  display: block;
  font-size: 18px;
}

.mode-hint-card p {
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.6;
}

:deep(.mode-segmented) {
  width: 100%;
  padding: 4px;
  background: rgba(255, 255, 255, 0.72);
  border-radius: 16px;
}

:deep(.mode-segmented .el-segmented__item) {
  min-height: 44px;
  font-weight: 700;
  color: var(--muted);
}

:deep(.mode-segmented .el-segmented__item-selected) {
  color: #ffffff;
}

:deep(.mode-segmented .el-segmented__thumb) {
  background: linear-gradient(135deg, #176087, #2f8bc0);
  border-radius: 12px;
  box-shadow: 0 10px 22px rgba(23, 96, 135, 0.24);
}

:deep(.traditional-method-segmented) {
  width: 100%;
  padding: 4px;
  background: rgba(255, 255, 255, 0.72);
  border-radius: 16px;
}

:deep(.traditional-method-segmented .el-segmented__item) {
  min-height: 40px;
  font-weight: 700;
  color: var(--muted);
}

:deep(.traditional-method-segmented .el-segmented__item-selected) {
  color: #ffffff;
}

:deep(.traditional-method-segmented .el-segmented__thumb) {
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.92), rgba(184, 90, 43, 0.92));
  border-radius: 12px;
  box-shadow: 0 8px 18px rgba(23, 96, 135, 0.18);
}

:deep(.preprocess-segmented) {
  width: 100%;
  padding: 4px;
  background: rgba(255, 255, 255, 0.72);
  border-radius: 16px;
}

:deep(.preprocess-segmented .el-segmented__group) {
  flex-wrap: wrap;
}

:deep(.preprocess-segmented .el-segmented__item) {
  min-height: 40px;
  font-weight: 700;
  color: var(--muted);
}

:deep(.preprocess-segmented .el-segmented__item-selected) {
  color: #ffffff;
}

:deep(.preprocess-segmented .el-segmented__thumb) {
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.92), rgba(47, 139, 192, 0.92));
  border-radius: 12px;
}

.route-panel-enter-active,
.route-panel-leave-active {
  transition: opacity 0.16s ease;
}

.route-panel-enter-from,
.route-panel-leave-to {
  opacity: 0;
}

.hidden-input {
  display: none;
}

.preview-dialog-body {
  display: grid;
  gap: 16px;
}

.preview-dialog-meta strong {
  display: block;
  margin-bottom: 4px;
}

.preview-dialog-meta p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.preview-dialog-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.preview-dialog-select {
  flex: 1 1 320px;
  min-width: 0;
}

.preview-dialog-counter {
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.preview-dialog-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.preview-panel {
  display: grid;
  gap: 12px;
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.preview-panel--skeleton {
  min-height: 320px;
}

.preview-panel__head {
  display: grid;
  gap: 4px;
}

.preview-panel__head span {
  color: var(--muted);
  font-size: 12px;
}

.preview-panel__image {
  width: 100%;
  max-height: 58vh;
  object-fit: contain;
  border-radius: 14px;
  background: rgba(18, 20, 24, 0.04);
}

.field-validation {
  margin: 0;
  font-size: 13px;
  color: #b2562d;
  font-weight: 600;
}

@media (max-width: 760px) {
  .section-head {
    flex-direction: column;
  }

  .section-inline-head--actions,
  .preprocess-preset-bar,
  .preprocess-preview-entry,
  .upload-progress-card__head,
  .preview-dialog-toolbar {
    flex-direction: column;
    align-items: stretch;
  }

  .two-columns,
  .traditional-config-shell,
  .route-config-grid,
  .preprocess-workflow-grid,
  .preview-dialog-grid {
    grid-template-columns: 1fr;
  }

  .form-span-2 {
    grid-column: auto;
  }

  .switch-card {
    flex-direction: column;
    align-items: flex-start;
  }

  .calibration-input-row {
    align-items: flex-start;
    flex-direction: column;
  }

  .upload-status-bar {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
