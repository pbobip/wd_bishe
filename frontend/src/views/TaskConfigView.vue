<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'

import { api } from '../api'
import ZoomableImageDialog from '../components/ZoomableImageDialog.vue'
import { useTaskWorkflow } from '../composables/useTaskWorkflow'

const router = useRouter()
const workflow = useTaskWorkflow()
const fileInputRef = ref<HTMLInputElement | null>(null)
const folderInputRef = ref<HTMLInputElement | null>(null)
const preprocessPreviewVisible = ref(false)
const preprocessPreviewLoading = ref(false)
const preprocessPreviewIndex = ref(0)
const preprocessPreviewRequestSeq = ref(0)
const zoomPreviewVisible = ref(false)
const zoomPreviewTitle = ref('图像预览')
const zoomPreviewSubtitle = ref('')
const zoomPreviewImageUrl = ref<string | null>(null)
const preprocessPreview = ref<{
  source_label: string
  footer_detected: boolean
  original_preview_url: string
  processed_preview_url: string
  message: string
  file_name: string
} | null>(null)

type ImportedImagePreviewEntry = {
  key?: string
  file?: File
  originalUrl?: string | null
  previewUrl?: string | null
  previewable?: boolean
  fileName: string
  relativePath?: string | null
}

const importedPreviewLoadingKey = ref<string | null>(null)

const previewFileOptions = computed(() =>
  workflow.selectedFiles.value.map((file, index) => ({
    label: `${index + 1}. ${file.name}`,
    value: index,
  })),
)
const currentPreviewFile = computed(() => workflow.selectedFiles.value[preprocessPreviewIndex.value] ?? null)
const preprocessPreviewReady = computed(
  () =>
    Boolean(
      workflow.selectedFiles.value.length
      && workflow.form.preprocess.enabled
      && workflow.preprocessSelectionValid.value,
    ),
)
const preprocessPreviewHint = computed(() => {
  if (!workflow.selectedFiles.value.length) {
    return '请先导入至少一张图像，再查看预处理预览。'
  }
  if (!workflow.form.preprocess.enabled) {
    return '启用预处理后，才会按当前流程生成预览图。'
  }
  if (!workflow.preprocessSelectionValid.value) {
    return workflow.preprocessValidationMessage.value || '请先补齐至少一种有效的预处理步骤。'
  }
  return '预览会基于当前本地选中的图像即时生成，不依赖上传登记是否完成。'
})
const syncFiles = async (event: Event, source: 'files' | 'folder') => {
  const target = event.target as HTMLInputElement
  const files = target.files ? Array.from(target.files) : []
  await workflow.setSelectedFiles(files, source)
  target.value = ''
}

const startProcessing = async () => {
  const success = await workflow.startTaskFromCreatePage()
  if (success && workflow.activeRunId.value) {
    router.push(`/runs/${workflow.activeRunId.value}`)
  }
}

const openResultWorkspace = () => {
  if (!workflow.activeRunId.value) return
  router.push(`/runs/${workflow.activeRunId.value}`)
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
  if (!preprocessPreviewReady.value) {
    ElMessage.warning(preprocessPreviewHint.value)
    return
  }

  if (preprocessPreviewIndex.value >= workflow.selectedFiles.value.length) {
    preprocessPreviewIndex.value = 0
  }

  await loadPreprocessPreview(preprocessPreviewIndex.value)
  if (preprocessPreview.value) {
    preprocessPreviewVisible.value = true
  }
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

const openZoomPreview = (variant: 'original' | 'processed') => {
  if (!preprocessPreview.value) return

  const fileName = preprocessPreview.value.file_name
  if (variant === 'original') {
    zoomPreviewTitle.value = `${fileName} · 分析区域`
    zoomPreviewSubtitle.value = preprocessPreview.value.footer_detected ? '已自动裁出分析区域' : '未检测到底栏，使用原图'
    zoomPreviewImageUrl.value = preprocessPreview.value.original_preview_url
  } else {
    zoomPreviewTitle.value = `${fileName} · 预处理后图像`
    zoomPreviewSubtitle.value = '按当前预处理配置生成'
    zoomPreviewImageUrl.value = preprocessPreview.value.processed_preview_url
  }
  zoomPreviewVisible.value = true
}

const ensureImportedImagePreviewUrl = async (entry: ImportedImagePreviewEntry) => {
  if (entry.previewable) {
    return entry.originalUrl || entry.previewUrl || null
  }

  if (entry.originalUrl) {
    return entry.originalUrl
  }

  if (!entry.file || !entry.key) {
    return entry.previewUrl || null
  }

  importedPreviewLoadingKey.value = entry.key
  try {
    const payload = new FormData()
    payload.append('file', entry.file)
    payload.append('preprocess', JSON.stringify({ ...workflow.form.preprocess, enabled: false }))
    payload.append('auto_crop_sem_region', 'false')
    const { data } = await api.post('/preprocess/preview', payload, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    if (typeof data?.original_preview_url === 'string' && data.original_preview_url) {
      entry.originalUrl = data.original_preview_url
      return entry.originalUrl
    }
    return entry.previewUrl || null
  } catch (error: any) {
    ElMessage.error(error?.response?.data?.detail ?? error?.message ?? '加载原图预览失败')
    return entry.previewUrl || null
  } finally {
    if (importedPreviewLoadingKey.value === entry.key) {
      importedPreviewLoadingKey.value = null
    }
  }
}

const openImportedImagePreview = async (entry: ImportedImagePreviewEntry) => {
  const sourceUrl = await ensureImportedImagePreviewUrl(entry)
  if (!sourceUrl) return
  zoomPreviewTitle.value = entry.fileName
  zoomPreviewSubtitle.value = entry.previewable
    ? (entry.relativePath?.trim() || '导入图像预览')
    : '原始 TIF 由后端转为高分辨率 PNG 预览展示'
  zoomPreviewImageUrl.value = sourceUrl
  zoomPreviewVisible.value = true
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
  workflow.initialize().catch(() => {
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
        </div>
        <div class="head-actions">
          <el-button
            class="jump-nav-button"
            type="primary"
            size="large"
            :disabled="!workflow.workflowReady.value"
            :loading="workflow.actionLoading.value"
            @click="startProcessing"
          >
            开始处理
          </el-button>
          <el-button
            v-if="workflow.canOpenResultWorkspace.value"
            class="jump-nav-button jump-nav-button--plain"
            plain
            size="large"
            @click="openResultWorkspace"
          >
            查看结果与后处理
          </el-button>
        </div>
      </div>

      <el-form label-position="top" class="task-create-form">
        <section class="form-section">
          <div class="section-inline-head">
            <h3>基础信息与导入</h3>
          </div>

          <div class="form-grid two-columns">
            <el-form-item class="form-span-2">
              <el-input v-model="workflow.form.name" />
            </el-form-item>

            <el-form-item class="form-span-2">
              <div class="upload-intake-shell">
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
                  <div class="upload-status-copy">
                    <strong class="upload-status-title">导入状态</strong>
                    <div class="upload-tip">
                      <span class="upload-count-tag">已选 {{ workflow.selectedFiles.value.length }} 张图像</span>
                      <span class="upload-mode-tag">{{ workflow.inputModeLabel }}</span>
                    </div>
                  </div>
                  <el-button class="upload-clear-button" plain type="danger" @click="workflow.clearSelectedFiles()">
                    清空已选
                  </el-button>
                </div>
              </div>
            </el-form-item>

            <div class="selection-card form-span-2">
              <div class="selection-head">
                <strong>本次导入图像与逐图标定</strong>
                <span>{{ workflow.selectedFiles.value.length ? workflow.calibrationReminder : '尚未选择图像' }}</span>
              </div>
              <el-empty v-if="!workflow.selectedFiles.value.length" description="支持单张、多张或整文件夹导入" />
              <div v-else class="selection-list">
                <div
                  v-for="entry in workflow.selectedImageEntries.value"
                  :key="entry.key"
                  class="selection-item selection-item--image"
                  :class="{ 'selection-item--pixel-mode': workflow.isImageCalibrationPixelMode(entry) }"
                >
                  <div class="selection-item__preview">
                    <button
                      v-if="entry.previewUrl"
                      type="button"
                      class="selection-item__preview-button"
                      :title="`查看 ${entry.fileName} 大图`"
                      :disabled="importedPreviewLoadingKey === entry.key"
                      :class="{ 'is-loading': importedPreviewLoadingKey === entry.key }"
                      @click="openImportedImagePreview(entry)"
                    >
                      <img :src="entry.previewUrl" :alt="entry.fileName" class="selection-item__thumb" />
                    </button>
                    <div v-else class="selection-item__placeholder">SEM</div>
                  </div>
                  <div class="selection-item__body">
                    <div class="selection-item__head">
                      <div class="selection-item__title">
                        <strong>{{ entry.fileName }}</strong>
                        <span v-if="workflow.shouldShowImageRelativePath(entry)">{{ entry.relativePath }}</span>
                      </div>
                      <span
                        class="selection-item__status"
                        :class="`selection-item__status--${workflow.getImageCalibrationStatusTone(entry)}`"
                      >
                        {{ workflow.getImageCalibrationStatusLabel(entry) }}
                      </span>
                    </div>

                    <div v-if="workflow.getImageCalibrationMeta(entry).length" class="selection-item__meta">
                      <span v-for="item in workflow.getImageCalibrationMeta(entry)" :key="item" class="selection-item__meta-chip">
                        {{ item }}
                      </span>
                    </div>

                    <p class="selection-item__detail">{{ workflow.getImageCalibrationDetail(entry) }}</p>

                    <div class="selection-item__actions">
                      <el-input-number
                        :model-value="entry.umPerPx ?? undefined"
                        :min="0.0001"
                        :step="0.001"
                        :precision="6"
                        placeholder="手动填写 um/px"
                        @update:model-value="(value: number | null | undefined) => workflow.updateImageCalibrationValue(entry.key, value)"
                      />
                      <span class="selection-item__unit">um / px</span>
                      <el-button
                        plain
                        :loading="entry.loading"
                        @click="workflow.runImageCalibrationProbe(entry.key, true)"
                      >
                        重新识别
                      </el-button>
                      <el-button
                        :type="workflow.isImageCalibrationPixelMode(entry) ? 'primary' : undefined"
                        :plain="!workflow.isImageCalibrationPixelMode(entry)"
                        :class="{ 'selection-item__pixel-button': workflow.isImageCalibrationPixelMode(entry) }"
                        @click="workflow.clearImageCalibrationValue(entry.key)"
                      >
                        {{ workflow.isImageCalibrationPixelMode(entry) ? '已按像素统计' : '按像素统计' }}
                      </el-button>
                    </div>

                    <p v-if="entry.error" class="selection-item__warning">{{ entry.error }}</p>
                  </div>
                </div>
              </div>
            </div>

            <div
              v-if="workflow.activeRunId.value || (workflow.selectedFiles.value.length && workflow.launchPhase.value !== 'idle')"
              class="upload-progress-card form-span-2"
            >
              <div class="upload-progress-card__head">
                <div>
                  <strong>任务进度</strong>
                  <p>{{ workflow.launchMessage.value }}</p>
                </div>
                <span>{{ workflow.activePayload.value ? workflow.activeStepProgressText.value : workflow.uploadProgressLabel.value }}</span>
              </div>
              <el-progress :percentage="workflow.executionProgress.value" />
              <small>
                <template v-if="workflow.activePayload.value">
                  当前任务状态：{{ workflow.runStatusLabel.value }}
                </template>
                <template v-else>
                  当前共 {{ Math.max(workflow.uploadState.totalChunks, workflow.selectedFiles.value.length ? Math.ceil(workflow.selectedFiles.value.length / 10) : 0) }} 批，
                  已上传 {{ workflow.uploadState.uploadedBytes }} / {{ workflow.uploadState.totalBytes || 0 }} bytes。
                </template>
              </small>
            </div>
          </div>
        </section>

        <section class="form-section">
          <div class="section-inline-head">
            <h3>分割方法</h3>
          </div>

          <div class="mode-selector">
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
                  </div>
                  <div class="traditional-config-grid traditional-config-grid--compact">
                    <div class="traditional-config-cell">
                      <el-form-item label="边界图算子">
                        <el-select v-model="workflow.form.traditional_seg.edge_operator">
                          <el-option label="Canny" value="canny" />
                          <el-option label="Sobel" value="sobel" />
                          <el-option label="Laplacian" value="laplacian" />
                        </el-select>
                      </el-form-item>
                    </div>

                    <div
                      v-if="['threshold', 'adaptive'].includes(workflow.form.traditional_seg.method)"
                      class="traditional-config-cell"
                    >
                      <el-form-item label="前景目标">
                        <el-select v-model="workflow.form.traditional_seg.foreground_target">
                          <el-option label="暗相 / γ′ 颗粒" value="dark" />
                          <el-option label="亮相 / 通道区域" value="bright" />
                        </el-select>
                      </el-form-item>
                    </div>

                    <div
                      v-else-if="workflow.form.traditional_seg.method === 'clustering'"
                      class="traditional-config-cell"
                    >
                      <el-form-item label="目标簇选择">
                        <el-select v-model="workflow.form.traditional_seg.cluster_target">
                          <el-option label="暗相簇" value="dark" />
                          <el-option label="亮相簇" value="bright" />
                          <el-option label="最大簇" value="largest" />
                        </el-select>
                      </el-form-item>
                    </div>
                  </div>
                </section>

                <section class="traditional-config-card route-config-card">
                  <div class="traditional-config-card__head">
                    <strong>路线专属参数</strong>
                  </div>

                  <Transition name="route-panel" mode="out-in">
                    <div :key="workflow.form.traditional_seg.method" class="route-config-surface route-config-surface--traditional">
                      <div
                        v-if="workflow.form.traditional_seg.method === 'threshold'"
                        class="route-config-grid route-config-grid--traditional route-config-grid--threshold"
                      >
                        <div class="route-config-slot">
                          <el-form-item label="阈值模式">
                            <el-select v-model="workflow.form.traditional_seg.threshold_mode">
                              <el-option label="Otsu" value="otsu" />
                              <el-option label="全局阈值" value="global" />
                              <el-option label="固定阈值" value="fixed" />
                            </el-select>
                          </el-form-item>
                        </div>

                        <div
                          v-if="['global', 'fixed'].includes(workflow.form.traditional_seg.threshold_mode)"
                          class="route-config-slot route-config-slot--compact"
                        >
                          <el-form-item v-if="workflow.form.traditional_seg.threshold_mode === 'global'" label="全局阈值">
                            <el-input-number v-model="workflow.form.traditional_seg.global_threshold" :min="1" :max="254" />
                          </el-form-item>

                          <el-form-item v-else-if="workflow.form.traditional_seg.threshold_mode === 'fixed'" label="固定阈值">
                            <el-input-number v-model="workflow.form.traditional_seg.fixed_threshold" :min="1" :max="254" />
                          </el-form-item>
                        </div>
                      </div>

                      <div
                        v-else-if="workflow.form.traditional_seg.method === 'adaptive'"
                        class="route-config-grid route-config-grid--traditional route-config-grid--adaptive"
                      >
                        <div class="route-config-slot route-config-slot--compact route-config-slot--adaptive-mode">
                          <el-form-item label="自适应模式">
                            <el-select v-model="workflow.form.traditional_seg.adaptive_method">
                              <el-option label="Gaussian" value="gaussian" />
                              <el-option label="Mean" value="mean" />
                            </el-select>
                          </el-form-item>
                        </div>

                        <div class="route-config-slot route-config-slot--compact">
                          <el-form-item label="局部窗口大小">
                            <el-input-number v-model="workflow.form.traditional_seg.adaptive_block_size" :min="3" :step="2" />
                          </el-form-item>
                        </div>

                        <div class="route-config-slot route-config-slot--compact">
                          <el-form-item label="自适应偏移 C">
                            <el-input-number v-model="workflow.form.traditional_seg.adaptive_c" :min="-50" :max="50" :step="1" />
                          </el-form-item>
                        </div>
                      </div>

                      <div
                        v-else-if="workflow.form.traditional_seg.method === 'edge'"
                        class="route-config-grid route-config-grid--traditional route-config-grid--edge"
                      >
                        <div class="route-config-slot route-config-slot--compact">
                          <el-form-item label="边缘预模糊核">
                            <el-input-number v-model="workflow.form.traditional_seg.edge_blur_kernel" :min="1" :step="2" />
                          </el-form-item>
                        </div>

                        <div class="route-config-slot route-config-slot--compact">
                          <el-form-item label="边缘膨胀次数">
                            <el-input-number v-model="workflow.form.traditional_seg.edge_dilate_iterations" :min="0" :step="1" />
                          </el-form-item>
                        </div>

                        <div class="route-config-slot route-config-slot--compact">
                          <template v-if="workflow.form.traditional_seg.edge_operator === 'canny'">
                            <el-form-item label="Canny 下阈值">
                              <el-input-number v-model="workflow.form.traditional_seg.edge_threshold1" :min="0" :max="255" />
                            </el-form-item>
                          </template>
                          <div v-else class="route-static-note route-static-note--compact">
                            <strong>{{ workflow.form.traditional_seg.edge_operator === 'sobel' ? 'Sobel' : 'Laplacian' }} 算子</strong>
                            <p>当前路线会先计算边缘强度，再自动估计二值阈值。</p>
                          </div>
                        </div>

                        <div
                          v-if="workflow.form.traditional_seg.edge_operator === 'canny'"
                          class="route-config-slot route-config-slot--compact"
                        >
                          <el-form-item label="Canny 上阈值">
                            <el-input-number v-model="workflow.form.traditional_seg.edge_threshold2" :min="0" :max="255" />
                          </el-form-item>
                        </div>
                      </div>

                      <div v-else class="route-config-grid route-config-grid--traditional">
                        <div class="route-config-slot">
                          <el-form-item label="聚类数">
                            <el-input-number v-model="workflow.form.traditional_seg.kmeans_clusters" :min="2" :max="6" :step="1" />
                          </el-form-item>
                        </div>

                        <div class="route-config-slot">
                          <el-form-item label="尝试次数">
                            <el-input-number v-model="workflow.form.traditional_seg.kmeans_attempts" :min="1" :max="20" :step="1" />
                          </el-form-item>
                        </div>
                      </div>
                    </div>
                  </Transition>
                </section>
              </div>
            </template>

            <template v-if="workflow.form.segmentation_mode !== 'traditional'">
              <el-form-item label="模型槽位">
                <el-select v-model="workflow.form.dl_model.model_slot">
                  <el-option label="MBU-Net++ 主模型" value="mbu_netpp" />
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
            </template>
          </div>
        </section>

        <section class="form-section">
          <div class="form-grid two-columns">
            <div class="switch-card form-span-2">
              <div>
                <strong>预处理开关</strong>
                <p>{{ workflow.preprocessSummary }}</p>
              </div>
              <el-switch v-model="workflow.form.preprocess.enabled" active-text="启用" inactive-text="跳过" />
            </div>

            <template v-if="workflow.form.preprocess.enabled">
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
                </div>
                <el-button
                  type="primary"
                  plain
                  :disabled="!preprocessPreviewReady"
                  :loading="preprocessPreviewLoading"
                  @click="openPreprocessPreview"
                >
                  {{ preprocessPreviewVisible ? '刷新预处理预览' : '查看预处理预览' }}
                </el-button>
              </div>

              <Transition name="route-panel" mode="out-in">
                <div v-if="preprocessPreviewVisible" class="preprocess-preview-panel form-span-2">
                  <div class="preprocess-preview-panel__head">
                    <div>
                      <strong>{{ preprocessPreview?.file_name ?? currentPreviewFile?.name ?? '图像预览' }}</strong>
                      <p>{{ preprocessPreview?.message ?? '支持逐张切换查看全部已选图像，不会一次性生成整批预处理图。' }}</p>
                    </div>
                    <el-button plain @click="preprocessPreviewVisible = false">收起预览</el-button>
                  </div>

                  <div v-if="workflow.selectedFiles.value.length > 1" class="preview-dialog-toolbar">
                    <el-select
                      :model-value="preprocessPreviewIndex"
                      class="preview-dialog-select"
                      filterable
                      placeholder="选择要预览的图像"
                      @change="(value: string | number) => changePreviewFile(Number(value))"
                    >
                      <el-option
                        v-for="option in previewFileOptions"
                        :key="option.value"
                        :label="option.label"
                        :value="option.value"
                      />
                    </el-select>
                    <div class="preview-dialog-actions">
                      <el-button plain :disabled="preprocessPreviewLoading || preprocessPreviewIndex <= 0" @click="shiftPreviewFile(-1)">
                        上一张
                      </el-button>
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
                  </div>

                  <el-skeleton :loading="preprocessPreviewLoading && !preprocessPreview" animated>
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
                        </div>
                        <div class="preview-panel__media">
                          <button type="button" class="preview-panel__image-button" @click="openZoomPreview('original')">
                            <img :src="preprocessPreview.original_preview_url" alt="预处理前图像" class="preview-panel__image" />
                            <span class="preview-panel__image-hint">点击放大</span>
                          </button>
                        </div>
                      </div>
                      <div class="preview-panel">
                        <div class="preview-panel__head">
                          <strong>预处理后图像</strong>
                        </div>
                        <div class="preview-panel__media">
                          <button type="button" class="preview-panel__image-button" @click="openZoomPreview('processed')">
                            <img :src="preprocessPreview.processed_preview_url" alt="预处理后图像" class="preview-panel__image" />
                            <span class="preview-panel__image-hint">点击放大</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </el-skeleton>
                </div>
              </Transition>

              <p v-if="workflow.preprocessValidationMessage.value" class="field-validation form-span-2">
                {{ workflow.preprocessValidationMessage.value }}
              </p>
            </template>
          </div>
        </section>

      </el-form>
    </section>

    <ZoomableImageDialog
      v-model="zoomPreviewVisible"
      :image-url="zoomPreviewImageUrl"
      :image-alt="zoomPreviewTitle"
      :title="zoomPreviewTitle"
      :subtitle="zoomPreviewSubtitle"
    />
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
  min-width: 156px;
  justify-content: center;
  border-radius: 14px;
  padding-inline: 18px;
  font-weight: 700;
}

.head-actions :deep(.el-button + .el-button) {
  margin-left: 0;
}

.jump-nav-button {
  box-shadow: 0 12px 22px rgba(23, 96, 135, 0.18);
}

.jump-nav-button:disabled,
.jump-nav-button.is-disabled {
  box-shadow: 0 10px 24px rgba(21, 40, 62, 0.06);
}

.task-create-form {
  display: grid;
  gap: 16px;
}

.form-section {
  padding: 20px;
  border-radius: 22px;
  background: rgba(255, 252, 247, 0.62);
  border: 1px solid rgba(31, 40, 48, 0.07);
}

.section-inline-head {
  margin-bottom: 16px;
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
  margin: 0 0 6px;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 0.01em;
}

.section-inline-head span {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
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

.upload-intake-shell {
  display: grid;
  gap: 12px;
  width: 100%;
}

.upload-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  width: 100%;
}

.upload-status-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(31, 40, 48, 0.07);
  box-shadow: 0 8px 18px rgba(44, 32, 20, 0.04);
}

.upload-status-copy {
  display: grid;
  gap: 8px;
  min-width: 0;
}

.upload-status-title {
  color: var(--ink);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.01em;
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
  background: rgba(31, 40, 48, 0.06);
  color: var(--ink);
  font-size: 13px;
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
  font-size: 13px;
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
  padding: 16px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(31, 40, 48, 0.07);
  box-shadow: 0 6px 16px rgba(44, 32, 20, 0.04);
}

.selection-item--image {
  display: grid;
  grid-template-columns: 108px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.selection-item--pixel-mode {
  border-color: rgba(23, 96, 135, 0.16);
  box-shadow:
    0 10px 24px rgba(44, 32, 20, 0.05),
    inset 0 0 0 1px rgba(23, 96, 135, 0.06);
}

.selection-item__preview {
  width: 108px;
  height: 108px;
  overflow: hidden;
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: linear-gradient(145deg, rgba(248, 250, 252, 0.94), rgba(235, 240, 245, 0.86));
}

.selection-item__preview-button {
  width: 100%;
  height: 100%;
  padding: 0;
  border: none;
  border-radius: inherit;
  background: transparent;
  cursor: zoom-in;
  overflow: hidden;
}

.selection-item__preview-button.is-loading {
  cursor: progress;
  opacity: 0.76;
}

.selection-item__preview-button:focus-visible {
  outline: 2px solid rgba(23, 96, 135, 0.35);
  outline-offset: -2px;
}

.selection-item__thumb,
.selection-item__placeholder {
  width: 100%;
  height: 100%;
}

.selection-item__thumb {
  display: block;
  object-fit: cover;
  transition: transform 0.22s ease;
}

.selection-item__preview-button:hover .selection-item__thumb {
  transform: scale(1.04);
}

.selection-item__placeholder {
  display: grid;
  place-items: center;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.selection-item__body {
  display: grid;
  gap: 10px;
  min-width: 0;
}

.selection-item__head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.selection-item__title {
  display: grid;
  gap: 4px;
  min-width: 0;
}

.selection-item__title strong {
  font-size: 14px;
  word-break: break-all;
}

.selection-item__title span {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
  word-break: break-all;
}

.selection-item__status {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 0 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.selection-item__status--success {
  background: rgba(33, 150, 83, 0.1);
  color: #1c8b4b;
}

.selection-item__status--warning {
  background: rgba(214, 141, 41, 0.12);
  color: #b26b14;
}

.selection-item__status--pending {
  background: rgba(23, 96, 135, 0.1);
  color: var(--accent);
}

.selection-item__status--neutral {
  background: rgba(31, 40, 48, 0.06);
  color: var(--muted);
}

.selection-item__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.selection-item__meta-chip {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 0 10px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 12px;
  font-weight: 600;
}

.selection-item__detail {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.6;
}

.selection-item__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}

.selection-item__unit {
  color: var(--muted);
  font-size: 13px;
  font-weight: 600;
}

.selection-item__warning {
  margin: 0;
  color: #b26b14;
  font-size: 12px;
  line-height: 1.6;
}

:deep(.selection-item__actions .el-input-number) {
  width: 220px;
}

:deep(.selection-item__pixel-button.el-button--primary) {
  border-color: rgba(23, 96, 135, 0.14);
  background: linear-gradient(135deg, #3f95e9, #176087);
  box-shadow: 0 8px 18px rgba(23, 96, 135, 0.18);
}

.upload-progress-card {
  display: grid;
  gap: 12px;
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(31, 40, 48, 0.07);
  box-shadow: 0 8px 18px rgba(44, 32, 20, 0.04);
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
  font-size: 16px;
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
  display: grid;
  grid-template-columns: minmax(0, 332px) max-content;
  align-items: center;
  gap: 12px 14px;
}

.calibration-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}

.calibration-unit {
  color: var(--muted);
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
}

:deep(.calibration-input-row .el-input-number) {
  width: 100%;
}

:deep(.calibration-action-button.el-button) {
  min-height: 38px;
  padding-inline: 16px;
  border-radius: 12px;
}

.field-helper {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(31, 40, 48, 0.07);
  color: var(--muted);
  font-size: 13px;
  line-height: 1.6;
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
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(31, 40, 48, 0.07);
  box-shadow: 0 8px 18px rgba(44, 32, 20, 0.04);
}

.switch-card strong {
  display: block;
  margin-bottom: 4px;
  font-size: 15px;
  font-weight: 700;
}

.switch-card p {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.6;
}

.mode-selector {
  display: flex;
  flex-direction: column;
  gap: 12px;
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
  padding: 18px 20px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(31, 40, 48, 0.07);
  box-shadow: 0 8px 18px rgba(44, 32, 20, 0.04);
}

.preprocess-preview-entry strong {
  display: block;
  margin-bottom: 4px;
  font-size: 16px;
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

.preprocess-preview-entry__note--warning {
  color: #b2562d !important;
}

.preprocess-preview-panel {
  display: grid;
  gap: 16px;
  padding: 18px 20px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(31, 40, 48, 0.07);
  box-shadow: 0 10px 24px rgba(44, 32, 20, 0.05);
}

.preprocess-preview-panel__head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.preprocess-preview-panel__head strong {
  display: block;
  margin-bottom: 4px;
  font-size: 16px;
}

.preprocess-preview-panel__head p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.traditional-method-panel {
  display: grid;
  gap: 12px;
  width: 100%;
  min-width: 0;
  flex: 1 1 auto;
}

.traditional-config-shell {
  display: grid;
  grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.traditional-config-card {
  display: grid;
  gap: 12px;
  align-content: start;
  padding: 18px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(31, 40, 48, 0.07);
  box-shadow: 0 10px 24px rgba(44, 32, 20, 0.05);
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
  font-size: 16px;
  letter-spacing: 0.01em;
}

.traditional-config-card__head span {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.5;
}

.traditional-config-grid,
.route-config-grid {
  display: grid;
  gap: 14px 16px;
}

.traditional-config-grid--compact {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px 14px;
  align-items: start;
}

.traditional-config-cell {
  display: grid;
  min-height: 0;
  align-content: start;
}

.traditional-config-cell :deep(.el-form-item) {
  margin-bottom: 0;
}

.route-config-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-items: stretch;
}

.route-config-grid--traditional {
  grid-auto-rows: minmax(0, auto);
  align-content: start;
}

.route-config-surface {
  display: grid;
  align-content: start;
  min-height: 0;
}

.route-config-surface--traditional {
  min-height: 0;
}

.route-config-slot {
  display: grid;
  min-height: 0;
  align-content: start;
}

.route-config-slot--compact {
  min-height: 0;
}

.route-config-grid--adaptive {
  grid-template-columns: minmax(220px, 240px) 172px 172px;
  justify-content: start;
  gap: 12px 18px;
  align-items: start;
}

.route-config-grid--edge {
  grid-template-columns: repeat(4, minmax(0, 1fr));
  align-items: start;
}

.route-config-slot--adaptive-mode :deep(.el-select) {
  width: 100%;
  max-width: 240px;
}

.route-config-grid--adaptive :deep(.el-input-number) {
  width: 172px;
}

.route-config-slot--wide {
  grid-column: 1 / -1;
  min-height: auto;
}

.route-span-2 {
  grid-column: 1 / -1;
}

.route-config-slot :deep(.el-form-item) {
  height: 100%;
  margin-bottom: 0;
}

.route-static-note {
  display: grid;
  gap: 6px;
  align-content: start;
  padding: 16px 18px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.05), rgba(255, 255, 255, 0.92));
  border: 1px solid rgba(23, 96, 135, 0.1);
}

.route-static-note--compact {
  min-height: 100%;
}

.route-static-note strong {
  font-size: 14px;
}

.route-static-note p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.traditional-method-hint {
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(31, 40, 48, 0.07);
  box-shadow: 0 8px 18px rgba(44, 32, 20, 0.04);
}

.traditional-method-hint strong {
  display: block;
  margin-bottom: 6px;
  font-size: 16px;
}

.traditional-method-hint p {
  margin: 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
}

:deep(.mode-segmented) {
  width: 100%;
  padding: 6px;
  background: rgba(248, 244, 238, 0.96);
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

:deep(.mode-segmented .el-segmented__item) {
  min-height: 46px;
  font-size: 15px;
  font-weight: 600;
  color: rgba(31, 40, 48, 0.72);
}

:deep(.mode-segmented .el-segmented__item-selected) {
  color: #ffffff;
}

:deep(.mode-segmented .el-segmented__thumb) {
  background: linear-gradient(135deg, #28556f, #176087);
  border-radius: 16px;
  box-shadow: 0 10px 22px rgba(23, 96, 135, 0.18);
}

:deep(.traditional-method-segmented) {
  width: 100%;
  padding: 6px;
  background: rgba(247, 242, 235, 0.96);
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
}

:deep(.traditional-method-segmented .el-segmented__group) {
  display: flex;
  width: 100%;
}

:deep(.traditional-method-segmented .el-segmented__item) {
  flex: 1 1 0;
  min-width: 0;
  min-height: 40px;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
  color: rgba(31, 40, 48, 0.72);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

:deep(.traditional-method-segmented .el-segmented__item-selected) {
  background: linear-gradient(135deg, #4d9cf6, #409eff);
  border: 1px solid rgba(37, 115, 195, 0.16);
  border-radius: 14px;
  box-shadow: 0 8px 18px rgba(64, 158, 255, 0.18);
}

:deep(.traditional-method-segmented .el-segmented__item.is-selected) {
  color: var(--ink);
}

:deep(.preprocess-segmented) {
  width: 100%;
  padding: 6px;
  background: rgba(247, 242, 235, 0.96);
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
}

:deep(.preprocess-segmented .el-segmented__group) {
  flex-wrap: wrap;
}

:deep(.preprocess-segmented .el-segmented__item) {
  min-height: 40px;
  font-size: 14px;
  font-weight: 600;
  color: rgba(31, 40, 48, 0.72);
}

:deep(.preprocess-segmented .el-segmented__item-selected) {
  color: var(--ink);
}

:deep(.preprocess-segmented .el-segmented__thumb) {
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid rgba(31, 40, 48, 0.08);
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(44, 32, 20, 0.07);
}

.task-create-layout :deep(.el-form-item__label) {
  padding-bottom: 8px;
  font-size: 13px;
  font-weight: 600;
  color: rgba(31, 40, 48, 0.82);
  line-height: 1.4;
}

.task-create-layout :deep(.el-input__wrapper),
.task-create-layout :deep(.el-select__wrapper),
.task-create-layout :deep(.el-textarea__wrapper),
.task-create-layout :deep(.el-input-number),
.task-create-layout :deep(.el-input-number__decrease),
.task-create-layout :deep(.el-input-number__increase) {
  border-radius: 14px;
}

.task-create-layout :deep(.el-input__wrapper),
.task-create-layout :deep(.el-select__wrapper),
.task-create-layout :deep(.el-textarea__wrapper) {
  min-height: 44px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: inset 0 0 0 1px rgba(31, 40, 48, 0.08);
}

.task-create-layout :deep(.el-input__inner),
.task-create-layout :deep(.el-select__selected-item),
.task-create-layout :deep(.el-textarea__inner) {
  font-size: 14px;
  color: var(--ink);
}

.task-create-layout :deep(.el-input__wrapper.is-focus),
.task-create-layout :deep(.el-select__wrapper.is-focused),
.task-create-layout :deep(.el-textarea__wrapper.is-focus) {
  box-shadow:
    inset 0 0 0 1px rgba(23, 96, 135, 0.24),
    0 0 0 4px rgba(23, 96, 135, 0.08);
}

.task-create-layout :deep(.el-input-number) {
  overflow: hidden;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: inset 0 0 0 1px rgba(31, 40, 48, 0.08);
}

.task-create-layout :deep(.el-input-number .el-input__wrapper) {
  box-shadow: none;
  background: transparent;
}

.task-create-layout :deep(.el-input-number__decrease),
.task-create-layout :deep(.el-input-number__increase) {
  background: rgba(247, 242, 235, 0.96);
  color: rgba(31, 40, 48, 0.72);
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

.preview-dialog-actions {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
  flex-wrap: wrap;
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

.preview-panel__media {
  display: grid;
  align-items: center;
  min-height: 320px;
  aspect-ratio: 4 / 3;
  border-radius: 14px;
  overflow: hidden;
  background: rgba(18, 20, 24, 0.04);
}

.preview-panel__image-button {
  position: relative;
  width: 100%;
  height: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: zoom-in;
  overflow: hidden;
}

.preview-panel__image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.preview-panel__image-hint {
  position: absolute;
  right: 14px;
  bottom: 14px;
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 0 12px;
  border-radius: 999px;
  background: rgba(18, 20, 24, 0.58);
  color: rgba(255, 255, 255, 0.92);
  font-size: 12px;
  font-weight: 700;
  backdrop-filter: blur(6px);
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
  .preprocess-preview-entry,
  .preprocess-preview-panel__head,
  .upload-progress-card__head,
  .preview-dialog-toolbar,
  .preview-dialog-actions {
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

  .selection-item--image {
    grid-template-columns: 1fr;
  }

  .selection-item__preview {
    width: 100%;
    height: 180px;
  }

  .selection-item__head {
    flex-direction: column;
  }

  .calibration-input-row {
    grid-template-columns: 1fr;
    align-items: flex-start;
  }

  .calibration-actions {
    width: 100%;
  }

  .upload-status-bar {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
