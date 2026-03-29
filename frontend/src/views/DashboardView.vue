<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '../api'
import type { ModelRunner, Project } from '../types'
import {
  buildUmPerPxFromScaleBar,
  inspectCalibrationProbe,
  type CalibrationProbe,
} from '../utils/calibration'

const router = useRouter()
const route = useRoute()
const SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
const UPLOAD_CHUNK_SIZE = 10

const projects = ref<Project[]>([])
const runners = ref<ModelRunner[]>([])
const selectedFiles = ref<File[]>([])
const fileInputRef = ref<HTMLInputElement | null>(null)
const folderInputRef = ref<HTMLInputElement | null>(null)
const loading = ref(false)
const calibrationProbe = ref<CalibrationProbe | null>(null)
const calibrationProbeLoading = ref(false)
const calibrationDialogVisible = ref(false)
const calibrationScaleBarUm = ref<number | null>(null)
const calibrationScaleBarPixels = ref<number | null>(null)
let calibrationProbeSeq = 0
const selectionSource = ref<'files' | 'folder'>('files')
const segmentationOptions = [
  { label: '传统分割', value: 'traditional' },
  { label: '深度学习', value: 'dl' },
  { label: '结果对比', value: 'compare' },
]

const form = reactive({
  project_id: 1,
  name: `任务-${new Date().toLocaleString()}`,
  input_mode: 'batch',
  segmentation_mode: 'traditional',
  input_config: {
    result_dir_name: 'default',
    um_per_px: undefined as number | undefined,
    auto_crop_sem_region: true,
    save_sem_footer: true,
  },
  preprocess: {
    enabled: false,
    operations: [] as string[],
    wavelet_strength: 0.12,
    mean_kernel: 3,
    gaussian_kernel: 3,
    median_kernel: 3,
    apply_hist_equalization: false,
  },
  traditional_seg: {
    threshold_mode: 'otsu',
    global_threshold: 120,
    fixed_threshold: 120,
    edge_operator: 'canny',
    min_area: 30,
    remove_border: true,
    open_kernel: 3,
    close_kernel: 3,
  },
  dl_model: {
    model_slot: 'custom',
    runner_id: undefined as number | undefined,
    weight_path: '',
    input_size: 1024,
    threshold: 0.3,
    device: 'auto',
    extra_params: {},
  },
  compare: {
    enabled: false,
  },
  stats: {
    enabled: true,
    export_csv: true,
    export_xlsx: true,
  },
  export: {
    include_masks: true,
    include_overlays: true,
    include_tables: true,
    include_charts: true,
    include_config_snapshot: true,
  },
})

const loadInitialData = async () => {
  const [projectRes, runnerRes] = await Promise.all([api.get<Project[]>('/projects'), api.get<ModelRunner[]>('/model-runners')])
  projects.value = projectRes.data
  runners.value = runnerRes.data
  if (projects.value[0]) {
    const queryProjectId = Number(route.query.project_id)
    const matchedProject = projects.value.find((project) => project.id === queryProjectId)
    form.project_id = matchedProject?.id ?? projects.value[0].id
  }
}

const segmentationHint = computed(() => {
  if (form.segmentation_mode === 'traditional') {
    return '适合特征明显、速度优先的场景，输出传统阈值分割、边界图和统计结果。'
  }
  if (form.segmentation_mode === 'dl') {
    return '调用深度学习模型接口做自动分割，适合后续接入 MatSAM、SAM LoRA 或 ResNeXt50。'
  }
  return '同时输出传统分割和深度学习结果，适合答辩展示双路线差异。'
})

const currentModeLabel = computed(
  () => segmentationOptions.find((item) => item.value === form.segmentation_mode)?.label ?? '传统分割',
)

const preprocessSummary = computed(() => {
  if (!form.preprocess.enabled) return '当前跳过预处理'
  if (!form.preprocess.operations.length) return '已启用预处理，等待选择具体流程'
  return `已启用：${form.preprocess.operations.join(' / ')}`
})

const calibrationNeedsWarning = computed(() => form.stats.enabled && !form.input_config.um_per_px)
const calibrationReminder = computed(() =>
  calibrationNeedsWarning.value
    ? '当前未填写 um_per_px，本次任务仍可运行，但所有统计将只输出 px / px²。'
    : `当前标定值：${form.input_config.um_per_px?.toFixed(4)} um/px`,
)
const calibrationProbeSummary = computed(() => {
  if (calibrationProbeLoading.value) {
    return '正在检测底栏标定信息...'
  }
  if (!calibrationProbe.value) {
    return '尚未检测到自动标定提示，仍可手动填写比例尺对应的 μm。'
  }
  const parts = [
    calibrationProbe.value.footer_detected ? '底栏已识别' : '未识别底栏',
    calibrationProbe.value.scale_bar_detected ? '比例尺已识别' : '未识别比例尺',
  ]
  if (typeof calibrationProbe.value.scale_bar_pixels === 'number') {
    parts.push(`比例尺像素 ${calibrationProbe.value.scale_bar_pixels.toFixed(0)} px`)
  }
  if (typeof calibrationProbe.value.ocr_scale_bar_um === 'number') {
    parts.push(`OCR 标尺 ${calibrationProbe.value.ocr_scale_bar_um} μm`)
  }
  return parts.join(' · ')
})
const calibrationComputedValue = computed(() =>
  buildUmPerPxFromScaleBar(calibrationScaleBarUm.value, calibrationScaleBarPixels.value),
)
const calibrationCandidates = computed(() => calibrationProbe.value?.common_scale_candidates ?? [])

const currentRunner = computed(() => {
  if (form.segmentation_mode === 'traditional') return null
  return runners.value.find((runner) => runner.slot === form.dl_model.model_slot) ?? null
})

const runnerNotes = computed(() => {
  if (form.segmentation_mode === 'traditional') {
    return ['传统分割当前不依赖外部模型运行器', '可直接输出掩码、边界和统计结果']
  }
  return [
    currentRunner.value ? `当前槽位：${currentRunner.value.display_name}` : '当前槽位尚未匹配到运行器',
    '深度学习或结果对比模式会调用对应环境执行推理',
  ]
})

const onFilesChange = (event: Event, source: 'files' | 'folder') => {
  const target = event.target as HTMLInputElement
  const files = target.files ? Array.from(target.files) : []
  const accepted = files.filter((file) => {
    const lower = file.name.toLowerCase()
    return SUPPORTED_EXTENSIONS.some((ext) => lower.endsWith(ext))
  })
  const skipped = files.length - accepted.length
  selectionSource.value = source
  selectedFiles.value = accepted
  form.input_mode = source === 'folder' || selectedFiles.value.length > 1 ? 'batch' : 'single'
  calibrationProbe.value = null
  calibrationScaleBarPixels.value = null
  calibrationScaleBarUm.value = null
  if (accepted.length) {
    void refreshCalibrationProbe(accepted)
  }
  if (skipped > 0) {
    ElMessage.warning(`已跳过 ${skipped} 个非图像文件，仅保留 png/jpg/jpeg/tif/tiff`)
  }
}

const refreshCalibrationProbe = async (files: File[]) => {
  const requestSeq = ++calibrationProbeSeq
  calibrationProbeLoading.value = true
  try {
    const probe = await inspectCalibrationProbe(files)
    if (requestSeq !== calibrationProbeSeq) return
    calibrationProbe.value = probe
    if (probe?.scale_bar_pixels && !calibrationScaleBarPixels.value) {
      calibrationScaleBarPixels.value = probe.scale_bar_pixels
    }
  } finally {
    if (requestSeq === calibrationProbeSeq) {
      calibrationProbeLoading.value = false
    }
  }
}

const openFilePicker = () => {
  fileInputRef.value?.click()
}

const openFolderPicker = () => {
  folderInputRef.value?.click()
}

const chunkFiles = <T,>(items: T[], size: number) => {
  const chunks: T[][] = []
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size))
  }
  return chunks
}

const uploadSelectedFiles = async (runId: number) => {
  const chunks = chunkFiles(selectedFiles.value, UPLOAD_CHUNK_SIZE)
  for (const [index, chunk] of chunks.entries()) {
    const multipart = new FormData()
    chunk.forEach((file) => multipart.append('files', file))
    await api.post(`/runs/${runId}/images`, multipart, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    if (chunks.length > 1) {
      ElMessage.success(`已上传第 ${index + 1}/${chunks.length} 批图像`)
    }
  }
}

const executeTask = async () => {
  if (!selectedFiles.value.length) {
    ElMessage.warning('请先选择图像')
    return
  }
  loading.value = true
  try {
    form.compare.enabled = form.segmentation_mode === 'compare'
    if (form.segmentation_mode !== 'traditional') {
      const matched = runners.value.find((runner) => runner.slot === form.dl_model.model_slot)
      form.dl_model.runner_id = matched?.id
    }
    const runRes = await api.post('/runs', form)
    const runId = runRes.data.id
    await uploadSelectedFiles(runId)
    await api.post(`/runs/${runId}/execute`)
    ElMessage.success('任务已创建，正在执行')
    await router.push(`/runs/${runId}`)
  } catch (error: any) {
    ElMessage.error(error?.response?.data?.detail ?? error?.message ?? '任务创建失败')
  } finally {
    loading.value = false
  }
}

const openCalibrationDialog = () => {
  calibrationDialogVisible.value = true
  calibrationScaleBarUm.value = calibrationProbe.value?.ocr_scale_bar_um ?? null
  calibrationScaleBarPixels.value = calibrationProbe.value?.scale_bar_pixels ?? null
}

const continueWithoutCalibration = async () => {
  calibrationDialogVisible.value = false
  await executeTask()
}

const applyCalibrationAndContinue = async () => {
  const value = calibrationComputedValue.value
  if (value === null) {
    ElMessage.warning('请先填写有效的比例尺 μm 数值和像素长度')
    return
  }
  form.input_config.um_per_px = value
  calibrationDialogVisible.value = false
  await executeTask()
}

const applySuggestedCalibration = async (value: number) => {
  form.input_config.um_per_px = value
  calibrationDialogVisible.value = false
  await executeTask()
}

const submitTask = async () => {
  if (!selectedFiles.value.length) {
    ElMessage.warning('请先选择图像')
    return
  }
  if (calibrationNeedsWarning.value) {
    openCalibrationDialog()
    return
  }
  await executeTask()
}

onMounted(() => {
  loadInitialData().catch(() => ElMessage.error('初始化数据加载失败'))
})
</script>

<template>
  <div class="dashboard-layout">
    <div class="dashboard-main-column">
      <section class="glass-card panel panel-form">
        <div class="panel-header panel-header-with-action">
          <div>
            <h2 class="section-title">新建任务</h2>
            <p class="section-subtitle">先选输入，再定模式，预处理单独开关控制</p>
          </div>
          <div class="panel-header-action">
            <el-button class="header-submit-button" type="primary" size="large" :loading="loading" @click="submitTask">
              创建并执行
            </el-button>
          </div>
        </div>

        <el-form label-position="top">
        <section class="form-section">
          <div class="section-inline-head">
            <h3>基础信息</h3>
            <span>任务名称、图像来源和模式选择</span>
          </div>

          <div class="form-grid two-columns">
            <el-form-item v-if="projects.length > 1" label="项目">
              <el-select v-model="form.project_id">
                <el-option v-for="project in projects" :key="project.id" :label="project.name" :value="project.id" />
              </el-select>
            </el-form-item>

            <el-form-item label="任务名称" class="form-span-2">
              <el-input v-model="form.name" />
            </el-form-item>

            <el-form-item label="图像选择" class="form-span-2">
              <div class="upload-actions">
                <el-button @click="openFilePicker">选择图片文件</el-button>
                <el-button plain @click="openFolderPicker">选择文件夹</el-button>
              </div>
              <input
                ref="fileInputRef"
                type="file"
                multiple
                accept=".png,.jpg,.jpeg,.tif,.tiff"
                class="hidden-input"
                @change="(event) => onFilesChange(event, 'files')"
              />
              <input
                ref="folderInputRef"
                type="file"
                multiple
                webkitdirectory
                directory
                accept=".png,.jpg,.jpeg,.tif,.tiff"
                class="hidden-input"
                @change="(event) => onFilesChange(event, 'folder')"
              />
              <div class="upload-tip">
                已选 {{ selectedFiles.length }} 张图像
                <span v-if="selectedFiles.length" class="upload-mode-tag">
                  {{ form.input_mode === 'single' ? '当前为单张输入' : `当前为批处理 · ${selectionSource === 'folder' ? '来自文件夹' : '来自多图选择'}` }}
                </span>
              </div>
            </el-form-item>

            <el-form-item label="分割模式" class="form-span-2">
              <div class="mode-selector">
                <el-segmented
                  v-model="form.segmentation_mode"
                  :options="segmentationOptions"
                  class="mode-segmented"
                />
                <div class="mode-hint-card">
                  <strong>{{ currentModeLabel }}</strong>
                  <p>{{ segmentationHint }}</p>
                </div>
              </div>
            </el-form-item>
          </div>
        </section>

        <section class="form-section">
          <div class="section-inline-head">
            <h3>统一配置</h3>
            <span>输出目录、比例尺和预处理开关</span>
          </div>

          <div class="form-grid two-columns">
            <el-form-item class="form-span-2" label="um_per_px">
              <el-input-number v-model="form.input_config.um_per_px" :min="0.0001" :step="0.001" :precision="4" />
              <div class="field-helper">
                <span>不填写时，面积/尺寸/通道宽度仅输出像素单位。</span>
                <span class="field-helper-hint">{{ calibrationProbeSummary }}</span>
                <span class="field-helper-status">{{ calibrationReminder }}</span>
              </div>
            </el-form-item>

            <div class="switch-card form-span-2">
              <div>
                <strong>预处理</strong>
                <p>{{ preprocessSummary }}</p>
              </div>
              <el-switch v-model="form.preprocess.enabled" active-text="启用" inactive-text="跳过" />
            </div>

            <el-form-item v-if="form.preprocess.enabled" label="预处理流程" class="form-span-2">
              <el-checkbox-group v-model="form.preprocess.operations" class="process-grid">
                <el-checkbox label="wavelet">改进小波阈值</el-checkbox>
                <el-checkbox label="mean">均值滤波</el-checkbox>
                <el-checkbox label="gaussian">高斯滤波</el-checkbox>
                <el-checkbox label="median">中值滤波</el-checkbox>
                <el-checkbox label="hist_equalization">直方图均衡化</el-checkbox>
              </el-checkbox-group>
            </el-form-item>
          </div>
        </section>

        <section class="form-section">
          <div class="section-inline-head">
            <h3>{{ form.segmentation_mode === 'traditional' ? '传统分割参数' : form.segmentation_mode === 'dl' ? '深度学习参数' : '分割参数' }}</h3>
            <span>{{ form.segmentation_mode === 'compare' ? '当前会同时保留两条路线的配置' : '按当前分割模式填写必要参数' }}</span>
          </div>

          <div class="form-grid two-columns">
            <template v-if="form.segmentation_mode !== 'dl'">
              <el-form-item label="阈值模式">
                <el-select v-model="form.traditional_seg.threshold_mode">
                  <el-option label="Otsu" value="otsu" />
                  <el-option label="全局阈值" value="global" />
                  <el-option label="固定阈值" value="fixed" />
                </el-select>
              </el-form-item>
              <el-form-item label="边界算子">
                <el-select v-model="form.traditional_seg.edge_operator">
                  <el-option label="Canny" value="canny" />
                  <el-option label="Sobel" value="sobel" />
                  <el-option label="Laplacian" value="laplacian" />
                </el-select>
              </el-form-item>
              <el-form-item label="最小颗粒面积">
                <el-input-number v-model="form.traditional_seg.min_area" :min="1" />
              </el-form-item>
            </template>

            <template v-if="form.segmentation_mode !== 'traditional'">
              <el-form-item label="模型槽位">
                <el-select v-model="form.dl_model.model_slot">
                  <el-option label="MatSAM" value="matsam" />
                  <el-option label="SAM LoRA" value="sam_lora" />
                  <el-option label="ResNeXt50" value="resnext50" />
                  <el-option label="Custom 占位运行器" value="custom" />
                </el-select>
              </el-form-item>
              <el-form-item label="权重路径">
                <el-input v-model="form.dl_model.weight_path" placeholder="可留空，后续在模型运行器里补齐" />
              </el-form-item>
              <el-form-item label="输入尺寸">
                <el-input-number v-model="form.dl_model.input_size" :min="256" :step="128" />
              </el-form-item>
              <el-form-item label="阈值">
                <el-slider v-model="form.dl_model.threshold" :min="0.1" :max="0.9" :step="0.05" />
              </el-form-item>
            </template>
          </div>
        </section>

        </el-form>
      </section>
    </div>

    <aside class="dashboard-side-column">
      <section class="glass-card panel panel-guide">
        <div class="panel-header compact-header">
          <div>
            <h2 class="section-title">当前方案</h2>
            <p class="section-subtitle">把关键决策压缩到一眼能看完的摘要区。</p>
          </div>
        </div>

        <div class="guide-summary">
          <div class="guide-metric">
            <span>输入方式</span>
            <strong>{{ form.input_mode }}</strong>
          </div>
          <div class="guide-metric">
            <span>分割模式</span>
            <strong>{{ currentModeLabel }}</strong>
          </div>
          <div class="guide-metric">
            <span>预处理</span>
            <strong>{{ form.preprocess.enabled ? '已启用' : '已跳过' }}</strong>
          </div>
          <div class="guide-metric">
            <span>图像数</span>
            <strong>{{ selectedFiles.length }}</strong>
          </div>
        </div>

        <div class="guide-block compact-block calibration-check-card">
          <span class="status-chip">{{ calibrationNeedsWarning ? '未标定' : '已标定' }}</span>
          <p>{{ calibrationReminder }}</p>
          <small>{{ calibrationProbeSummary }}</small>
        </div>

        <div class="guide-block compact-block">
          <span class="status-chip">流程</span>
          <p>导入图像后直接执行，结果会自动入库，并在结果中心和历史记录里回看。</p>
        </div>
      </section>

      <section class="glass-card panel panel-side">
        <div class="panel-header compact-header">
          <div>
            <h2 class="section-title">运行器状态</h2>
            <p class="section-subtitle">这里只保留和当前模式相关的信息。</p>
          </div>
        </div>

        <div class="runner-current">
          <strong>{{ currentRunner?.display_name ?? '传统分割不依赖运行器' }}</strong>
          <p v-for="note in runnerNotes" :key="note">{{ note }}</p>
        </div>

        <div v-if="form.segmentation_mode !== 'traditional'" class="runner-list compact-runner-list">
          <div v-for="runner in runners" :key="runner.id" class="runner-item" :class="{ 'is-selected': runner.id === currentRunner?.id }">
            <strong>{{ runner.display_name }}</strong>
            <p>{{ runner.slot }} · {{ runner.env_name || '未命名环境' }}</p>
          </div>
        </div>
      </section>
    </aside>

    <el-dialog
      v-model="calibrationDialogVisible"
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

        <div v-if="calibrationProbe" class="calibration-probe-grid">
          <div class="probe-card">
            <span>底栏检测</span>
            <strong>{{ calibrationProbe.footer_detected ? '已检测' : '未检测' }}</strong>
          </div>
          <div class="probe-card">
            <span>比例尺检测</span>
            <strong>{{ calibrationProbe.scale_bar_detected ? '已检测' : '未检测' }}</strong>
          </div>
          <div class="probe-card">
            <span>比例尺像素</span>
            <strong>{{ calibrationProbe.scale_bar_pixels ?? '--' }}</strong>
          </div>
          <div class="probe-card">
            <span>分析宽度</span>
            <strong>{{ calibrationProbe.analysis_width_px ?? '--' }}</strong>
          </div>
          <div class="probe-card">
            <span>源图宽度</span>
            <strong>{{ calibrationProbe.source_width_px ?? '--' }}</strong>
          </div>
          <div class="probe-card">
            <span>分析高度</span>
            <strong>{{ calibrationProbe.analysis_height_px ?? '--' }}</strong>
          </div>
          <div class="probe-card">
            <span>OCR 标尺</span>
            <strong>{{ calibrationProbe.ocr_scale_bar_um ?? '--' }} μm</strong>
          </div>
          <div class="probe-card">
            <span>FoV</span>
            <strong>{{ calibrationProbe.ocr_fov_um ?? '--' }} μm</strong>
          </div>
          <div class="probe-card">
            <span>放大倍数</span>
            <strong>{{ calibrationProbe.ocr_magnification_text ?? '--' }}</strong>
          </div>
          <div class="probe-card">
            <span>WD</span>
            <strong>{{ calibrationProbe.ocr_wd_mm ?? '--' }} mm</strong>
          </div>
          <div class="probe-card">
            <span>探测器</span>
            <strong>{{ calibrationProbe.ocr_detector ?? '--' }}</strong>
          </div>
          <div class="probe-card">
            <span>真空模式</span>
            <strong>{{ calibrationProbe.ocr_vacuum_mode ?? '--' }}</strong>
          </div>
        </div>

        <div class="calibration-form">
          <el-form label-position="top">
            <el-form-item label="比例尺对应的 μm 数值">
              <el-input-number v-model="calibrationScaleBarUm" :min="0.0001" :step="0.1" :precision="4" />
            </el-form-item>
            <el-form-item label="比例尺像素长度">
              <el-input-number v-model="calibrationScaleBarPixels" :min="1" :step="1" :precision="0" />
            </el-form-item>
          </el-form>
          <div class="calibration-result">
            <span>自动换算结果</span>
            <strong>{{ calibrationComputedValue !== null ? `${calibrationComputedValue.toFixed(6)} um/px` : '--' }}</strong>
          </div>
        </div>

        <div v-if="calibrationCandidates.length" class="candidate-row">
          <span>快捷候选</span>
          <div class="candidate-actions">
            <el-button
              v-if="calibrationProbe?.suggested_um_per_px"
              size="small"
              type="primary"
              plain
              @click="applySuggestedCalibration(calibrationProbe.suggested_um_per_px)"
            >
              OCR 建议值 → {{ calibrationProbe.suggested_um_per_px.toFixed(6) }} um/px
            </el-button>
            <el-button
              v-for="candidate in calibrationCandidates"
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
          <el-button @click="calibrationDialogVisible = false">返回填写</el-button>
          <el-button plain @click="continueWithoutCalibration">继续以像素单位运行</el-button>
          <el-button type="primary" :disabled="calibrationComputedValue === null" @click="applyCalibrationAndContinue">
            使用换算结果并继续
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.dashboard-layout {
  display: flex;
  gap: 18px;
  align-items: start;
}

.dashboard-main-column {
  flex: 1 1 auto;
  min-width: 0;
}

.dashboard-side-column {
  width: 360px;
  flex: 0 0 360px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.panel {
  padding: 20px;
}

.panel-header {
  margin-bottom: 18px;
}

.panel-guide,
.panel-side {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.panel-header-with-action {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.panel-header-action {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  flex-shrink: 0;
}

.header-submit-button {
  min-width: 152px;
  min-height: 46px;
  padding-inline: 18px;
  border-radius: 14px;
}

.compact-header {
  margin-bottom: 14px;
}

.form-section {
  padding: 18px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.46);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.form-section + .form-section {
  margin-top: 16px;
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

.upload-tip {
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
  display: flex;
  align-items: flex-start;
  gap: 8px;
  flex-wrap: wrap;
}

.field-helper {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 8px;
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

.upload-actions {
  display: flex;
  gap: 12px;
}

.upload-mode-tag {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 0 10px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-weight: 600;
  white-space: nowrap;
  word-break: keep-all;
  overflow-wrap: normal;
  flex: 0 0 auto;
}

.mode-selector {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
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

.hidden-input {
  display: none;
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

.process-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 14px;
}

.guide-block {
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.compact-block {
  margin-top: 14px;
}

.guide-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.guide-metric {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.guide-metric span {
  display: block;
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 12px;
}

.guide-metric strong {
  font-size: 18px;
}

.guide-block p {
  margin: 10px 0 0;
  color: var(--muted);
  line-height: 1.6;
}

.calibration-check-card small {
  display: block;
  margin-top: 8px;
  color: var(--muted);
  line-height: 1.5;
}

.runner-current {
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.runner-current strong {
  display: block;
  margin-bottom: 8px;
  font-size: 18px;
}

.runner-current p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.runner-current p + p {
  margin-top: 6px;
}

.runner-list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
  margin-top: 14px;
}

.runner-item {
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.runner-item.is-selected {
  border-color: rgba(23, 96, 135, 0.18);
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.08), rgba(184, 90, 43, 0.08));
}

.runner-item p {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
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
  box-shadow: 0 10px 22px rgba(23, 96, 135, 0.24);
  border-radius: 12px;
}

.calibration-dialog-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.calibration-warning {
  margin: 0;
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(184, 90, 43, 0.09);
  border: 1px solid rgba(184, 90, 43, 0.18);
  line-height: 1.7;
}

.calibration-warning code {
  padding: 2px 6px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.75);
}

.calibration-probe-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.probe-card {
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.probe-card span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 6px;
}

.probe-card strong {
  font-size: 16px;
}

.calibration-form {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 220px;
  gap: 14px;
  align-items: end;
}

.calibration-result {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(23, 96, 135, 0.08);
  border: 1px solid rgba(23, 96, 135, 0.14);
}

.calibration-result span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 6px;
}

.calibration-result strong {
  font-size: 18px;
}

.candidate-row {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.candidate-row > span {
  color: var(--muted);
  font-size: 13px;
}

.candidate-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.dialog-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

@media (max-width: 1100px) {
  .panel-header-with-action {
    flex-direction: column;
  }

  .panel-header-action {
    width: 100%;
    justify-content: stretch;
  }

  .header-submit-button {
    width: 100%;
  }

  .calibration-probe-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .calibration-form {
    grid-template-columns: 1fr;
  }
}
</style>
