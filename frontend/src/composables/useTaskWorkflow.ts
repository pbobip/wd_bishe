import { ElMessage } from 'element-plus'
import { computed, reactive, ref } from 'vue'

import { api } from '../api'
import type { ModelRunner, Project, RunResultsPayload } from '../types'
import {
  buildUmPerPxFromScaleBar,
  inspectCalibrationProbe,
  type CalibrationProbe,
} from '../utils/calibration'

export type FileSelectionSource = 'files' | 'folder'
type LaunchPhase = 'idle' | 'creating' | 'uploading' | 'ready' | 'starting' | 'running' | 'completed' | 'failed'
type PostprocessSmoothingMethod = 'mean' | 'gaussian' | 'median'
type TraditionalMethod = 'threshold' | 'adaptive' | 'edge' | 'clustering'
type PreprocessBackgroundMethod = 'none' | 'tophat' | 'rolling_ball'
type PreprocessDenoiseMethod = 'none' | 'wavelet' | 'gaussian' | 'median' | 'bilateral' | 'mean'
type PreprocessEnhanceMethod = 'none' | 'clahe' | 'hist_equalization' | 'gamma'
type MeasurementKey =
  | 'vf'
  | 'particle_count'
  | 'area'
  | 'size'
  | 'channel_width'
  | 'mean'
  | 'median'
  | 'std'

const SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
const UPLOAD_CHUNK_SIZE = 10
const ACTIVE_RUN_STORAGE_KEY = 'wd_bishe.active_run_id'

const segmentationOptions = [
  { label: '传统分割', value: 'traditional' },
  { label: '深度学习', value: 'dl' },
  { label: '结果对比', value: 'compare' },
]

const traditionalMethodOptions = [
  { label: '阈值分割', value: 'threshold' },
  { label: '自适应阈值', value: 'adaptive' },
  { label: '边缘分割', value: 'edge' },
  { label: '聚类分割', value: 'clustering' },
]

const preprocessBackgroundOptions = [
  { label: '无背景校正', value: 'none' },
  { label: 'Top-hat', value: 'tophat' },
  { label: 'Rolling-ball', value: 'rolling_ball' },
]

const preprocessDenoiseOptions = [
  { label: '不去噪', value: 'none' },
  { label: '改进小波', value: 'wavelet' },
  { label: '高斯滤波', value: 'gaussian' },
  { label: '中值滤波', value: 'median' },
  { label: '双边滤波', value: 'bilateral' },
  { label: '均值滤波', value: 'mean' },
]

const preprocessEnhanceOptions = [
  { label: '不增强', value: 'none' },
  { label: 'CLAHE', value: 'clahe' },
  { label: '直方图均衡化', value: 'hist_equalization' },
  { label: 'Gamma 校正', value: 'gamma' },
]

const measurementOptions = [
  { label: 'Vf', value: 'vf' },
  { label: '颗粒数', value: 'particle_count' },
  { label: '面积', value: 'area' },
  { label: '尺寸', value: 'size' },
  { label: '通道宽度', value: 'channel_width' },
  { label: '均值', value: 'mean' },
  { label: '中位数', value: 'median' },
  { label: '标准差', value: 'std' },
] as const

const createDefaultForm = () => ({
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
    background: {
      method: 'none' as PreprocessBackgroundMethod,
      radius: 25,
    },
    denoise: {
      method: 'wavelet' as PreprocessDenoiseMethod,
      wavelet_strength: 0.12,
      mean_kernel: 3,
      gaussian_kernel: 3,
      median_kernel: 3,
      bilateral_diameter: 5,
      bilateral_sigma_color: 45,
      bilateral_sigma_space: 9,
    },
    enhance: {
      method: 'clahe' as PreprocessEnhanceMethod,
      clahe_clip_limit: 2.0,
      clahe_tile_size: 8,
      gamma: 1.0,
    },
    extras: {
      unsharp: false,
      unsharp_radius: 3,
      unsharp_amount: 1.0,
    },
  },
  traditional_seg: {
    method: 'threshold' as TraditionalMethod,
    threshold_mode: 'otsu',
    global_threshold: 120,
    fixed_threshold: 120,
    adaptive_method: 'gaussian',
    adaptive_block_size: 35,
    adaptive_c: 5,
    edge_operator: 'canny',
    edge_blur_kernel: 3,
    edge_threshold1: 60,
    edge_threshold2: 180,
    edge_dilate_iterations: 1,
    kmeans_clusters: 2,
    kmeans_attempts: 5,
    cluster_target: 'bright',
    fill_holes: true,
    watershed: false,
    boundary_smoothing: true,
    boundary_smoothing_kernel: 3,
    min_area: 30,
    max_area: undefined as number | undefined,
    min_solidity: 0.0,
    min_circularity: 0.0,
    min_roundness: 0.0,
    max_aspect_ratio: undefined as number | undefined,
    remove_border: true,
    open_kernel: 3,
    close_kernel: 3,
  },
  postprocess: {
    fill_holes: true,
    watershed: false,
    smoothing: {
      enabled: true,
      method: 'gaussian' as PostprocessSmoothingMethod,
      kernel: 3,
    },
    shape_filter: {
      enabled: true,
      min_area: 30,
      max_area: undefined as number | undefined,
      min_solidity: 0.0,
      min_circularity: 0.0,
      min_roundness: 0.0,
      max_aspect_ratio: undefined as number | undefined,
    },
    remove_border: true,
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
    measurements: {
      vf: true,
      particle_count: true,
      area: true,
      size: true,
      channel_width: true,
      mean: true,
      median: true,
      std: true,
    } as Record<MeasurementKey, boolean>,
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

const form = reactive(createDefaultForm())
const projects = ref<Project[]>([])
const runners = ref<ModelRunner[]>([])
const selectedFiles = ref<File[]>([])
const selectionSource = ref<FileSelectionSource>('files')
const initializing = ref(false)
const initialized = ref(false)
const calibrationProbe = ref<CalibrationProbe | null>(null)
const calibrationProbeLoading = ref(false)
const calibrationDialogVisible = ref(false)
const calibrationScaleBarUm = ref<number | null>(null)
const calibrationScaleBarPixels = ref<number | null>(null)
const launchPhase = ref<LaunchPhase>('idle')
const launchMessage = ref('等待创建任务')
const launchError = ref<string | null>(null)
const activeRunId = ref<number | null>(null)
const activePayload = ref<RunResultsPayload | null>(null)
const actionLoading = ref(false)
const uploadState = reactive({
  totalFiles: 0,
  uploadedFiles: 0,
  totalBytes: 0,
  uploadedBytes: 0,
  totalChunks: 0,
  completedChunks: 0,
  currentChunkIndex: 0,
  currentChunkProgress: 0,
  percentage: 0,
})

let calibrationProbeSeq = 0
let calibrationProbeKey: string | null = null
let initPromise: Promise<void> | null = null
let pollTimer: number | null = null
let preparedSelectionKey: string | null = null

const canUseSessionStorage = () => typeof window !== 'undefined' && typeof window.sessionStorage !== 'undefined'

const readStoredActiveRunId = () => {
  if (!canUseSessionStorage()) return null
  const raw = window.sessionStorage.getItem(ACTIVE_RUN_STORAGE_KEY)
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

const persistActiveRunId = (runId: number | null) => {
  if (!canUseSessionStorage()) return
  if (typeof runId === 'number' && Number.isFinite(runId) && runId > 0) {
    window.sessionStorage.setItem(ACTIVE_RUN_STORAGE_KEY, String(runId))
    return
  }
  window.sessionStorage.removeItem(ACTIVE_RUN_STORAGE_KEY)
}

const isPlainObject = (value: unknown): value is Record<string, any> =>
  Object.prototype.toString.call(value) === '[object Object]'

const deepAssign = (target: Record<string, any>, source: Record<string, any>) => {
  Object.entries(source).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      target[key] = [...value]
      return
    }
    if (isPlainObject(value)) {
      if (!isPlainObject(target[key])) {
        target[key] = {}
      }
      deepAssign(target[key], value)
      return
    }
    target[key] = value
  })
}

const applyRunConfig = (config?: Record<string, any> | null) => {
  const defaults = createDefaultForm()
  deepAssign(form as unknown as Record<string, any>, defaults as unknown as Record<string, any>)
  if (config && isPlainObject(config)) {
    deepAssign(form as unknown as Record<string, any>, config)
  }
}

const ensureProjectSelection = (projectId?: number | null) => {
  if (!projects.value.length) return
  const requested = typeof projectId === 'number' && Number.isFinite(projectId) ? projectId : null
  const matched = requested ? projects.value.find((project) => project.id === requested) : null
  if (matched) {
    form.project_id = matched.id
    return
  }
  if (!projects.value.some((project) => project.id === form.project_id)) {
    form.project_id = projects.value[0].id
  }
}

const loadReferenceData = async () => {
  const [projectRes, runnerRes] = await Promise.all([
    api.get<Project[]>('/projects'),
    api.get<ModelRunner[]>('/model-runners'),
  ])
  projects.value = projectRes.data
  runners.value = runnerRes.data
}

const initialize = async (projectId?: number | null) => {
  if (!initialized.value) {
    if (!initPromise) {
      initializing.value = true
      initPromise = loadReferenceData()
        .then(() => {
          initialized.value = true
        })
        .finally(() => {
          initializing.value = false
          initPromise = null
        })
    }
    await initPromise
  }
  ensureProjectSelection(projectId)
}

const traditionalMethodLabelMap: Record<TraditionalMethod, string> = {
  threshold: '阈值分割',
  adaptive: '自适应阈值',
  edge: '边缘分割',
  clustering: '聚类分割',
}

const traditionalMethodHint = computed(() => {
  if (form.traditional_seg.method === 'adaptive') {
    return '按局部窗口自适应估计阈值，适合光照不均或背景起伏较明显的图像。'
  }
  if (form.traditional_seg.method === 'edge') {
    return '先提边，再闭合与填充轮廓，适合边界清楚但灰度阈值不稳定的图像。'
  }
  if (form.traditional_seg.method === 'clustering') {
    return '按灰度聚类后选择目标簇，适合双峰不明显但仍有层次差的图像。'
  }
  return '使用 Otsu、全局阈值或固定阈值作为传统基线，速度快且便于与深度学习对照。'
})

const segmentationHint = computed(() => {
  if (form.segmentation_mode === 'traditional') {
    return `适合特征明显、速度优先的场景，当前采用 ${traditionalMethodLabelMap[form.traditional_seg.method]} 路线。`
  }
  if (form.segmentation_mode === 'dl') {
    return '调用深度学习模型接口做自动分割，适合后续接入 MatSAM、SAM LoRA 或 ResNeXt50。'
  }
  return '同时输出传统分割和深度学习结果，适合答辩展示双路线差异。'
})

const currentModeLabel = computed(
  () => segmentationOptions.find((item) => item.value === form.segmentation_mode)?.label ?? '传统分割',
)

const preprocessBackgroundLabelMap: Record<PreprocessBackgroundMethod, string> = {
  none: '无背景校正',
  tophat: 'Top-hat',
  rolling_ball: 'Rolling-ball',
}

const preprocessDenoiseLabelMap: Record<PreprocessDenoiseMethod, string> = {
  none: '不去噪',
  wavelet: '改进小波',
  gaussian: '高斯滤波',
  median: '中值滤波',
  bilateral: '双边滤波',
  mean: '均值滤波',
}

const preprocessEnhanceLabelMap: Record<PreprocessEnhanceMethod, string> = {
  none: '不增强',
  clahe: 'CLAHE',
  hist_equalization: '直方图均衡化',
  gamma: 'Gamma 校正',
}

const preprocessBackgroundHint = computed(() => {
  if (form.preprocess.background.method === 'tophat') {
    return '适合小尺度亮相结构增强，可压制缓慢变化的背景起伏。'
  }
  if (form.preprocess.background.method === 'rolling_ball') {
    return '更适合 SEM 图像的背景漂移校正，再进入后续去噪和分割。'
  }
  return '不做背景校正，保留原始亮度分布。'
})

const preprocessDenoiseHint = computed(() => {
  if (form.preprocess.denoise.method === 'wavelet') {
    return '默认推荐给传统分割，兼顾保边和抑噪。'
  }
  if (form.preprocess.denoise.method === 'gaussian') {
    return '适合做轻度平滑，降低随机噪声，但会稍微软化边界。'
  }
  if (form.preprocess.denoise.method === 'median') {
    return '对脉冲噪声更稳，适合局部亮点或椒盐感更明显的图像。'
  }
  if (form.preprocess.denoise.method === 'bilateral') {
    return '保留边界的同时平滑局部纹理，适合想兼顾轮廓与细节的场景。'
  }
  if (form.preprocess.denoise.method === 'mean') {
    return '均值滤波更适合作为兼容选项，不建议长期作为默认主方法。'
  }
  return '跳过去噪，适合深度学习路线或原图质量已经较稳定的场景。'
})

const preprocessEnhanceHint = computed(() => {
  if (form.preprocess.enhance.method === 'clahe') {
    return '推荐默认增强方案，适合局部对比不足的 SEM 图像。'
  }
  if (form.preprocess.enhance.method === 'hist_equalization') {
    return '做全局对比拉伸，适合整体灰度偏平但照明相对均匀的图像。'
  }
  if (form.preprocess.enhance.method === 'gamma') {
    return '通过 Gamma 调整亮暗层次，适合整体偏暗或偏亮的样本。'
  }
  return '跳过主增强，保留去噪后的自然灰度分布。'
})

const preprocessHasEffectiveStep = computed(() => {
  if (!form.preprocess.enabled) return true
  return (
    form.preprocess.background.method !== 'none'
    || form.preprocess.denoise.method !== 'none'
    || form.preprocess.enhance.method !== 'none'
    || form.preprocess.extras.unsharp
  )
})

const preprocessSummary = computed(() => {
  if (!form.preprocess.enabled) return '当前跳过预处理'
  const parts: string[] = []
  if (form.preprocess.background.method !== 'none') {
    parts.push(preprocessBackgroundLabelMap[form.preprocess.background.method])
  }
  if (form.preprocess.denoise.method !== 'none') {
    parts.push(preprocessDenoiseLabelMap[form.preprocess.denoise.method])
  }
  if (form.preprocess.enhance.method !== 'none') {
    parts.push(preprocessEnhanceLabelMap[form.preprocess.enhance.method])
  }
  if (form.preprocess.extras.unsharp) {
    parts.push('轻度锐化')
  }
  return parts.length ? `已启用：${parts.join(' / ')}` : '已启用预处理，等待选择至少一种有效流程'
})
const preprocessSelectionValid = computed(() => preprocessHasEffectiveStep.value)
const preprocessValidationMessage = computed(() =>
  preprocessSelectionValid.value ? '' : '已启用预处理，请至少选择一种背景校正、去噪、增强或可选增强。',
)

type PreprocessPreset = 'traditional' | 'dl' | 'compare'

const applyPreprocessPreset = (preset: PreprocessPreset) => {
  form.preprocess.enabled = true
  form.preprocess.extras.unsharp = false
  form.preprocess.extras.unsharp_radius = 3
  form.preprocess.extras.unsharp_amount = 1.0

  if (preset === 'traditional') {
    form.preprocess.background.method = 'rolling_ball'
    form.preprocess.background.radius = 25
    form.preprocess.denoise.method = 'wavelet'
    form.preprocess.denoise.wavelet_strength = 0.12
    form.preprocess.enhance.method = 'clahe'
    form.preprocess.enhance.clahe_clip_limit = 2.0
    form.preprocess.enhance.clahe_tile_size = 8
    return
  }

  if (preset === 'dl') {
    form.preprocess.background.method = 'none'
    form.preprocess.denoise.method = 'gaussian'
    form.preprocess.denoise.gaussian_kernel = 3
    form.preprocess.enhance.method = 'none'
    return
  }

  form.preprocess.background.method = 'rolling_ball'
  form.preprocess.background.radius = 25
  form.preprocess.denoise.method = 'wavelet'
  form.preprocess.denoise.wavelet_strength = 0.1
  form.preprocess.enhance.method = 'clahe'
  form.preprocess.enhance.clahe_clip_limit = 1.8
  form.preprocess.enhance.clahe_tile_size = 8
}

const postprocessSummary = computed(() => {
  const parts: string[] = []
  if (form.postprocess.fill_holes) parts.push('填孔')
  if (form.postprocess.watershed) parts.push('watershed')
  if (form.postprocess.smoothing.enabled) {
    parts.push(`平滑(${form.postprocess.smoothing.method}/${form.postprocess.smoothing.kernel})`)
  }
  if (form.postprocess.shape_filter.enabled) parts.push('形状过滤')
  if (form.postprocess.remove_border) parts.push('触边剔除')
  return parts.length ? `后处理已启用：${parts.join(' / ')}` : '当前后处理仅保留模型/阈值输出'
})

const measurementLabels: Record<MeasurementKey, string> = {
  vf: 'Vf',
  particle_count: '颗粒数',
  area: '面积',
  size: '尺寸',
  channel_width: '通道宽度',
  mean: '均值',
  median: '中位数',
  std: '标准差',
}

const measurementSummary = computed(() => {
  const enabled = Object.entries(form.stats.measurements)
    .filter(([, value]) => value)
    .map(([key]) => measurementLabels[key as MeasurementKey])
  return enabled.length ? `测量项：${enabled.join(' / ')}` : '未勾选测量项'
})

const measurementCount = computed(() => Object.values(form.stats.measurements).filter(Boolean).length)

const calibrationNeedsWarning = computed(() => form.stats.enabled && !form.input_config.um_per_px)
const calibrationReminder = computed(() =>
  calibrationNeedsWarning.value
    ? '当前未填写 um_per_px，本次任务仍可运行，但所有统计将只输出 px / px²。'
    : `当前标定值：${form.input_config.um_per_px?.toFixed(4)} um/px`,
)
const calibrationProbeSummary = computed(() => {
  if (!selectedFiles.value.length) return ''
  if (calibrationProbeLoading.value) return '正在后台识别底栏和比例尺...'
  if (!calibrationProbe.value) return ''
  if (calibrationProbe.value.suggested_um_per_px) {
    return `已识别到比例尺建议值 ${calibrationProbe.value.suggested_um_per_px.toFixed(6)} um/px`
  }
  if (calibrationProbe.value.scale_bar_detected) {
    return `已识别到比例尺横线 ${calibrationProbe.value.scale_bar_pixels ?? '--'} px，可继续确认 μm 数值`
  }
  if (calibrationProbe.value.footer_detected) {
    return '已检测到底部信息栏，但未稳定识别比例尺，请手动填写 um_per_px。'
  }
  return '首张图未检测到底栏或比例尺；如果整批都无信息栏，可直接手动填写或按像素单位运行。'
})
const calibrationComputedValue = computed(() =>
  buildUmPerPxFromScaleBar(calibrationScaleBarUm.value, calibrationScaleBarPixels.value),
)
const calibrationCandidates = computed(() => calibrationProbe.value?.common_scale_candidates ?? [])
const calibrationSuggestedValue = computed(() => calibrationProbe.value?.suggested_um_per_px ?? null)
const calibrationActionLabel = computed(() => {
  if (calibrationProbeLoading.value) return '识别中...'
  return calibrationProbe.value ? '重新识别比例尺' : '自动识别比例尺'
})
const calibrationFooterHint = computed(() => {
  if (!selectedFiles.value.length) return '如图像没有底部信息栏，运行时会自动跳过这一步。'
  if (calibrationProbeLoading.value) return '如有底栏，系统会在需要时提取 Mag、FoV、WD 和比例尺信息。'
  if (!calibrationProbe.value) return '可按需识别底栏信息；若整批图像没有信息栏，运行时会自动跳过。'
  if (!calibrationProbe.value.footer_detected) {
    return '首张图未检测到底部信息栏；若整批都无底栏，运行时会自动跳过保存。'
  }
  return '已检测到首张图底栏，运行时可保留 Mag、FoV、WD 和比例尺图片用于追溯。'
})

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

const inputModeLabel = computed(() =>
  form.input_mode === 'single'
    ? '单张输入'
    : `批处理 · ${selectionSource.value === 'folder' ? '来自文件夹' : '来自多图选择'}`,
)

const workflowReady = computed(() => Boolean(selectedFiles.value.length && form.name.trim()))
const canNavigateToConfig = computed(() => workflowReady.value)
const canLaunch = computed(() => workflowReady.value && preprocessSelectionValid.value && !actionLoading.value)

const currentProject = computed(
  () => projects.value.find((project) => project.id === form.project_id) ?? projects.value[0] ?? null,
)

const resetUploadState = () => {
  uploadState.totalFiles = selectedFiles.value.length
  uploadState.uploadedFiles = 0
  uploadState.totalBytes = selectedFiles.value.reduce((sum, file) => sum + file.size, 0)
  uploadState.uploadedBytes = 0
  uploadState.totalChunks = 0
  uploadState.completedChunks = 0
  uploadState.currentChunkIndex = 0
  uploadState.currentChunkProgress = 0
  uploadState.percentage = 0
}

const resetRunSession = () => {
  if (pollTimer) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
  activeRunId.value = null
  activePayload.value = null
  persistActiveRunId(null)
  launchError.value = null
  launchPhase.value = 'idle'
  launchMessage.value = '等待创建任务'
  preparedSelectionKey = null
  resetUploadState()
}

const stopPolling = () => {
  if (pollTimer) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

const activeRunningStep = computed(() => activePayload.value?.steps.find((step) => step.status === 'running') ?? null)
const activeStepProgressText = computed(() => {
  const details = activeRunningStep.value?.details
  if (!details || !details.total_images) return '等待逐图进度'
  return `已处理 ${details.processed_images} / ${details.total_images} 张`
})

const loadActivePayload = async () => {
  if (!activeRunId.value) return
  const response = await api.get<RunResultsPayload>(`/runs/${activeRunId.value}/results`)
  activePayload.value = response.data
  applyRunConfig(response.data.run.config ?? null)
  persistActiveRunId(response.data.run.id)
  const status = response.data.run.status
  const runningStep = response.data.steps.find((step) => step.status === 'running')
  if (['completed', 'partial_success'].includes(status)) {
    launchPhase.value = 'completed'
    launchMessage.value = status === 'partial_success' ? '任务部分成功完成' : '任务已完成'
  } else if (status === 'failed') {
    launchPhase.value = 'failed'
    launchMessage.value = response.data.run.error_message || '任务执行失败'
  } else if (status === 'draft') {
    launchPhase.value = 'ready'
    launchMessage.value = '草稿任务已恢复，可继续配置并运行'
  } else {
    launchPhase.value = 'running'
    launchMessage.value = runningStep?.details?.message || runningStep?.message || '任务执行中，正在刷新步骤状态'
  }
}

const restoreDraftRunFromSession = async () => {
  if (activeRunId.value) {
    return true
  }
  const storedRunId = readStoredActiveRunId()
  if (!storedRunId) {
    return false
  }
  activeRunId.value = storedRunId
  try {
    await loadActivePayload()
    return true
  } catch {
    activeRunId.value = null
    activePayload.value = null
    persistActiveRunId(null)
    return false
  }
}

const startPolling = () => {
  stopPolling()
  pollTimer = window.setInterval(async () => {
    try {
      await loadActivePayload()
      const status = activePayload.value?.run.status
      if (status && ['completed', 'failed', 'partial_success'].includes(status)) {
        stopPolling()
      }
    } catch {
      stopPolling()
    }
  }, 2500)
}

const getCalibrationProbeFileKey = () => {
  const file = selectedFiles.value[0]
  if (!file) return null
  return `${file.name}:${file.size}:${file.lastModified}`
}

const runCalibrationProbe = async (force = false) => {
  if (!selectedFiles.value.length) {
    calibrationProbe.value = null
    calibrationProbeKey = null
    return null
  }

  if (calibrationProbeLoading.value) {
    return calibrationProbe.value
  }

  const probeKey = getCalibrationProbeFileKey()
  if (!force && probeKey && calibrationProbe.value && calibrationProbeKey === probeKey) {
    return calibrationProbe.value
  }

  const requestSeq = ++calibrationProbeSeq
  calibrationProbeLoading.value = true
  try {
    const probe = await inspectCalibrationProbe([selectedFiles.value[0]])
    if (requestSeq !== calibrationProbeSeq) return calibrationProbe.value
    calibrationProbe.value = probe
    calibrationProbeKey = probeKey
    calibrationScaleBarUm.value = probe?.ocr_scale_bar_um ?? calibrationScaleBarUm.value
    if (probe?.scale_bar_pixels) {
      calibrationScaleBarPixels.value = probe.scale_bar_pixels
    }
    return probe
  } finally {
    if (requestSeq === calibrationProbeSeq) {
      calibrationProbeLoading.value = false
    }
  }
}

const setSelectedFiles = async (files: File[], source: FileSelectionSource) => {
  resetRunSession()
  const accepted = files.filter((file) => {
    const lower = file.name.toLowerCase()
    return SUPPORTED_EXTENSIONS.some((ext) => lower.endsWith(ext))
  })
  const skipped = files.length - accepted.length
  selectionSource.value = source
  selectedFiles.value = accepted
  form.input_mode = source === 'folder' || accepted.length > 1 ? 'batch' : 'single'
  calibrationProbe.value = null
  calibrationScaleBarPixels.value = null
  calibrationScaleBarUm.value = null
  calibrationProbeKey = null
  calibrationProbeLoading.value = false
  if (skipped > 0) {
    ElMessage.warning(`已跳过 ${skipped} 个非图像文件，仅保留 png/jpg/jpeg/tif/tiff`)
  }
}

const clearSelectedFiles = () => {
  resetRunSession()
  selectedFiles.value = []
  selectionSource.value = 'files'
  form.input_mode = 'batch'
  calibrationProbe.value = null
  calibrationScaleBarPixels.value = null
  calibrationScaleBarUm.value = null
  calibrationProbeKey = null
  calibrationProbeLoading.value = false
  calibrationProbeSeq += 1
}

const openCalibrationDialog = async () => {
  calibrationDialogVisible.value = true
  if (!calibrationProbe.value && selectedFiles.value.length) {
    await runCalibrationProbe()
  }
  calibrationScaleBarUm.value = calibrationProbe.value?.ocr_scale_bar_um ?? calibrationScaleBarUm.value
  calibrationScaleBarPixels.value = calibrationProbe.value?.scale_bar_pixels ?? calibrationScaleBarPixels.value
}

const closeCalibrationDialog = () => {
  calibrationDialogVisible.value = false
}

const applyCalibrationValue = (value: number | null) => {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    form.input_config.um_per_px = value
  }
}

const chunkFiles = <T,>(items: T[], size: number) => {
  const chunks: T[][] = []
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size))
  }
  return chunks
}

const getSelectedFilesKey = () =>
  selectedFiles.value.map((file) => `${file.name}:${file.size}:${file.lastModified}`).join('|')

const uploadSelectedFiles = async (runId: number) => {
  const chunks = chunkFiles(selectedFiles.value, UPLOAD_CHUNK_SIZE)
  uploadState.totalFiles = selectedFiles.value.length
  uploadState.totalBytes = selectedFiles.value.reduce((sum, file) => sum + file.size, 0)
  uploadState.totalChunks = chunks.length
  uploadState.completedChunks = 0

  let completedFiles = 0
  let uploadedBytes = 0
  for (const [index, chunk] of chunks.entries()) {
    const chunkBytes = chunk.reduce((sum, file) => sum + file.size, 0)
    uploadState.currentChunkIndex = index + 1
    uploadState.currentChunkProgress = 0
    launchMessage.value = `正在上传第 ${index + 1}/${chunks.length} 批图像`
    const multipart = new FormData()
    chunk.forEach((file) => {
      multipart.append('files', file)
      const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name
      multipart.append('relative_paths', relativePath)
    })
    await api.post(`/runs/${runId}/images`, multipart, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (event) => {
        const ratio = event.total ? event.loaded / event.total : 0
        uploadState.currentChunkProgress = ratio
        uploadState.uploadedBytes = Math.min(uploadState.totalBytes, uploadedBytes + (event.loaded ?? 0))
        uploadState.percentage = Number(
          (((uploadedBytes + chunkBytes * ratio) / Math.max(uploadState.totalBytes, 1)) * 100).toFixed(2),
        )
      },
    })
    completedFiles += chunk.length
    uploadedBytes += chunkBytes
    uploadState.uploadedFiles = completedFiles
    uploadState.uploadedBytes = uploadedBytes
    uploadState.completedChunks = index + 1
    uploadState.currentChunkProgress = 1
    uploadState.percentage = Number(((uploadedBytes / Math.max(uploadState.totalBytes, 1)) * 100).toFixed(2))
  }
}

const syncDraftRun = async (runId: number) => {
  const response = await api.put(`/runs/${runId}`, buildRunPayload())
  return response.data
}

const buildRunPayload = () => {
  form.compare.enabled = form.segmentation_mode === 'compare'
  if (!form.preprocess.enabled) {
    form.preprocess.background.method = 'none'
    form.preprocess.denoise.method = 'none'
    form.preprocess.enhance.method = 'none'
    form.preprocess.extras.unsharp = false
  }
  form.traditional_seg.fill_holes = form.postprocess.fill_holes
  form.traditional_seg.watershed = form.postprocess.watershed
  form.traditional_seg.boundary_smoothing = form.postprocess.smoothing.enabled
  form.traditional_seg.boundary_smoothing_kernel = form.postprocess.smoothing.kernel
  form.traditional_seg.min_area = form.postprocess.shape_filter.enabled ? form.postprocess.shape_filter.min_area : 1
  form.traditional_seg.max_area = form.postprocess.shape_filter.enabled ? form.postprocess.shape_filter.max_area : undefined
  form.traditional_seg.min_solidity = form.postprocess.shape_filter.enabled ? form.postprocess.shape_filter.min_solidity : 0
  form.traditional_seg.min_circularity = form.postprocess.shape_filter.enabled ? form.postprocess.shape_filter.min_circularity : 0
  form.traditional_seg.min_roundness = form.postprocess.shape_filter.enabled ? form.postprocess.shape_filter.min_roundness : 0
  form.traditional_seg.max_aspect_ratio = form.postprocess.shape_filter.enabled
    ? form.postprocess.shape_filter.max_aspect_ratio
    : undefined
  form.traditional_seg.remove_border = form.postprocess.remove_border
  if (form.segmentation_mode !== 'traditional') {
    const matched = runners.value.find((runner) => runner.slot === form.dl_model.model_slot)
    form.dl_model.runner_id = matched?.id
  } else {
    form.dl_model.runner_id = undefined
  }
  return form
}

const ensurePreprocessReady = (notify = true) => {
  if (preprocessSelectionValid.value) return true
  if (notify) {
    ElMessage.warning(preprocessValidationMessage.value)
  }
  return false
}

const prepareCurrentTask = async () => {
  if (!selectedFiles.value.length) {
    ElMessage.warning('请先选择图像')
    return false
  }
  if (!ensurePreprocessReady()) {
    return false
  }

  actionLoading.value = true
  launchError.value = null
  stopPolling()
  const selectionKey = getSelectedFilesKey()

  try {
    const canReuseDraft =
      Boolean(activeRunId.value)
      && preparedSelectionKey === selectionKey
      && ['draft', 'queued'].includes(activePayload.value?.run.status ?? 'draft')

    if (canReuseDraft && activeRunId.value) {
      persistActiveRunId(activeRunId.value)
      launchPhase.value = 'creating'
      launchMessage.value = '正在同步任务草稿配置'
      await syncDraftRun(activeRunId.value)
      await loadActivePayload()
      launchPhase.value = 'ready'
      launchMessage.value = '图像已登记，可进入后处理页继续配置并运行'
      return true
    }

    activePayload.value = null
    activeRunId.value = null
    resetUploadState()

    launchPhase.value = 'creating'
    launchMessage.value = '正在创建任务草稿'
    const runRes = await api.post('/runs', buildRunPayload())
    const runId = Number(runRes.data.id)
    activeRunId.value = runId
    persistActiveRunId(runId)

    launchPhase.value = 'uploading'
    await uploadSelectedFiles(runId)

    await loadActivePayload()
    preparedSelectionKey = selectionKey
    launchPhase.value = 'ready'
    launchMessage.value = '图像已登记，可进入后处理页继续配置并运行'
    ElMessage.success('图像已上传并登记到任务草稿')
    return true
  } catch (error: any) {
    launchPhase.value = 'failed'
    const message = error?.response?.data?.detail ?? error?.message ?? '任务创建失败'
    launchError.value = String(message)
    launchMessage.value = launchError.value
    ElMessage.error(launchError.value)
    return false
  } finally {
    actionLoading.value = false
  }
}

const launchCurrentTask = async () => {
  if (!ensurePreprocessReady()) {
    return false
  }
  if (!activeRunId.value) {
    ElMessage.warning('请先在任务创建页完成图像上传与草稿准备')
    return false
  }
  if (!selectedFiles.value.length && !(activePayload.value?.images.length ?? 0)) {
    ElMessage.warning('当前草稿任务还没有已登记图像，请先回到任务创建页导入图片')
    return false
  }

  actionLoading.value = true
  launchError.value = null
  stopPolling()

  try {
    launchPhase.value = 'starting'
    launchMessage.value = '正在同步后处理配置'
    await syncDraftRun(activeRunId.value)

    launchMessage.value = '图像已就绪，正在启动任务'
    await api.post(`/runs/${activeRunId.value}/execute`)

    launchPhase.value = 'running'
    launchMessage.value = '任务已启动，正在获取执行状态'
    await loadActivePayload()
    const payload = activePayload.value as RunResultsPayload | null
    const currentStatus = payload?.run.status
    if (!currentStatus || !['completed', 'failed', 'partial_success'].includes(currentStatus)) {
      startPolling()
    }
    ElMessage.success('任务已启动，后处理页会持续显示执行状态')
    return true
  } catch (error: any) {
    launchPhase.value = 'failed'
    const message = error?.response?.data?.detail ?? error?.message ?? '任务启动失败'
    launchError.value = String(message)
    launchMessage.value = launchError.value
    ElMessage.error(launchError.value)
    return false
  } finally {
    actionLoading.value = false
  }
}

const executionProgress = computed(() => {
  if (activePayload.value) {
    return Math.round((activePayload.value.run.progress ?? 0) * 100)
  }
  return Math.round(uploadState.percentage)
})

const uploadProgressLabel = computed(() => {
  const registeredImageCount = activePayload.value?.images.length ?? 0
  if (!selectedFiles.value.length && !registeredImageCount) return '尚未选择图像'
  if (launchPhase.value === 'idle') return `待上传 ${selectedFiles.value.length} 张图像`
  if (launchPhase.value === 'creating') return '正在创建任务记录'
  if (launchPhase.value === 'uploading') {
    return `第 ${uploadState.currentChunkIndex}/${Math.max(uploadState.totalChunks, 1)} 批 · ${uploadState.uploadedFiles}/${uploadState.totalFiles} 张 · ${uploadState.percentage.toFixed(0)}%`
  }
  if (launchPhase.value === 'ready') {
    return `已登记 ${registeredImageCount || uploadState.totalFiles || selectedFiles.value.length} 张图像`
  }
  if (registeredImageCount || uploadState.totalFiles) {
    const count = registeredImageCount || uploadState.totalFiles
    return `已上传 ${count}/${count} 张图像`
  }
  return '等待上传'
})

const runStatusLabel = computed(() => {
  if (activePayload.value) return activePayload.value.run.status
  if (launchPhase.value === 'idle') return '待启动'
  if (launchPhase.value === 'failed') return '创建失败'
  return launchPhase.value
})

const workflowSteps = computed(() => {
  const registeredImageCount = activePayload.value?.images.length ?? selectedFiles.value.length
  const steps = [
    {
      key: 'draft',
      status: workflowReady.value || Boolean(activeRunId.value) ? 'completed' : 'pending',
      message:
        workflowReady.value || Boolean(activeRunId.value)
          ? `草稿已就绪 · ${registeredImageCount} 张图像待执行`
          : '请先补齐任务名称与图像导入',
    },
    {
      key: '图像登记',
      status:
        launchPhase.value === 'uploading'
          ? 'running'
          : ['starting', 'running', 'completed'].includes(launchPhase.value)
            ? 'completed'
            : launchPhase.value === 'failed'
              ? 'failed'
              : 'pending',
      message: uploadProgressLabel.value,
    },
  ]

  if (activePayload.value?.steps.length) {
    return steps.concat(
      activePayload.value.steps.map((step) => ({
        key: step.step_key,
        status: step.status,
        message: step.details?.message || step.message || '等待执行',
      })),
    )
  }

  steps.push({
    key: 'execute',
    status:
      launchPhase.value === 'running'
        ? 'running'
        : ['completed'].includes(launchPhase.value)
          ? 'completed'
          : launchPhase.value === 'failed'
            ? 'failed'
            : 'pending',
    message: launchMessage.value,
  })

  return steps
})

const postProcessCards = computed(() => {
  const summary = activePayload.value?.run.summary ?? {}
  const exports = activePayload.value?.exports ?? []
  return [
    {
      label: '后处理概览',
      value: exports.length ? `${exports.length} 份导出` : '等待生成',
      note: exports.length ? '已生成统计表、图表或压缩包' : '执行完成后自动汇总导出结果',
    },
    {
      label: '统计摘要',
      value:
        summary && Object.keys(summary).length
          ? `${summary.vf ?? summary.avg_vf ?? summary.avg_volume_fraction ?? '--'}`
          : '--',
      note: summary.vf || summary.avg_vf || summary.avg_volume_fraction ? '当前可查看 Vf 或批量摘要' : '完成任务后在此展示摘要',
    },
  ]
})

export const useTaskWorkflow = () => ({
  initializing,
  initialized,
  projects,
  runners,
  selectedFiles,
  selectionSource,
  segmentationOptions,
  traditionalMethodOptions,
  measurementOptions,
  form,
  preprocessBackgroundOptions,
  preprocessDenoiseOptions,
  preprocessEnhanceOptions,
  calibrationProbe,
  calibrationProbeLoading,
  calibrationDialogVisible,
  calibrationScaleBarUm,
  calibrationScaleBarPixels,
  launchPhase,
  launchMessage,
  launchError,
  activeRunId,
  activePayload,
  actionLoading,
  uploadState,
  currentModeLabel,
  traditionalMethodHint,
  segmentationHint,
  preprocessBackgroundHint,
  preprocessDenoiseHint,
  preprocessEnhanceHint,
  preprocessSummary,
  preprocessSelectionValid,
  preprocessValidationMessage,
  postprocessSummary,
  measurementSummary,
  measurementCount,
  calibrationNeedsWarning,
  calibrationReminder,
  calibrationProbeSummary,
  calibrationComputedValue,
  calibrationCandidates,
  calibrationSuggestedValue,
  calibrationActionLabel,
  calibrationFooterHint,
  currentRunner,
  runnerNotes,
  inputModeLabel,
  workflowReady,
  canNavigateToConfig,
  canLaunch,
  currentProject,
  executionProgress,
  uploadProgressLabel,
  runStatusLabel,
  workflowSteps,
  postProcessCards,
  activeRunningStep,
  activeStepProgressText,
  initialize,
  setSelectedFiles,
  clearSelectedFiles,
  runCalibrationProbe,
  openCalibrationDialog,
  closeCalibrationDialog,
  applyCalibrationValue,
  applyPreprocessPreset,
  prepareCurrentTask,
  launchCurrentTask,
  loadActivePayload,
  restoreDraftRunFromSession,
  ensurePreprocessReady,
  startPolling,
  stopPolling,
  resetRunSession,
})
