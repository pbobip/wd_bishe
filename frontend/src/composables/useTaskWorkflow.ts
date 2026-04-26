import { ElMessage } from 'element-plus'
import { computed, reactive, ref } from 'vue'

import { api, formatApiError } from '../api'
import type { ModelRunner, RunResultsPayload } from '../types'
import {
  buildUmPerPxFromScaleBar,
  inspectCalibrationProbe,
  type CalibrationProbe,
} from '../utils/calibration'

export type FileSelectionSource = 'files' | 'folder'
type LaunchPhase = 'idle' | 'creating' | 'uploading' | 'ready' | 'starting' | 'running' | 'completed' | 'failed'
type PostprocessSmoothingMethod = 'mean' | 'gaussian' | 'median'
type TraditionalMethod = 'threshold' | 'adaptive' | 'edge' | 'clustering'
type TraditionalForegroundTarget = 'dark' | 'bright'
type PreprocessBackgroundMethod = 'none' | 'tophat' | 'rolling_ball'
type PreprocessDenoiseMethod = 'none' | 'wavelet' | 'gaussian' | 'median' | 'bilateral' | 'mean'
type PreprocessEnhanceMethod = 'none' | 'clahe' | 'hist_equalization' | 'gamma'
type ImageCalibrationSource = 'auto' | 'manual' | 'pixel' | 'none'
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
const PREVIEWABLE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
const UPLOAD_CHUNK_SIZE = 10
const ACTIVE_RUN_STORAGE_KEY = 'wd_bishe.active_run_id'

interface ImageCalibrationEntry {
  key: string
  file: File
  fileName: string
  relativePath: string
  originalUrl: string | null
  previewUrl: string | null
  previewable: boolean
  loading: boolean
  probe: CalibrationProbe | null
  umPerPx: number | null
  source: ImageCalibrationSource
  message: string
  error: string | null
}

const segmentationOptions = [
  { label: '传统分割', value: 'traditional' },
  { label: '深度学习', value: 'dl' },
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
  name: `任务-${new Date().toLocaleString()}`,
  input_mode: 'batch',
  segmentation_mode: 'traditional',
  input_config: {
    result_dir_name: 'default',
    um_per_px: undefined as number | undefined,
    auto_crop_sem_region: true,
    save_sem_footer: true,
    image_calibrations: [] as Array<{
      relative_path: string
      original_name?: string
      um_per_px?: number
    }>,
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
    foreground_target: 'dark' as TraditionalForegroundTarget,
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
    cluster_target: 'dark',
    fill_holes: false,
    watershed: false,
    boundary_smoothing: false,
    boundary_smoothing_kernel: 3,
    min_area: 1,
    max_area: undefined as number | undefined,
    min_solidity: 0.0,
    min_circularity: 0.0,
    min_roundness: 0.0,
    max_aspect_ratio: undefined as number | undefined,
    remove_border: false,
    open_kernel: 3,
    close_kernel: 3,
  },
  postprocess: {
    fill_holes: false,
    watershed: false,
    watershed_params: {
      separation: 35,
      background_iterations: 1,
      min_marker_area: 12,
    },
    smoothing: {
      enabled: false,
      method: 'gaussian' as PostprocessSmoothingMethod,
      kernel: 3,
    },
    shape_filter: {
      enabled: false,
      min_area: 30,
      max_area: undefined as number | undefined,
      min_solidity: 0.0,
      min_circularity: 0.0,
      min_roundness: 0.0,
      max_aspect_ratio: undefined as number | undefined,
    },
    morphology: {
      opening_enabled: false,
      opening_kernel: 3,
      closing_enabled: false,
      closing_kernel: 3,
    },
    remove_border: false,
  },
  dl_model: {
    model_slot: 'mbu_netpp',
    runner_id: undefined as number | undefined,
    weight_path: '',
    input_size: 256,
    device: 'auto',
    extra_params: {},
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
const runners = ref<ModelRunner[]>([])
const selectedFiles = ref<File[]>([])
const selectedImageEntries = ref<ImageCalibrationEntry[]>([])
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
let imageCalibrationSweepSeq = 0
let imageCalibrationSweepPromise: Promise<void> | null = null
let initPromise: Promise<void> | null = null
let pollTimer: number | null = null
let preparedSelectionKey: string | null = null

const toFinitePositiveNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return value
  }
  return null
}

const getFileRelativePath = (file: File) =>
  ((file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name).replace(/\\/g, '/')

const getSelectedFileKey = (file: File) => `${getFileRelativePath(file)}:${file.size}:${file.lastModified}`

const isPreviewableImage = (file: File) => {
  const lower = file.name.toLowerCase()
  return PREVIEWABLE_EXTENSIONS.some((ext) => lower.endsWith(ext))
}

const revokeSelectedImagePreviews = () => {
  if (typeof URL === 'undefined' || typeof URL.revokeObjectURL !== 'function') return
  const revoked = new Set<string>()
  selectedImageEntries.value.forEach((entry) => {
    const urls = [entry.originalUrl, entry.previewUrl].filter((value): value is string => Boolean(value?.startsWith('blob:')))
    urls.forEach((url) => {
      if (revoked.has(url)) return
      URL.revokeObjectURL(url)
      revoked.add(url)
    })
  })
}

const buildConfiguredCalibrationMap = () => {
  const mapping = new Map<string, number>()
  const configured = Array.isArray(form.input_config.image_calibrations) ? form.input_config.image_calibrations : []
  configured.forEach((item) => {
    const relativePath =
      typeof item?.relative_path === 'string' ? item.relative_path.trim().replace(/\\/g, '/') : ''
    const value = toFinitePositiveNumber(item?.um_per_px)
    if (relativePath && value !== null) {
      mapping.set(relativePath, value)
    }
  })
  return mapping
}

const createImageCalibrationEntry = (file: File, configuredCalibrationMap: Map<string, number>, defaultCalibration: number | null) => {
  const relativePath = getFileRelativePath(file)
  const configuredCalibration = configuredCalibrationMap.get(relativePath) ?? defaultCalibration
  const previewable = isPreviewableImage(file)
  const originalUrl =
    previewable && typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function'
      ? URL.createObjectURL(file)
      : null
  const previewUrl = previewable ? originalUrl : null
  return {
    key: getSelectedFileKey(file),
    file,
    fileName: file.name,
    relativePath,
    originalUrl,
    previewUrl,
    previewable,
    loading: false,
    probe: null,
    umPerPx: configuredCalibration ?? null,
    source: configuredCalibration != null ? 'manual' : 'none',
    message: configuredCalibration != null ? `当前使用 ${configuredCalibration.toFixed(6)} um/px` : '等待自动识别',
    error: null,
  } satisfies ImageCalibrationEntry
}

const findImageCalibrationEntry = (entryKey: string) =>
  selectedImageEntries.value.find((entry) => entry.key === entryKey) ?? null

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

const RUNNERS_STORAGE_KEY = 'wd_bishe.model_runners'
const RUNNERS_TTL_KEY = 'wd_bishe.model_runners_ttl'
const RUNNERS_TTL_MS = 5 * 60 * 1000 // 5 minutes

const loadReferenceData = async () => {
  // Try sessionStorage cache first to avoid redundant API call on startup
  try {
    const cached = sessionStorage.getItem(RUNNERS_STORAGE_KEY)
    const ttl = sessionStorage.getItem(RUNNERS_TTL_KEY)
    if (cached && ttl && Date.now() < parseInt(ttl, 10)) {
      const parsed = JSON.parse(cached)
      if (Array.isArray(parsed)) {
        runners.value = parsed
        return
      }
    }
  } catch {
    // sessionStorage unavailable or corrupted, fall through to API
  }

  const [runnerRes] = await Promise.all([
    api.get<ModelRunner[]>('/model-runners'),
  ])
  runners.value = runnerRes.data

  // Cache in sessionStorage
  try {
    sessionStorage.setItem(RUNNERS_STORAGE_KEY, JSON.stringify(runnerRes.data))
    sessionStorage.setItem(RUNNERS_TTL_KEY, String(Date.now() + RUNNERS_TTL_MS))
  } catch {
    // ignore storage quota errors
  }
}

const initialize = async () => {
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
    return '调用 MBU-Net++ 主模型或其他已配置运行器做自动分割。'
  }
  return '请选择传统分割或深度学习路线。'
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

type PreprocessPreset = 'traditional' | 'dl'

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

}

const postprocessSummary = computed(() => {
  const parts: string[] = []
  if (form.postprocess.fill_holes) parts.push('填孔')
  if (form.postprocess.watershed) parts.push('watershed')
  if (form.postprocess.smoothing.enabled) {
    parts.push(`平滑(${form.postprocess.smoothing.method}/${form.postprocess.smoothing.kernel})`)
  }
  if (form.postprocess.morphology.opening_enabled || form.postprocess.shape_filter.enabled) parts.push('去除杂点')
  if (form.postprocess.morphology.closing_enabled) parts.push('缝隙闭合')
  if (form.postprocess.remove_border) parts.push('触边剔除')
  return parts.length ? `当前后处理流程：${parts.join(' → ')}` : '当前未加入后处理步骤'
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

const calibratedImageCount = computed(
  () => selectedImageEntries.value.filter((entry) => toFinitePositiveNumber(entry.umPerPx) !== null).length,
)
const calibrationPendingCount = computed(() => selectedImageEntries.value.filter((entry) => entry.loading).length)
const pxFallbackImageCount = computed(() =>
  selectedImageEntries.value.filter((entry) => !entry.loading && toFinitePositiveNumber(entry.umPerPx) === null).length,
)
const calibrationNeedsWarning = computed(
  () => form.stats.enabled && selectedFiles.value.length > 0 && calibratedImageCount.value === 0,
)
const calibrationReminder = computed(() => {
  if (!selectedFiles.value.length) return '尚未选择图像'
  if (calibrationPendingCount.value) {
    return `正在自动识别 ${calibrationPendingCount.value} 张图像的标定信息。`
  }
  if (!calibratedImageCount.value) {
    return '当前没有自动标定结果，可手动填写 um/px。'
  }
  if (!pxFallbackImageCount.value) {
    return `当前 ${calibratedImageCount.value}/${selectedImageEntries.value.length} 张图像均已标定。`
  }
  return `当前已标定 ${calibratedImageCount.value}/${selectedImageEntries.value.length} 张；其余图像可手动填写或按像素统计。`
})
const calibrationProbeSummary = computed(() => calibrationReminder.value)
const calibrationComputedValue = computed(() =>
  buildUmPerPxFromScaleBar(calibrationScaleBarUm.value, calibrationScaleBarPixels.value),
)
const calibrationCandidates = computed(() => calibrationProbe.value?.common_scale_candidates ?? [])
const calibrationSuggestedValue = computed(() => calibrationProbe.value?.suggested_um_per_px ?? null)
const calibrationActionLabel = computed(() => (calibrationPendingCount.value ? '识别中...' : '重新识别比例尺'))
const calibrationFooterHint = computed(() => {
  if (!selectedFiles.value.length) return '如图像没有底部信息栏，运行时会自动跳过这一步。'
  return '当前默认自动裁分析区域并保存可用底栏信息；如单张图识别失败，可直接在该图下手动补充标定值。'
})

const syncPrimaryCalibrationProbe = () => {
  const firstEntry = selectedImageEntries.value[0] ?? null
  calibrationProbe.value = firstEntry?.probe ?? null
  calibrationProbeLoading.value = Boolean(firstEntry?.loading)
  calibrationScaleBarUm.value = firstEntry?.probe?.ocr_scale_bar_um ?? null
  calibrationScaleBarPixels.value = firstEntry?.probe?.scale_bar_pixels ?? null
}

const setImageCalibrationValue = (
  entry: ImageCalibrationEntry,
  value: number | null,
  source: ImageCalibrationSource,
  message?: string,
) => {
  const normalized = toFinitePositiveNumber(value)
  entry.umPerPx = normalized
  entry.source = normalized !== null ? source : source === 'pixel' ? 'pixel' : 'none'
  if (message) {
    entry.message = message
  } else if (normalized !== null) {
    entry.message = `${source === 'auto' ? '已自动识别' : '当前使用'} ${normalized.toFixed(6)} um/px`
  } else if (source === 'pixel') {
    entry.message = '当前按像素统计。'
  } else {
    entry.message = '未自动标定，请手动填写 um/px。'
  }
}

const inspectImageCalibrationEntry = async (
  entry: ImageCalibrationEntry,
  options?: { force?: boolean; sweepSeq?: number },
) => {
  if (!options?.force && entry.probe) {
    return entry.probe
  }

  entry.loading = true
  entry.error = null
  entry.message = '正在自动识别...'
  if (selectedImageEntries.value[0]?.key === entry.key) {
    calibrationProbeLoading.value = true
  }

  try {
    const probe = await inspectCalibrationProbe([entry.file])
    if (options?.sweepSeq && options.sweepSeq !== imageCalibrationSweepSeq) {
      return entry.probe
    }
    entry.probe = probe
    if (probe?.preview_url && !entry.previewUrl) {
      entry.previewUrl = probe.preview_url
    }
    if (probe?.suggested_um_per_px && entry.source !== 'manual') {
      setImageCalibrationValue(entry, probe.suggested_um_per_px, 'auto', `已自动识别 ${probe.suggested_um_per_px.toFixed(6)} um/px`)
    } else if (probe?.suggested_um_per_px && entry.source === 'manual' && entry.umPerPx !== null) {
      entry.message = `已保留手动输入 ${entry.umPerPx.toFixed(6)} um/px`
    } else if (entry.source !== 'manual') {
      setImageCalibrationValue(
        entry,
        null,
        entry.source === 'pixel' ? 'pixel' : 'none',
        entry.source === 'pixel' ? '当前按像素统计。' : probe?.message,
      )
    } else if (entry.umPerPx !== null) {
      entry.message = `当前使用 ${entry.umPerPx.toFixed(6)} um/px`
    }
    if (!probe && entry.source !== 'manual' && entry.source !== 'pixel') {
      entry.message = '未自动标定，请手动填写 um/px。'
    }
    return probe
  } finally {
    if (!options?.sweepSeq || options.sweepSeq === imageCalibrationSweepSeq) {
      entry.loading = false
      syncPrimaryCalibrationProbe()
    }
  }
}

const startImageCalibrationSweep = () => {
  if (!selectedImageEntries.value.length) {
    imageCalibrationSweepPromise = null
    return
  }
  const sweepSeq = ++imageCalibrationSweepSeq
  const snapshot = [...selectedImageEntries.value]
  imageCalibrationSweepPromise = (async () => {
    for (const entry of snapshot) {
      if (sweepSeq !== imageCalibrationSweepSeq) return
      await inspectImageCalibrationEntry(entry, { sweepSeq })
    }
  })().finally(() => {
    if (sweepSeq === imageCalibrationSweepSeq) {
      imageCalibrationSweepPromise = null
      syncPrimaryCalibrationProbe()
    }
  })
}

const waitForImageCalibrationSweep = async () => {
  if (imageCalibrationSweepPromise) {
    await imageCalibrationSweepPromise
  }
}

const updateImageCalibrationValue = (entryKey: string, value: number | null | undefined) => {
  const entry = findImageCalibrationEntry(entryKey)
  if (!entry) return
  setImageCalibrationValue(entry, value ?? null, 'manual')
  entry.error = null
  syncPrimaryCalibrationProbe()
}

const clearImageCalibrationValue = (entryKey: string) => {
  const entry = findImageCalibrationEntry(entryKey)
  if (!entry) return
  setImageCalibrationValue(entry, null, 'pixel')
  entry.error = null
  syncPrimaryCalibrationProbe()
}

const runImageCalibrationProbe = async (entryKey: string, force = true) => {
  const entry = findImageCalibrationEntry(entryKey)
  if (!entry) return null
  const probeTask = inspectImageCalibrationEntry(entry, { force })
  const waitTask = probeTask.then(() => undefined)
  imageCalibrationSweepPromise = waitTask
  waitTask.finally(() => {
    if (imageCalibrationSweepPromise === waitTask) {
      imageCalibrationSweepPromise = null
    }
  })
  return probeTask
}

const getImageCalibrationStatusLabel = (entry: ImageCalibrationEntry) => {
  if (entry.loading) return '识别中'
  if (entry.umPerPx !== null && entry.source === 'auto') return '已自动标定'
  if (entry.umPerPx !== null) return '已手动填写'
  if (entry.source === 'pixel') return '按像素统计'
  return '待手动填写'
}

const getImageCalibrationStatusTone = (entry: ImageCalibrationEntry) => {
  if (entry.loading) return 'pending'
  if (entry.umPerPx !== null) return 'success'
  if (entry.source === 'pixel') return 'neutral'
  return 'warning'
}

const getImageCalibrationDetail = (entry: ImageCalibrationEntry) => {
  if (entry.loading) return '正在自动识别标定信息。'
  if (entry.umPerPx !== null) {
    return `${entry.source === 'auto' ? '自动识别' : '手动输入'}：${entry.umPerPx.toFixed(6)} um/px`
  }
  if (entry.source === 'pixel') {
    return '当前按像素单位统计。'
  }
  if (entry.probe?.scale_bar_detected) {
    return `检测到比例尺横线 ${entry.probe.scale_bar_pixels ?? '--'} px，请填写对应的 μm 数值。`
  }
  return '未自动标定，请手动填写 um/px。'
}

const getImageCalibrationMeta = (entry: ImageCalibrationEntry) => {
  const meta: string[] = []
  if (entry.probe?.background_cropped) meta.push('已裁分析区')
  if (entry.umPerPx !== null && entry.probe?.ocr_scale_bar_um) meta.push(`OCR ${entry.probe.ocr_scale_bar_um} μm`)
  if (entry.umPerPx !== null && entry.probe?.ocr_magnification_text) meta.push(String(entry.probe.ocr_magnification_text))
  return meta
}

const shouldShowImageRelativePath = (entry: ImageCalibrationEntry) => {
  const normalizedPath = entry.relativePath.trim().replace(/\\/g, '/')
  const normalizedName = entry.fileName.trim()
  return Boolean(normalizedPath && normalizedPath !== normalizedName)
}

const isImageCalibrationPixelMode = (entry: ImageCalibrationEntry) => entry.source === 'pixel'

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
    '深度学习模式会调用对应环境执行推理',
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
const canOpenResultWorkspace = computed(() => Boolean(activeRunId.value))
const canOpenStatistics = computed(() =>
  Boolean(
    activeRunId.value
    && activePayload.value
    && ['completed', 'partial_success'].includes(activePayload.value.run.status),
  ),
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
    launchMessage.value = status === 'partial_success' ? '主分割已完成，但有部分模式执行失败' : '主分割已完成，可进入结果页核查'
  } else if (status === 'failed') {
    launchPhase.value = 'failed'
    launchMessage.value = response.data.run.error_message || '任务执行失败'
  } else if (status === 'draft') {
    launchPhase.value = 'ready'
    launchMessage.value = '草稿任务已恢复，可继续启动主分割'
  } else {
    launchPhase.value = 'running'
    launchMessage.value = runningStep?.details?.message || runningStep?.message || '主分割执行中，正在刷新步骤状态'
  }
}

let restorePromise: Promise<boolean> | null = null

const restoreDraftRunFromSession = async () => {
  if (activeRunId.value) {
    return true
  }
  if (!restorePromise) {
    restorePromise = (async () => {
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
    })()
  }
  return restorePromise
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

const runCalibrationProbe = async (force = false) => {
  const firstEntry = selectedImageEntries.value[0]
  if (!firstEntry) {
    calibrationProbe.value = null
    return null
  }
  const requestSeq = ++calibrationProbeSeq
  const probe = await runImageCalibrationProbe(firstEntry.key, force)
  if (requestSeq === calibrationProbeSeq) {
    syncPrimaryCalibrationProbe()
  }
  return probe
}

const setSelectedFiles = async (files: File[], source: FileSelectionSource) => {
  resetRunSession()
  const accepted = files.filter((file) => {
    const lower = file.name.toLowerCase()
    return SUPPORTED_EXTENSIONS.some((ext) => lower.endsWith(ext))
  })
  const skipped = files.length - accepted.length
  imageCalibrationSweepSeq += 1
  imageCalibrationSweepPromise = null
  revokeSelectedImagePreviews()
  selectionSource.value = source
  selectedFiles.value = accepted
  const configuredCalibrationMap = buildConfiguredCalibrationMap()
  const defaultCalibration = toFinitePositiveNumber(form.input_config.um_per_px)
  selectedImageEntries.value = accepted.map((file) =>
    createImageCalibrationEntry(file, configuredCalibrationMap, defaultCalibration),
  )
  form.input_mode = source === 'folder' || accepted.length > 1 ? 'batch' : 'single'
  form.input_config.um_per_px = undefined
  form.input_config.auto_crop_sem_region = true
  form.input_config.save_sem_footer = true
  form.input_config.image_calibrations = []
  calibrationProbe.value = null
  calibrationScaleBarPixels.value = null
  calibrationScaleBarUm.value = null
  calibrationProbeLoading.value = false
  if (accepted.length) {
    startImageCalibrationSweep()
  }
  if (skipped > 0) {
    ElMessage.warning(`已跳过 ${skipped} 个非图像文件，仅保留 png/jpg/jpeg/tif/tiff`)
  }
}

const clearSelectedFiles = () => {
  resetRunSession()
  imageCalibrationSweepSeq += 1
  imageCalibrationSweepPromise = null
  revokeSelectedImagePreviews()
  selectedFiles.value = []
  selectedImageEntries.value = []
  selectionSource.value = 'files'
  form.input_mode = 'batch'
  form.input_config.um_per_px = undefined
  form.input_config.image_calibrations = []
  calibrationProbe.value = null
  calibrationScaleBarPixels.value = null
  calibrationScaleBarUm.value = null
  calibrationProbeLoading.value = false
  calibrationProbeSeq += 1
}

const openCalibrationDialog = async () => {
  calibrationDialogVisible.value = true
  if (!calibrationProbe.value && selectedImageEntries.value.length) {
    await runCalibrationProbe()
  }
  calibrationScaleBarUm.value = calibrationProbe.value?.ocr_scale_bar_um ?? calibrationScaleBarUm.value
  calibrationScaleBarPixels.value = calibrationProbe.value?.scale_bar_pixels ?? calibrationScaleBarPixels.value
}

const closeCalibrationDialog = () => {
  calibrationDialogVisible.value = false
}

const applyCalibrationValue = (value: number | null) => {
  const firstEntry = selectedImageEntries.value[0]
  if (!firstEntry) return
  const normalized = toFinitePositiveNumber(value)
  if (normalized !== null) {
    setImageCalibrationValue(firstEntry, normalized, 'manual')
    syncPrimaryCalibrationProbe()
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
  selectedFiles.value.map((file) => getSelectedFileKey(file)).join('|')

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
      multipart.append('relative_paths', getFileRelativePath(file))
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
  form.input_config.um_per_px = undefined
  form.input_config.auto_crop_sem_region = true
  form.input_config.save_sem_footer = true
  form.input_config.image_calibrations = selectedImageEntries.value
    .map((entry) => ({
      relative_path: entry.relativePath,
      original_name: entry.fileName,
      um_per_px: toFinitePositiveNumber(entry.umPerPx) ?? undefined,
    }))
    .filter((entry) => typeof entry.um_per_px === 'number')
  if (!form.preprocess.enabled) {
    form.preprocess.background.method = 'none'
    form.preprocess.denoise.method = 'none'
    form.preprocess.enhance.method = 'none'
    form.preprocess.extras.unsharp = false
  }
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
    launchMessage.value = '正在确认逐图标定状态'
    await waitForImageCalibrationSweep()
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
      launchMessage.value = '图像已登记，可开始主分割并进入结果页核查'
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
    launchMessage.value = '图像已登记，可开始主分割并进入结果页核查'
    ElMessage.success('图像已上传并登记到任务草稿')
    return true
  } catch (error: any) {
    launchPhase.value = 'failed'
    launchError.value = formatApiError(error, '任务创建失败')
    launchMessage.value = launchError.value
    ElMessage.error(launchError.value)
    return false
  } finally {
    actionLoading.value = false
  }
}

const ensurePostprocessDraftReady = async () => {
  if (!workflowReady.value) {
    ElMessage.warning('请先填写任务名称并导入图像')
    return false
  }
  return prepareCurrentTask()
}

const startTaskFromCreatePage = async () => {
  if (!workflowReady.value) {
    ElMessage.warning('请先填写任务名称并导入图像')
    return false
  }
  const prepared = await prepareCurrentTask()
  if (!prepared) {
    return false
  }
  return launchCurrentTask()
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
    launchMessage.value = '正在同步任务配置'
    await syncDraftRun(activeRunId.value)

    launchMessage.value = '图像已就绪，正在启动主分割'
    await api.post(`/runs/${activeRunId.value}/execute`)

    launchPhase.value = 'running'
    launchMessage.value = '主分割已启动，正在获取执行状态'
    await loadActivePayload()
    const payload = activePayload.value as RunResultsPayload | null
    const currentStatus = payload?.run.status
    if (!currentStatus || !['completed', 'failed', 'partial_success'].includes(currentStatus)) {
      startPolling()
    }
    ElMessage.success('主分割已启动，可进入结果页查看执行状态')
    return true
  } catch (error: any) {
    launchPhase.value = 'failed'
    launchError.value = formatApiError(error, '任务启动失败')
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
  runners,
  selectedFiles,
  selectedImageEntries,
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
  calibratedImageCount,
  calibrationPendingCount,
  pxFallbackImageCount,
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
  canOpenResultWorkspace,
  canOpenStatistics,
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
  runImageCalibrationProbe,
  openCalibrationDialog,
  closeCalibrationDialog,
  applyCalibrationValue,
  updateImageCalibrationValue,
  clearImageCalibrationValue,
  getImageCalibrationStatusLabel,
  getImageCalibrationStatusTone,
  getImageCalibrationDetail,
  getImageCalibrationMeta,
  shouldShowImageRelativePath,
  isImageCalibrationPixelMode,
  applyPreprocessPreset,
  ensurePostprocessDraftReady,
  startTaskFromCreatePage,
  prepareCurrentTask,
  launchCurrentTask,
  loadActivePayload,
  restoreDraftRunFromSession,
  ensurePreprocessReady,
  startPolling,
  stopPolling,
  resetRunSession,
})
