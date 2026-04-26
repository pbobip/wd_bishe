<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { api } from '../api'
import CalibrationStatusBanner from '../components/CalibrationStatusBanner.vue'
import PostprocessPreviewDialog from '../components/PostprocessPreviewDialog.vue'
import PreviewWorkspace from '../components/PreviewWorkspace.vue'
import ZoomableImageDialog from '../components/ZoomableImageDialog.vue'
import { useTaskWorkflow } from '../composables/useTaskWorkflow'
import type {
  PostprocessConfirmPayload,
  PostprocessPreviewPayload,
  RunResultsPayload,
  RunStep,
} from '../types'

const route = useRoute()
const router = useRouter()
const workflow = useTaskWorkflow()

const runId = computed(() => Number(route.params.id))
const payload = ref<RunResultsPayload | null>(null)
const loading = ref(true)
const stepExpanded = ref(true)
const selectedImageId = ref<number | null>(null)
const selectedMode = ref<string | null>(null)
const previewLoading = ref(false)
const confirmLoading = ref(false)
const previewDialogVisible = ref(false)
const previewPayload = ref<PostprocessPreviewPayload | null>(null)
const hydratedPostprocessRunId = ref<number | null>(null)
const navigationInProgress = ref(false)

type PostprocessMethodKey =
  | 'fill_holes'
  | 'noise_cleanup'
  | 'smooth_boundary'
  | 'gap_closing'
  | 'separate_touching'
  | 'remove_edge'

const methodDialogVisible = ref(false)
const activePostprocessMethod = ref<PostprocessMethodKey>('fill_holes')
const methodPreviewLoading = ref(false)
const methodPreviewPayload = ref<PostprocessPreviewPayload | null>(null)
const methodPreviewError = ref<string | null>(null)
const methodPreviewPhase = ref<'before' | 'after'>('after')
const methodOpeningKey = ref<PostprocessMethodKey | null>(null)
const methodConfigSnapshot = ref<Record<string, any> | null>(null)
const zoomPreviewVisible = ref(false)
const zoomPreviewTitle = ref('后处理步骤预览')
const zoomPreviewSubtitle = ref('')
const zoomPreviewImageUrl = ref<string | null>(null)
let methodPreviewTimer: number | null = null
let methodPreviewSeq = 0

let timer: number | null = null

const STEP_LABELS: Record<string, string> = {
  input: '导入整理',
  preprocess: '预处理',
  traditional: '传统分割',
  dl: '深度学习',
  stats: '统计汇总',
  export: '导出结果',
}

const STEP_STATUS_LABELS: Record<string, string> = {
  completed: '已完成',
  running: '执行中',
  failed: '失败',
  pending: '待执行',
  queued: '排队中',
}

const INPUT_MODE_LABELS: Record<string, string> = {
  single: '单张输入',
  batch: '批量输入',
  folder: '文件夹输入',
}

const SEGMENTATION_MODE_LABELS: Record<string, string> = {
  traditional: '传统分割',
  dl: '深度学习',
}

const createEmptyPostprocessConfig = () => ({
  fill_holes: false,
  watershed: false,
  watershed_params: {
    separation: 35,
    background_iterations: 1,
    min_marker_area: 12,
  },
  remove_border: false,
  smoothing: {
    enabled: false,
    method: 'gaussian',
    kernel: 3,
  },
  shape_filter: {
    enabled: false,
    min_area: 30,
    max_area: undefined,
    min_solidity: 0.0,
    min_circularity: 0.0,
    min_roundness: 0.0,
    max_aspect_ratio: undefined,
  },
  morphology: {
    opening_enabled: false,
    opening_kernel: 3,
    closing_enabled: false,
    closing_kernel: 3,
  },
})

const modeLabel = (mode: string) => SEGMENTATION_MODE_LABELS[mode] ?? mode

const formatDateTime = (value?: string | null) => {
  if (!value) return '--'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.valueOf())) return value
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
    .format(parsed)
    .replace(/\//g, '-')
}

const stopPolling = () => {
  if (timer) {
    window.clearInterval(timer)
    timer = null
  }
}

const resetPreviewState = () => {
  previewDialogVisible.value = false
  previewPayload.value = null
  previewLoading.value = false
  confirmLoading.value = false
}

const syncSelection = (nextPayload: RunResultsPayload, preserveSelection = true) => {
  const images = nextPayload.images
  if (!images.length) {
    selectedImageId.value = null
    selectedMode.value = null
    return
  }

  const previousImageId = preserveSelection ? selectedImageId.value : null
  const previousMode = preserveSelection ? selectedMode.value : null
  const nextImage =
    images.find((image) => image.image_id === previousImageId) ??
    images[0]

  selectedImageId.value = nextImage.image_id

  const modes = Object.keys(nextImage.modes ?? {})
  if (!modes.length) {
    selectedMode.value = null
    return
  }

  if (previousMode && modes.includes(previousMode)) {
    selectedMode.value = previousMode
    return
  }

  if (modes.includes(nextPayload.run.segmentation_mode)) {
    selectedMode.value = nextPayload.run.segmentation_mode
    return
  }

  selectedMode.value = modes[0]
}

const hydratePostprocessConfig = (nextPayload: RunResultsPayload) => {
  if (hydratedPostprocessRunId.value === nextPayload.run.id) return

  const postprocessApplied = Boolean(nextPayload.run.summary?.postprocess_applied)
  const config = nextPayload.run.config?.postprocess as Record<string, any> | undefined
  if (!postprocessApplied || !config || typeof config !== 'object') {
    restorePostprocessConfig(createEmptyPostprocessConfig())
    hydratedPostprocessRunId.value = nextPayload.run.id
    return
  }

  if (config && typeof config === 'object') {
    workflow.form.postprocess.fill_holes = Boolean(config.fill_holes)
    workflow.form.postprocess.watershed = Boolean(config.watershed)
    workflow.form.postprocess.remove_border = Boolean(config.remove_border)

    const watershedParams = config.watershed_params as Record<string, any> | undefined
    if (watershedParams && typeof watershedParams === 'object') {
      if (typeof watershedParams.separation === 'number') {
        workflow.form.postprocess.watershed_params.separation = watershedParams.separation
      }
      if (typeof watershedParams.background_iterations === 'number') {
        workflow.form.postprocess.watershed_params.background_iterations = watershedParams.background_iterations
      }
      if (typeof watershedParams.min_marker_area === 'number') {
        workflow.form.postprocess.watershed_params.min_marker_area = watershedParams.min_marker_area
      }
    }

    const smoothing = config.smoothing as Record<string, any> | undefined
    if (smoothing && typeof smoothing === 'object') {
      workflow.form.postprocess.smoothing.enabled = Boolean(smoothing.enabled)
      if (typeof smoothing.method === 'string') {
        workflow.form.postprocess.smoothing.method = smoothing.method as typeof workflow.form.postprocess.smoothing.method
      }
      if (typeof smoothing.kernel === 'number') {
        workflow.form.postprocess.smoothing.kernel = smoothing.kernel
      }
    }

    const shapeFilter = config.shape_filter as Record<string, any> | undefined
    if (shapeFilter && typeof shapeFilter === 'object') {
      workflow.form.postprocess.shape_filter.enabled = Boolean(shapeFilter.enabled)
      workflow.form.postprocess.shape_filter.min_area = Number(shapeFilter.min_area ?? workflow.form.postprocess.shape_filter.min_area)
      workflow.form.postprocess.shape_filter.max_area = shapeFilter.max_area ?? undefined
      workflow.form.postprocess.shape_filter.min_solidity = Number(shapeFilter.min_solidity ?? workflow.form.postprocess.shape_filter.min_solidity)
      workflow.form.postprocess.shape_filter.min_circularity = Number(shapeFilter.min_circularity ?? workflow.form.postprocess.shape_filter.min_circularity)
      workflow.form.postprocess.shape_filter.min_roundness = Number(shapeFilter.min_roundness ?? workflow.form.postprocess.shape_filter.min_roundness)
      workflow.form.postprocess.shape_filter.max_aspect_ratio = shapeFilter.max_aspect_ratio ?? undefined
    }

    const morphology = config.morphology as Record<string, any> | undefined
    if (morphology && typeof morphology === 'object') {
      workflow.form.postprocess.morphology.opening_enabled = Boolean(morphology.opening_enabled)
      workflow.form.postprocess.morphology.opening_kernel = Number(morphology.opening_kernel ?? workflow.form.postprocess.morphology.opening_kernel)
      workflow.form.postprocess.morphology.closing_enabled = Boolean(morphology.closing_enabled)
      workflow.form.postprocess.morphology.closing_kernel = Number(morphology.closing_kernel ?? workflow.form.postprocess.morphology.closing_kernel)
    }
  }

  hydratedPostprocessRunId.value = nextPayload.run.id
}

const loadPayload = async (preserveSelection = true) => {
  const response = await api.get<RunResultsPayload>(`/runs/${runId.value}/results`)
  payload.value = response.data
  hydratePostprocessConfig(response.data)
  syncSelection(response.data, preserveSelection)
}

const startPolling = () => {
  stopPolling()
  timer = window.setInterval(async () => {
    try {
      await loadPayload()
      const status = payload.value?.run.status
      if (status && ['completed', 'failed', 'partial_success'].includes(status)) {
        stopPolling()
      }
    } catch {
      stopPolling()
    }
  }, 2500)
}

const stepIsInactive = (stepKey: string) => {
  const mode = payload.value?.run.segmentation_mode
  if (mode === 'traditional' && stepKey === 'dl') return true
  if (mode === 'dl' && stepKey === 'traditional') return true
  return false
}

const stepLabel = (stepKey: string) => STEP_LABELS[stepKey] ?? stepKey
const stepStatusLabel = (step: RunStep) => {
  if (stepIsInactive(step.step_key)) return '未启用'
  return STEP_STATUS_LABELS[step.status] ?? step.status
}

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

const stepCardClass = (step: RunStep) => ({
  'is-completed': step.status === 'completed',
  'is-running': step.status === 'running' || step.status === 'queued',
  'is-failed': step.status === 'failed',
  'is-pending': step.status === 'pending',
  'is-inactive': stepIsInactive(step.step_key),
})

const stepProgressCopy = (step: RunStep) => {
  if (stepIsInactive(step.step_key)) return null
  const details = step.details
  if (details?.total_images) {
    return `${details.processed_images}/${details.total_images}`
  }
  if (step.status === 'completed') return '完成'
  if (step.status === 'failed') return '失败'
  if (step.status === 'running' || step.status === 'queued') return '处理中'
  return null
}

const stepMessageCopy = (step: RunStep) => {
  if (stepIsInactive(step.step_key)) return '本次任务未启用该模式'
  let message = step.message?.trim() || '等待执行'
  const details = step.details
  if (details?.processed_images != null && details?.total_images != null) {
    const progressPattern = new RegExp(`[（(]\\s*已处理\\s*${details.processed_images}\\s*\\/\\s*${details.total_images}\\s*[)）]`, 'g')
    message = message.replace(progressPattern, '').trim()
  } else {
    message = message.replace(/[（(]\s*已处理\s*\d+\s*\/\s*\d+\s*[)）]/g, '').trim()
  }
  const currentImageName = details?.current_image_name?.trim()
  if (currentImageName) {
    const imagePattern = new RegExp(`\\s*[·•|-]\\s*${escapeRegExp(currentImageName)}$`)
    message = message.replace(imagePattern, '').trim()
  }
  return message.replace(/\s*[·•|-]\s*$/, '').trim() || '等待执行'
}

const stepImageCopy = (step: RunStep) => {
  if (stepIsInactive(step.step_key)) return null
  const details = step.details
  const currentImageName = details?.current_image_name?.trim()
  const totalImages = details?.total_images ?? 0
  if (totalImages > 1) {
    if ((step.status === 'running' || step.status === 'queued') && currentImageName) {
      return `当前：${currentImageName}`
    }
    return `共 ${totalImages} 张图像`
  }
  if ((step.status === 'running' || step.status === 'queued') && currentImageName) {
    return currentImageName
  }
  return null
}

const detailSummaryItems = computed(() => {
  if (!payload.value) return []
  const run = payload.value.run
  return [
    { label: '输入方式', value: INPUT_MODE_LABELS[run.input_mode] ?? run.input_mode },
    { label: '分割模式', value: SEGMENTATION_MODE_LABELS[run.segmentation_mode] ?? run.segmentation_mode },
    { label: '图像数', value: `${payload.value.images.length} 张` },
    { label: '当前状态', value: run.status },
  ]
})

const selectedImage = computed(
  () => payload.value?.images.find((image) => image.image_id === selectedImageId.value) ?? null,
)

const canOpenStatistics = computed(() =>
  Boolean(payload.value && ['completed', 'partial_success'].includes(payload.value.run.status)),
)

const canPreviewPostprocess = computed(() =>
  Boolean(
    payload.value
    && ['completed', 'partial_success'].includes(payload.value.run.status)
    && selectedImage.value
    && selectedMode.value
    && ['traditional', 'dl'].includes(selectedMode.value),
  ),
)

const postprocessLockedReason = computed(() => {
  if (!payload.value) return '正在读取当前任务结果。'
  if (payload.value.run.status === 'running' || payload.value.run.status === 'queued') {
    return '主分割仍在执行中，先核查当前结果，待运行结束后再应用后处理。'
  }
  if (payload.value.run.status === 'draft') {
    return '当前任务还没有开始处理，请先回到任务创建页启动主分割。'
  }
  if (payload.value.run.status === 'failed') {
    return '当前任务未生成可确认结果，暂时不能应用后处理。'
  }
  if (!selectedImage.value || !selectedMode.value) {
    return '请先在结果区选择图像和分割模式。'
  }
  return '会先基于当前选中图像生成一次性预览，只有确认后才覆盖正式结果。'
})

const postprocessOverview = computed(() => [
  {
    label: '当前模式',
    value: selectedMode.value ? modeLabel(selectedMode.value) : '等待选择',
  },
  {
    label: '当前图像',
    value: selectedImage.value?.image_name ?? '等待选择',
  },
  {
    label: '影响范围',
    value: payload.value ? `${payload.value.images.length} 张图像` : '--',
  },
])

const postprocessMethods = computed(() => [
  {
    key: 'fill_holes' as const,
    label: '填补空洞',
    enabled: workflow.form.postprocess.fill_holes,
  },
  {
    key: 'noise_cleanup' as const,
    label: '去除杂点',
    enabled: workflow.form.postprocess.morphology.opening_enabled || workflow.form.postprocess.shape_filter.enabled,
  },
  {
    key: 'smooth_boundary' as const,
    label: '边界平滑',
    enabled: workflow.form.postprocess.smoothing.enabled,
  },
  {
    key: 'gap_closing' as const,
    label: '缝隙闭合',
    enabled: workflow.form.postprocess.morphology.closing_enabled,
  },
  {
    key: 'separate_touching' as const,
    label: '粘连分离',
    enabled: workflow.form.postprocess.watershed,
  },
  {
    key: 'remove_edge' as const,
    label: '触边剔除',
    enabled: workflow.form.postprocess.remove_border,
  },
])

const activeMethod = computed(
  () => postprocessMethods.value.find((method) => method.key === activePostprocessMethod.value) ?? postprocessMethods.value[0],
)

const methodSliderMarks = {
  opening: { 1: '弱', 9: '中', 15: '强' },
  areaRange: { 1: '1', 120: '120', 500: '500', 1200: '不限' },
  quality: { 0: '关', 0.5: '中', 0.9: '严' },
  smooth: { 1: '轻', 9: '中', 15: '强' },
  closing: { 1: '轻', 11: '中', 21: '强' },
  watershed: { 5: '低', 45: '中', 85: '高' },
  background: { 1: '少', 3: '中', 5: '多' },
  markerArea: { 0: '细', 36: '中', 120: '稳' },
} as const

const noiseCleanupAreaRange = computed({
  get: (): number[] => [
    workflow.form.postprocess.shape_filter.min_area,
    workflow.form.postprocess.shape_filter.max_area ?? 1200,
  ],
  set: (value: number[]) => {
    const [minArea = 1, maxArea = 1200] = value
    workflow.form.postprocess.shape_filter.min_area = Math.max(1, Math.round(minArea))
    workflow.form.postprocess.shape_filter.max_area = maxArea >= 1200 ? undefined : Math.max(Math.round(maxArea), workflow.form.postprocess.shape_filter.min_area)
  },
})

const wasMethodEnabledInSnapshot = (key: PostprocessMethodKey, config: Record<string, any> | null) => {
  if (!config) return false
  if (key === 'fill_holes') return Boolean(config.fill_holes)
  if (key === 'noise_cleanup') return Boolean(config.morphology?.opening_enabled || config.shape_filter?.enabled)
  if (key === 'smooth_boundary') return Boolean(config.smoothing?.enabled)
  if (key === 'gap_closing') return Boolean(config.morphology?.closing_enabled)
  if (key === 'separate_touching') return Boolean(config.watershed)
  if (key === 'remove_edge') return Boolean(config.remove_border)
  return false
}

const canRemoveActiveMethod = computed(() => wasMethodEnabledInSnapshot(activePostprocessMethod.value, methodConfigSnapshot.value))

const primaryMethodActionLabel = computed(() => (canRemoveActiveMethod.value ? '更新步骤' : '加入流程'))

const currentMethodPreview = computed(() => {
  const preview = methodPreviewPayload.value
  if (!preview) return null

  const isBefore = methodPreviewPhase.value === 'before'
  return {
    phase: methodPreviewPhase.value,
    label: isBefore ? '处理前' : '处理后',
    imageUrl: isBefore
      ? (preview.before_overlay_url ?? preview.before_mask_url)
      : (preview.after_overlay_url ?? preview.after_mask_url),
  }
})

const previewDialogData = computed(() => {
  if (!previewPayload.value) return null
  return {
    imageName: previewPayload.value.image_name,
    beforeImageUrl: previewPayload.value.before_overlay_url ?? previewPayload.value.before_mask_url,
    afterImageUrl: previewPayload.value.after_overlay_url ?? previewPayload.value.after_mask_url,
    beforeLabel: '当前确认结果',
    afterLabel: '应用流程后结果',
    message: `当前仅展示选中图像的前后对比；确认后会把整套后处理流程应用到当前任务的 ${previewPayload.value.image_count} 张图像，并同步刷新主图、统计与导出结果。`,
  }
})

const refreshResults = async () => {
  try {
    await loadPayload()
    ElMessage.success('结果已刷新')
  } catch {
    ElMessage.error('结果刷新失败')
  }
}

const navigateToStatistics = async () => {
  if (navigationInProgress.value) return
  if (!payload.value) return
  navigationInProgress.value = true
  try {
    await router.push(`/runs/${payload.value.run.id}/statistics`)
  } finally {
    navigationInProgress.value = false
  }
}

const clonePostprocessConfig = () => JSON.parse(JSON.stringify(workflow.form.postprocess)) as Record<string, any>

const restorePostprocessConfig = (config: Record<string, any>) => {
  workflow.form.postprocess.fill_holes = Boolean(config.fill_holes)
  workflow.form.postprocess.watershed = Boolean(config.watershed)
  workflow.form.postprocess.remove_border = Boolean(config.remove_border)

  workflow.form.postprocess.watershed_params.separation = Number(config.watershed_params?.separation ?? 35)
  workflow.form.postprocess.watershed_params.background_iterations = Number(config.watershed_params?.background_iterations ?? 1)
  workflow.form.postprocess.watershed_params.min_marker_area = Number(config.watershed_params?.min_marker_area ?? 12)

  workflow.form.postprocess.smoothing.enabled = Boolean(config.smoothing?.enabled)
  workflow.form.postprocess.smoothing.method = (config.smoothing?.method ?? 'gaussian') as typeof workflow.form.postprocess.smoothing.method
  workflow.form.postprocess.smoothing.kernel = Number(config.smoothing?.kernel ?? 3)

  workflow.form.postprocess.shape_filter.enabled = Boolean(config.shape_filter?.enabled)
  workflow.form.postprocess.shape_filter.min_area = Number(config.shape_filter?.min_area ?? 30)
  workflow.form.postprocess.shape_filter.max_area = config.shape_filter?.max_area ?? undefined
  workflow.form.postprocess.shape_filter.min_solidity = Number(config.shape_filter?.min_solidity ?? 0)
  workflow.form.postprocess.shape_filter.min_circularity = Number(config.shape_filter?.min_circularity ?? 0)
  workflow.form.postprocess.shape_filter.min_roundness = Number(config.shape_filter?.min_roundness ?? 0)
  workflow.form.postprocess.shape_filter.max_aspect_ratio = config.shape_filter?.max_aspect_ratio ?? undefined

  workflow.form.postprocess.morphology.opening_enabled = Boolean(config.morphology?.opening_enabled)
  workflow.form.postprocess.morphology.opening_kernel = Number(config.morphology?.opening_kernel ?? 3)
  workflow.form.postprocess.morphology.closing_enabled = Boolean(config.morphology?.closing_enabled)
  workflow.form.postprocess.morphology.closing_kernel = Number(config.morphology?.closing_kernel ?? 3)
}

const clearMethodPreviewTimer = () => {
  if (methodPreviewTimer) {
    window.clearTimeout(methodPreviewTimer)
    methodPreviewTimer = null
  }
}

const enablePostprocessMethod = (key: PostprocessMethodKey) => {
  if (key === 'fill_holes') {
    workflow.form.postprocess.fill_holes = true
  } else if (key === 'noise_cleanup') {
    workflow.form.postprocess.morphology.opening_enabled = true
    workflow.form.postprocess.shape_filter.enabled = true
  } else if (key === 'smooth_boundary') {
    workflow.form.postprocess.smoothing.enabled = true
  } else if (key === 'gap_closing') {
    workflow.form.postprocess.morphology.closing_enabled = true
  } else if (key === 'separate_touching') {
    workflow.form.postprocess.watershed = true
  } else if (key === 'remove_edge') {
    workflow.form.postprocess.remove_border = true
  }
}

const disablePostprocessMethod = (key: PostprocessMethodKey) => {
  if (key === 'fill_holes') {
    workflow.form.postprocess.fill_holes = false
  } else if (key === 'noise_cleanup') {
    workflow.form.postprocess.morphology.opening_enabled = false
    workflow.form.postprocess.shape_filter.enabled = false
  } else if (key === 'smooth_boundary') {
    workflow.form.postprocess.smoothing.enabled = false
  } else if (key === 'gap_closing') {
    workflow.form.postprocess.morphology.closing_enabled = false
  } else if (key === 'separate_touching') {
    workflow.form.postprocess.watershed = false
  } else if (key === 'remove_edge') {
    workflow.form.postprocess.remove_border = false
  }
}

const createPostprocessPreview = async (target: 'method' | 'confirm') => {
  if (!canPreviewPostprocess.value || !selectedMode.value) {
    ElMessage.warning(postprocessLockedReason.value)
    return null
  }

  const response = await api.post<PostprocessPreviewPayload>(`/runs/${runId.value}/postprocess/preview`, {
    mode: selectedMode.value,
    selected_image_id: selectedImageId.value,
    postprocess: workflow.form.postprocess,
  })

  if (target === 'confirm') {
    previewPayload.value = response.data
  }
  return response.data
}

const generateMethodPreview = async () => {
  if (!methodDialogVisible.value || !canPreviewPostprocess.value) return

  const requestSeq = ++methodPreviewSeq
  methodPreviewLoading.value = true
  methodPreviewError.value = null
  try {
    const nextPreview = await createPostprocessPreview('method')
    if (requestSeq === methodPreviewSeq) {
      methodPreviewPayload.value = nextPreview
    }
  } catch (error: any) {
    const message = error?.response?.data?.detail ?? error?.message ?? '后处理步骤预览生成失败'
    if (requestSeq === methodPreviewSeq) {
      methodPreviewError.value = String(message)
      methodPreviewPayload.value = null
    }
  } finally {
    if (requestSeq === methodPreviewSeq) {
      methodPreviewLoading.value = false
    }
  }
}

const scheduleMethodPreview = (delay = 300) => {
  if (!methodDialogVisible.value) return
  clearMethodPreviewTimer()
  methodPreviewTimer = window.setTimeout(() => {
    void generateMethodPreview()
  }, delay)
}

const openPostprocessMethod = async (key: PostprocessMethodKey) => {
  if (!canPreviewPostprocess.value) {
    ElMessage.warning(postprocessLockedReason.value)
    return
  }
  if (methodOpeningKey.value) return

  activePostprocessMethod.value = key
  methodConfigSnapshot.value = clonePostprocessConfig()
  enablePostprocessMethod(key)
  methodPreviewPayload.value = null
  methodPreviewError.value = null
  methodPreviewPhase.value = 'after'
  methodOpeningKey.value = key
  methodPreviewLoading.value = true
  const requestSeq = ++methodPreviewSeq

  try {
    const nextPreview = await createPostprocessPreview('method')
    if (requestSeq !== methodPreviewSeq) return
    if (!nextPreview) {
      if (methodConfigSnapshot.value) {
        restorePostprocessConfig(methodConfigSnapshot.value)
      }
      methodConfigSnapshot.value = null
      return
    }
    methodPreviewPayload.value = nextPreview
    methodDialogVisible.value = true
  } catch (error: any) {
    const message = error?.response?.data?.detail ?? error?.message ?? '后处理步骤预览生成失败'
    methodPreviewError.value = String(message)
    if (methodConfigSnapshot.value) {
      restorePostprocessConfig(methodConfigSnapshot.value)
    }
    methodConfigSnapshot.value = null
    ElMessage.error(String(message))
  } finally {
    if (requestSeq === methodPreviewSeq) {
      methodPreviewLoading.value = false
    }
    methodOpeningKey.value = null
  }
}

const cancelPostprocessMethod = () => {
  clearMethodPreviewTimer()
  methodPreviewSeq += 1
  if (methodConfigSnapshot.value) {
    restorePostprocessConfig(methodConfigSnapshot.value)
  }
  methodPreviewPayload.value = null
  methodPreviewError.value = null
  methodDialogVisible.value = false
  methodConfigSnapshot.value = null
}

const keepPostprocessMethod = () => {
  const wasEnabled = canRemoveActiveMethod.value
  clearMethodPreviewTimer()
  methodPreviewPayload.value = null
  methodPreviewError.value = null
  methodDialogVisible.value = false
  methodConfigSnapshot.value = null
  ElMessage.success(wasEnabled ? '后处理步骤已更新' : '后处理步骤已加入流程')
}

const removeActiveMethod = () => {
  clearMethodPreviewTimer()
  methodPreviewSeq += 1
  disablePostprocessMethod(activePostprocessMethod.value)
  methodPreviewPayload.value = null
  methodPreviewError.value = null
  methodDialogVisible.value = false
  methodConfigSnapshot.value = null
  ElMessage.success('该后处理步骤已移出流程')
}

const openMethodPreviewZoom = (phase: 'before' | 'after') => {
  const preview = methodPreviewPayload.value
  if (!preview) return
  const imageUrl =
    phase === 'before'
      ? (preview.before_overlay_url ?? preview.before_mask_url)
      : (preview.after_overlay_url ?? preview.after_mask_url)
  if (!imageUrl) return

  zoomPreviewTitle.value = `${activeMethod.value?.label ?? '后处理'} · ${phase === 'before' ? '处理前' : '处理后'}`
  zoomPreviewSubtitle.value = preview.image_name
  zoomPreviewImageUrl.value = imageUrl
  zoomPreviewVisible.value = true
}

const openPostprocessPreview = async () => {
  if (!canPreviewPostprocess.value || !selectedMode.value) {
    ElMessage.warning(postprocessLockedReason.value)
    return
  }

  previewLoading.value = true
  try {
    await createPostprocessPreview('confirm')
    previewDialogVisible.value = true
  } catch (error: any) {
    const message = error?.response?.data?.detail ?? error?.message ?? '后处理流程预览生成失败'
    ElMessage.error(String(message))
  } finally {
    previewLoading.value = false
  }
}

const closePreviewDialog = () => {
  resetPreviewState()
}

const confirmPostprocessPreview = async () => {
  if (!previewPayload.value) return

  confirmLoading.value = true
  try {
    await api.post<PostprocessConfirmPayload>(`/runs/${runId.value}/postprocess/confirm`, {
      preview_token: previewPayload.value.preview_token,
    })
    resetPreviewState()
    await loadPayload()
    ElMessage.success('后处理已确认，结果与统计已刷新')
  } catch (error: any) {
    const status = error?.response?.status
    const message = error?.response?.data?.detail ?? error?.message ?? '后处理确认失败'
    if (status === 409) {
      resetPreviewState()
      await loadPayload()
      ElMessage.warning('预览已过期，请重新生成后处理流程预览。')
      return
    }
    if (status === 404) {
      resetPreviewState()
      ElMessage.warning('预览不存在或已失效，请重新生成。')
      return
    }
    ElMessage.error(String(message))
  } finally {
    confirmLoading.value = false
  }
}

const initializePage = async (preserveSelection = false) => {
  if (!Number.isFinite(runId.value) || runId.value <= 0) {
    ElMessage.error('任务不存在')
    router.replace('/history')
    return
  }

  loading.value = true
  resetPreviewState()
  stopPolling()

  try {
    await loadPayload(preserveSelection)
    const status = payload.value?.run.status
    if (status && !['completed', 'failed', 'partial_success'].includes(status)) {
      startPolling()
    }
  } catch {
    ElMessage.error('结果页加载失败')
  } finally {
    loading.value = false
  }
}

watch(runId, async (nextValue, previousValue) => {
  if (!previousValue || nextValue === previousValue) return
  hydratedPostprocessRunId.value = null
  await initializePage(false)
})

watch(
  () => [
    activePostprocessMethod.value,
    selectedImageId.value,
    selectedMode.value,
    JSON.stringify(workflow.form.postprocess),
  ],
  () => {
    if (!methodDialogVisible.value) return
    scheduleMethodPreview()
  },
)

onMounted(async () => {
  await initializePage(false)
})

onBeforeUnmount(() => {
  clearMethodPreviewTimer()
  stopPolling()
})
</script>

<template>
  <div class="detail-page">
    <template v-if="payload">
      <section class="glass-card detail-toolbar">
        <div class="detail-toolbar-top">
          <div class="detail-toolbar-head">
            <span class="detail-toolbar-kicker">结果工作台</span>
            <h2 class="section-title">结果核查与后处理</h2>
            <p class="section-subtitle">
              {{ payload.run.name }} · 创建于 {{ formatDateTime(payload.run.created_at) }}
            </p>
          </div>

          <div class="detail-toolbar-actions">
            <el-button class="detail-toolbar-jump" plain size="large" @click="refreshResults">刷新结果</el-button>
            <el-button
              class="detail-toolbar-jump detail-toolbar-jump--primary"
              type="primary"
              size="large"
              :disabled="!canOpenStatistics || navigationInProgress"
              @click="navigateToStatistics"
            >
              进入统计分析
            </el-button>
          </div>
        </div>

        <div class="detail-toolbar-summary">
          <article v-for="item in detailSummaryItems" :key="item.label" class="detail-toolbar-metric">
            <span>{{ item.label }}</span>
            <strong>{{ item.value }}</strong>
          </article>
        </div>
      </section>

      <CalibrationStatusBanner :run="payload.run" />

      <section class="glass-card step-card">
        <div class="step-card-shell">
          <button type="button" class="step-card-toggle" :aria-expanded="stepExpanded" @click="stepExpanded = !stepExpanded">
            <div class="step-card-copy">
              <h3 class="section-title">执行轨迹</h3>
            </div>
            <span class="step-toggle-indicator">{{ stepExpanded ? '−' : '+' }}</span>
          </button>

          <div v-show="stepExpanded" class="step-card-body">
            <div class="step-rail">
              <article
                v-for="(step, index) in payload.steps"
                :key="step.id"
                class="step-node"
                :class="stepCardClass(step)"
              >
                <div class="step-node-head">
                  <span class="step-order">{{ String(index + 1).padStart(2, '0') }}</span>
                  <span class="step-state">{{ stepStatusLabel(step) }}</span>
                </div>

                <div class="step-node-body">
                  <div class="step-node-dot" />
                  <div class="step-node-copy">
                    <strong>{{ stepLabel(step.step_key) }}</strong>
                    <span v-if="stepProgressCopy(step)" class="step-node-progress">{{ stepProgressCopy(step) }}</span>
                    <p>{{ stepMessageCopy(step) }}</p>
                    <small v-if="stepImageCopy(step)">{{ stepImageCopy(step) }}</small>
                  </div>
                </div>
              </article>
            </div>
          </div>
        </div>
      </section>

      <div class="detail-grid">
        <main class="detail-main">
          <PreviewWorkspace
            v-model:current-image-id="selectedImageId"
            v-model:current-mode="selectedMode"
            :images="payload.images"
            :mode="payload.run.segmentation_mode"
            empty-description="当前任务尚未产出可核查图层，完成主分割后会在这里展示原图、分析区域与分割结果。"
          />
        </main>

        <aside class="detail-side">
          <section class="glass-card postprocess-card">
            <div class="panel-head">
              <div>
                <h3 class="section-title">后处理步骤</h3>
                <p class="section-subtitle">{{ postprocessLockedReason }}</p>
              </div>
            </div>

            <div class="postprocess-overview">
              <article v-for="item in postprocessOverview" :key="item.label" class="overview-item">
                <span>{{ item.label }}</span>
                <strong>{{ item.value }}</strong>
              </article>
            </div>

            <div class="postprocess-method-grid">
              <button
                v-for="method in postprocessMethods"
                :key="method.key"
                type="button"
                class="postprocess-method-card"
                :class="{ 'is-enabled': method.enabled, 'is-opening': methodOpeningKey === method.key }"
                :disabled="!canPreviewPostprocess || Boolean(methodOpeningKey)"
                @click="openPostprocessMethod(method.key)"
              >
                <span class="postprocess-method-status">{{ methodOpeningKey === method.key ? '生成预览中' : method.enabled ? '已加入' : '未加入' }}</span>
                <strong>{{ method.label }}</strong>
              </button>
            </div>

            <div class="postprocess-actions">
              <div class="postprocess-display-note">主界面当前显示：已确认结果。加入步骤后需应用流程才会刷新。</div>

              <el-button
                class="postprocess-action"
                type="primary"
                size="large"
                :loading="previewLoading"
                :disabled="!canPreviewPostprocess"
                @click="openPostprocessPreview"
              >
                应用流程并刷新结果
              </el-button>
            </div>
          </section>
        </aside>
      </div>

      <el-dialog
        v-model="methodDialogVisible"
        width="min(1120px, 94vw)"
        top="5vh"
        append-to-body
        destroy-on-close
        :show-close="false"
        :close-on-click-modal="false"
        :close-on-press-escape="false"
        class="postprocess-method-dialog"
      >
        <template #header>
          <div class="method-dialog-head">
            <div class="method-dialog-copy">
              <strong>{{ activeMethod?.label }}</strong>
            </div>
          </div>
        </template>

        <div class="method-dialog-body">
          <section class="method-preview-panel">
            <div class="method-preview-head">
              <div class="method-preview-copy">
                <strong>步骤预览</strong>
              </div>
              <div class="method-preview-toolbar">
                <div class="method-phase-toggle">
                  <el-radio-group v-model="methodPreviewPhase" size="large">
                    <el-radio-button label="before">处理前</el-radio-button>
                    <el-radio-button label="after">处理后</el-radio-button>
                  </el-radio-group>
                </div>
                <el-button
                  plain
                  size="large"
                  :loading="methodPreviewLoading"
                  @click="generateMethodPreview"
                >
                  {{ methodPreviewPayload ? '刷新预览' : '生成预览' }}
                </el-button>
              </div>
            </div>

            <div v-if="methodPreviewLoading && !methodPreviewPayload" class="method-preview-state">
              <el-skeleton :rows="5" animated />
              <p>正在生成预览...</p>
            </div>

            <el-empty
              v-else-if="methodPreviewError"
              :description="methodPreviewError"
              class="method-preview-empty"
            />

            <div v-else-if="methodPreviewPayload" class="method-canvas-shell">
              <div class="method-canvas-meta">
                <strong>{{ selectedImage?.image_name ?? methodPreviewPayload.image_name }}</strong>
              </div>
              <div
                v-if="currentMethodPreview?.imageUrl"
                class="method-canvas-media"
              >
                <button
                  type="button"
                  class="method-canvas-image-button"
                  @click="openMethodPreviewZoom(methodPreviewPhase)"
                >
                  <img
                    :src="currentMethodPreview.imageUrl"
                    :alt="currentMethodPreview.label"
                  />
                </button>
                <span v-if="methodPreviewLoading" class="method-canvas-refreshing">刷新中</span>
              </div>
              <el-empty v-else :description="`暂无${currentMethodPreview?.label ?? '当前'}预览`" class="method-preview-empty" />
            </div>

            <el-empty v-else description="当前还没有预览结果" class="method-preview-empty" />
          </section>

          <section class="method-param-panel">
            <div class="method-param-head">
              <strong>参数</strong>
            </div>

            <el-form label-position="top" class="method-param-form">
              <template v-if="activePostprocessMethod === 'fill_holes'">
                <div class="method-param-empty">无额外参数</div>
              </template>

              <template v-else-if="activePostprocessMethod === 'noise_cleanup'">
                <el-form-item :label="`去噪强度：${workflow.form.postprocess.morphology.opening_kernel}`">
                  <el-slider
                    v-model="workflow.form.postprocess.morphology.opening_kernel"
                    :min="1"
                    :max="15"
                    :step="2"
                    :marks="methodSliderMarks.opening"
                    show-stops
                    show-input
                    :show-input-controls="false"
                  />
                </el-form-item>
                <el-form-item :label="`保留面积范围：${workflow.form.postprocess.shape_filter.min_area} - ${workflow.form.postprocess.shape_filter.max_area ?? '不限'}`">
                  <el-slider
                    v-model="noiseCleanupAreaRange"
                    range
                    :min="1"
                    :max="1200"
                    :step="1"
                    :marks="methodSliderMarks.areaRange"
                    show-stops
                  />
                </el-form-item>
                <el-form-item :label="`实心度阈值：${workflow.form.postprocess.shape_filter.min_solidity.toFixed(2)}`">
                  <el-slider
                    v-model="workflow.form.postprocess.shape_filter.min_solidity"
                    :min="0"
                    :max="0.95"
                    :step="0.05"
                    :marks="methodSliderMarks.quality"
                    show-input
                    :show-input-controls="false"
                  />
                </el-form-item>
                <el-form-item :label="`圆度阈值：${workflow.form.postprocess.shape_filter.min_circularity.toFixed(2)}`">
                  <el-slider
                    v-model="workflow.form.postprocess.shape_filter.min_circularity"
                    :min="0"
                    :max="0.95"
                    :step="0.05"
                    :marks="methodSliderMarks.quality"
                    show-input
                    :show-input-controls="false"
                  />
                </el-form-item>
              </template>

              <template v-else-if="activePostprocessMethod === 'smooth_boundary'">
                <el-form-item label="平滑方式">
                  <el-radio-group v-model="workflow.form.postprocess.smoothing.method" class="method-mode-select">
                    <el-radio-button label="gaussian">高斯</el-radio-button>
                    <el-radio-button label="median">中值</el-radio-button>
                    <el-radio-button label="mean">均值</el-radio-button>
                  </el-radio-group>
                </el-form-item>
                <el-form-item :label="`平滑核大小：${workflow.form.postprocess.smoothing.kernel}`">
                  <el-slider
                    v-model="workflow.form.postprocess.smoothing.kernel"
                    :min="1"
                    :max="15"
                    :step="2"
                    :marks="methodSliderMarks.smooth"
                    show-stops
                    show-input
                    :show-input-controls="false"
                  />
                </el-form-item>
              </template>

              <template v-else-if="activePostprocessMethod === 'gap_closing'">
                <el-form-item :label="`闭运算核大小：${workflow.form.postprocess.morphology.closing_kernel}`">
                  <el-slider
                    v-model="workflow.form.postprocess.morphology.closing_kernel"
                    :min="1"
                    :max="21"
                    :step="2"
                    :marks="methodSliderMarks.closing"
                    show-stops
                    show-input
                    :show-input-controls="false"
                  />
                </el-form-item>
              </template>

              <template v-else-if="activePostprocessMethod === 'separate_touching'">
                <el-form-item :label="`分离强度：${workflow.form.postprocess.watershed_params.separation}`">
                  <el-slider
                    v-model="workflow.form.postprocess.watershed_params.separation"
                    :min="5"
                    :max="85"
                    :marks="methodSliderMarks.watershed"
                    show-input
                    :show-input-controls="false"
                  />
                </el-form-item>
                <el-form-item :label="`背景扩张次数：${workflow.form.postprocess.watershed_params.background_iterations}`">
                  <el-slider
                    v-model="workflow.form.postprocess.watershed_params.background_iterations"
                    :min="1"
                    :max="5"
                    :step="1"
                    :marks="methodSliderMarks.background"
                    show-stops
                    show-input
                    :show-input-controls="false"
                  />
                </el-form-item>
                <el-form-item :label="`最小种子面积：${workflow.form.postprocess.watershed_params.min_marker_area}`">
                  <el-slider
                    v-model="workflow.form.postprocess.watershed_params.min_marker_area"
                    :min="0"
                    :max="120"
                    :step="1"
                    :marks="methodSliderMarks.markerArea"
                    show-input
                    :show-input-controls="false"
                  />
                </el-form-item>
              </template>

              <template v-else-if="activePostprocessMethod === 'remove_edge'">
                <div class="method-param-empty">无额外参数</div>
              </template>
            </el-form>
          </section>
        </div>

        <template #footer>
          <div class="method-dialog-actions">
            <div class="method-dialog-actions__buttons">
              <el-button v-if="canRemoveActiveMethod" plain @click="removeActiveMethod">移出流程</el-button>
              <el-button size="large" @click="cancelPostprocessMethod">取消</el-button>
              <el-button type="primary" size="large" @click="keepPostprocessMethod">{{ primaryMethodActionLabel }}</el-button>
            </div>
          </div>
        </template>
      </el-dialog>

      <PostprocessPreviewDialog
        v-model="previewDialogVisible"
        :preview="previewDialogData"
        :confirm-loading="confirmLoading"
        @cancel="closePreviewDialog"
        @confirm="confirmPostprocessPreview"
      />
      <ZoomableImageDialog
        v-model="zoomPreviewVisible"
        :image-url="zoomPreviewImageUrl"
        :image-alt="zoomPreviewTitle"
        :title="zoomPreviewTitle"
        :subtitle="zoomPreviewSubtitle"
      />
    </template>

    <section v-else class="glass-card loading-card">
      <el-skeleton :rows="10" animated />
    </section>
  </div>
</template>

<style scoped>
.detail-page {
  display: grid;
  gap: 16px;
  min-width: 0;
}

.detail-toolbar,
.step-card,
.postprocess-card,
.loading-card {
  padding: 18px;
}

.detail-toolbar {
  display: grid;
  gap: 14px;
}

.detail-toolbar-top,
.panel-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 14px;
}

.detail-toolbar-head {
  display: grid;
  gap: 6px;
}

.detail-toolbar-kicker {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  width: fit-content;
  padding: 0 10px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 11px;
  font-weight: 700;
}

.detail-toolbar-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.detail-toolbar-jump,
.postprocess-action {
  min-width: 152px;
  min-height: 44px;
  border-radius: 15px;
  font-weight: 700;
}

.detail-toolbar-jump--primary,
.postprocess-action {
  box-shadow: 0 12px 22px rgba(23, 96, 135, 0.18);
}

.detail-toolbar-summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.detail-toolbar-metric,
.overview-item {
  display: grid;
  gap: 5px;
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.detail-toolbar-metric span,
.overview-item span {
  color: var(--muted);
  font-size: 12px;
}

.detail-toolbar-metric strong,
.overview-item strong {
  font-size: 16px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.step-card-shell {
  display: grid;
  gap: 0;
  padding: 14px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.84), rgba(247, 242, 235, 0.78));
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.step-card-toggle {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  text-align: left;
  cursor: pointer;
}

.step-card-copy {
  display: grid;
  gap: 6px;
}

.step-toggle-indicator {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 999px;
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
  font-size: 18px;
}

.step-card-body {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid rgba(31, 40, 48, 0.06);
}

.step-rail {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: minmax(188px, 220px);
  gap: 12px;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 4px 8px 12px;
  scrollbar-width: thin;
  scrollbar-color: rgba(23, 96, 135, 0.28) transparent;
}

.step-node {
  position: relative;
  display: grid;
  gap: 12px;
  min-height: 148px;
  padding: 13px;
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.7);
  box-shadow: 0 12px 28px rgba(44, 32, 20, 0.05);
}

.step-node::after {
  content: '';
  position: absolute;
  top: 50%;
  left: calc(100% + 5px);
  width: 12px;
  height: 2px;
  background: rgba(23, 96, 135, 0.16);
  transform: translateY(-50%);
}

.step-node:last-child::after {
  display: none;
}

.step-node.is-completed {
  border-color: rgba(84, 174, 93, 0.18);
}

.step-node.is-running {
  border-color: rgba(23, 96, 135, 0.16);
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.08), rgba(255, 255, 255, 0.72));
}

.step-node.is-failed {
  border-color: rgba(184, 90, 43, 0.18);
  background: linear-gradient(135deg, rgba(184, 90, 43, 0.08), rgba(255, 255, 255, 0.72));
}

.step-node.is-pending,
.step-node.is-inactive {
  border-style: dashed;
}

.step-node-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.step-order,
.step-state {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 26px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
}

.step-order {
  background: rgba(31, 40, 48, 0.06);
  color: var(--muted);
}

.step-state {
  background: rgba(23, 96, 135, 0.08);
  color: var(--accent);
}

.step-node.is-inactive .step-state {
  background: rgba(31, 40, 48, 0.06);
  color: var(--muted);
}

.step-node-body {
  display: grid;
  grid-template-columns: 12px minmax(0, 1fr);
  gap: 12px;
  align-items: flex-start;
}

.step-node-dot {
  width: 12px;
  height: 12px;
  margin-top: 5px;
  border-radius: 999px;
  background: #54ae5d;
  box-shadow: 0 0 0 4px rgba(84, 174, 93, 0.12);
}

.step-node.is-running .step-node-dot {
  background: var(--accent);
  box-shadow: 0 0 0 4px rgba(23, 96, 135, 0.12);
}

.step-node.is-failed .step-node-dot {
  background: var(--primary);
  box-shadow: 0 0 0 4px rgba(184, 90, 43, 0.12);
}

.step-node.is-pending .step-node-dot {
  background: transparent;
  border: 2px solid rgba(23, 96, 135, 0.5);
  box-shadow: none;
}

.step-node-copy {
  display: grid;
  gap: 5px;
}

.step-node-copy strong {
  font-size: 16px;
  line-height: 1.25;
}

.step-node-progress {
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}

.step-node-copy p,
.step-node-copy small {
  margin: 0;
  color: var(--muted);
  line-height: 1.5;
}

.detail-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.82fr);
  gap: 16px;
  align-items: start;
}

.detail-main,
.detail-side {
  min-width: 0;
  display: grid;
  gap: 16px;
}

.postprocess-card {
  display: grid;
  gap: 16px;
}

.postprocess-overview {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.postprocess-method-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px 16px;
}

.postprocess-method-card {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 14px;
  min-height: 92px;
  padding: 18px 20px;
  align-items: center;
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.07);
  background: rgba(255, 255, 255, 0.74);
  color: inherit;
  text-align: left;
  cursor: pointer;
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.postprocess-method-card:hover:not(:disabled) {
  transform: translateY(-1px);
  border-color: rgba(23, 96, 135, 0.18);
  box-shadow: 0 14px 26px rgba(23, 96, 135, 0.1);
}

.postprocess-method-card:disabled {
  cursor: not-allowed;
  opacity: 0.62;
}

.postprocess-method-card.is-enabled {
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.1), rgba(255, 255, 255, 0.78));
  border-color: rgba(23, 96, 135, 0.14);
}

.postprocess-method-card.is-opening {
  cursor: progress;
  border-color: rgba(23, 96, 135, 0.2);
  box-shadow: inset 0 0 0 1px rgba(23, 96, 135, 0.08);
}

.postprocess-method-card strong {
  display: block;
  font-size: 18px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.postprocess-method-status {
  display: inline-flex;
  width: fit-content;
  align-items: center;
  min-height: 26px;
  padding: 0 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  background: rgba(31, 40, 48, 0.06);
  color: var(--muted);
}

.postprocess-method-card.is-enabled .postprocess-method-status {
  background: rgba(23, 96, 135, 0.1);
  color: var(--accent);
}

.postprocess-actions {
  display: grid;
  gap: 10px;
}

.postprocess-display-note {
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(23, 96, 135, 0.07);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  line-height: 1.55;
}

.method-dialog-head strong {
  display: block;
  margin: 4px 0;
  font-size: 22px;
  line-height: 1.3;
}

.method-dialog-copy {
  display: grid;
  gap: 4px;
}

.method-dialog-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.method-param-panel,
.method-preview-panel {
  border-radius: 18px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.78);
}

.method-param-panel,
.method-preview-panel {
  display: grid;
  gap: 14px;
  padding: 16px;
}

.method-param-head strong,
.method-preview-head strong {
  display: block;
  font-size: 16px;
}

.method-param-form {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 16px;
}

.method-param-form .el-form-item {
  margin-bottom: 0;
}

.method-param-form .el-select,
.method-param-form .el-input-number,
.method-mode-select {
  width: 100%;
}

.method-param-form :deep(.el-form-item__label) {
  color: var(--text);
  font-weight: 700;
}

.method-param-form :deep(.el-slider) {
  --el-slider-main-bg-color: var(--accent);
  --el-slider-runway-bg-color: rgba(31, 40, 48, 0.1);
}

.method-param-empty {
  display: flex;
  min-height: 58px;
  align-items: center;
  padding: 0 16px;
  border-radius: 16px;
  background: rgba(247, 242, 235, 0.76);
  border: 1px solid rgba(31, 40, 48, 0.06);
  color: var(--muted);
  font-weight: 700;
}

.method-preview-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.method-preview-copy {
  display: grid;
  gap: 4px;
}

.method-preview-toolbar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
  flex-wrap: wrap;
}

.method-phase-toggle {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 16px;
  background: rgba(247, 242, 235, 0.88);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.method-phase-toggle :deep(.el-radio-button__inner) {
  min-width: 96px;
  min-height: 40px;
  font-weight: 700;
}

.method-mode-select :deep(.el-radio-button__inner) {
  min-width: 88px;
  font-weight: 700;
}

.method-canvas-shell {
  display: grid;
  gap: 8px;
}

.method-canvas-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  min-height: 24px;
}

.method-canvas-meta strong {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.method-canvas-media {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: min(60vh, 560px);
  border-radius: 14px;
  overflow: hidden;
  background:
    linear-gradient(45deg, rgba(255, 255, 255, 0.035) 25%, transparent 25%),
    linear-gradient(-45deg, rgba(255, 255, 255, 0.035) 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, rgba(255, 255, 255, 0.035) 75%),
    linear-gradient(-45deg, transparent 75%, rgba(255, 255, 255, 0.035) 75%),
    linear-gradient(180deg, rgba(18, 23, 28, 0.96), rgba(31, 37, 42, 0.92));
  background-position: 0 0, 0 8px, 8px -8px, -8px 0, 0 0;
  background-size: 16px 16px, 16px 16px, 16px 16px, 16px 16px, auto;
}

.method-preview-state,
.method-preview-empty {
  min-height: 280px;
  border-radius: 16px;
  background: rgba(247, 242, 235, 0.62);
  border: 1px dashed rgba(31, 40, 48, 0.12);
}

.method-preview-state {
  display: grid;
  align-content: center;
  gap: 14px;
  padding: 24px;
}

.method-preview-state p {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  text-align: center;
}

.method-canvas-media img {
  display: block;
  max-width: 100%;
  max-height: min(60vh, 560px);
  object-fit: contain;
}

.method-canvas-image-button {
  display: flex;
  width: 100%;
  min-height: min(60vh, 560px);
  align-items: center;
  justify-content: center;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: zoom-in;
}

.method-canvas-refreshing {
  position: absolute;
  top: 12px;
  right: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.88);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}

.method-dialog-actions {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 12px;
  padding-top: 10px;
  border-top: 1px solid rgba(31, 40, 48, 0.08);
}

.method-dialog-actions__buttons {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  flex-wrap: wrap;
}

.method-dialog-actions__buttons .el-button {
  min-width: 132px;
  min-height: 42px;
  border-radius: 14px;
  font-weight: 700;
}

.loading-card {
  padding: 24px;
}

@media (max-width: 1180px) {
  .detail-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 920px) {
  .detail-toolbar-top,
  .panel-head,
  .step-card-toggle {
    flex-direction: column;
    align-items: stretch;
  }

  .detail-toolbar-summary,
  .postprocess-overview,
  .postprocess-method-grid {
    grid-template-columns: 1fr 1fr;
  }

  .postprocess-actions {
    justify-content: stretch;
  }

  .postprocess-action {
    width: 100%;
  }
}

@media (max-width: 640px) {
  .detail-toolbar-summary,
  .postprocess-overview,
  .postprocess-method-grid {
    grid-template-columns: 1fr;
  }

  .method-preview-head,
  .method-canvas-meta,
  .method-dialog-actions,
  .method-preview-toolbar,
  .method-dialog-actions__buttons {
    flex-direction: column;
    align-items: stretch;
  }

  .method-phase-toggle {
    justify-content: space-between;
  }
}
</style>
