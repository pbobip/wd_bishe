<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'

import type { RunResultImage } from '../types'

interface ArtifactItem {
  key: string
  label: string
  url: string
  group: 'source' | 'result'
  hint: string
  fit?: 'contain'
}

const NUMBERED_ARTIFACT_KEYS = [
  'numbered_overlay_url',
  'object_number_overlay_url',
  'object_overlay_url',
  'labels_overlay_url',
  'label_overlay_url',
  'instance_overlay_url',
  'instance_label_overlay_url',
  'id_overlay_url',
]

const props = withDefaults(
  defineProps<{
    images: RunResultImage[]
    mode: string
    currentImageId?: number | null
    currentMode?: string | null
    emptyDescription?: string
  }>(),
  {
    currentImageId: null,
    currentMode: null,
    emptyDescription: '运行完成后会在这里显示图像核查结果',
  },
)

const emit = defineEmits<{
  'update:currentImageId': [value: number | null]
  'update:currentMode': [value: string | null]
}>()

const currentId = ref<number | null>(null)
const selectedMode = ref('')
const selectedArtifactKey = ref('')
const previewVisible = ref(false)
const previewTitle = ref('')
const previewImageUrl = ref('')
const previewContainerRef = ref<HTMLDivElement | null>(null)
const previewImageRef = ref<HTMLImageElement | null>(null)
const fitToWindow = ref(true)
const zoom = ref(1)
const dragging = ref(false)
const dragStartX = ref(0)
const dragStartY = ref(0)
const scrollStartLeft = ref(0)
const scrollStartTop = ref(0)
const naturalWidth = ref(0)
const naturalHeight = ref(0)

const MIN_PREVIEW_ZOOM = 1
const MAX_PREVIEW_ZOOM = 6
const PREVIEW_ZOOM_STEP = 0.12

const modeLabel = (mode: string) => {
  if (mode === 'traditional') return '传统分割'
  if (mode === 'dl') return '深度学习'
  return mode
}

const currentIndex = computed(() => props.images.findIndex((image) => image.image_id === currentId.value))
const currentImage = computed(() => props.images.find((image) => image.image_id === currentId.value) ?? null)
const availableModes = computed(() => Object.keys(currentImage.value?.modes ?? {}))

const activeModeOptions = computed(() =>
  availableModes.value.map((mode) => ({
    label: modeLabel(mode),
    value: mode,
  })),
)

const resolveOptionalArtifact = (artifacts: Record<string, string | null> | undefined) => {
  if (!artifacts) return null
  for (const key of NUMBERED_ARTIFACT_KEYS) {
    const url = artifacts[key]
    if (url) {
      return {
        key,
        label: '对象编号图',
        url,
        group: 'result' as const,
        hint: '用于逐个核查颗粒编号与连通域标注。',
      }
    }
  }
  return null
}

const sourceArtifacts = computed<ArtifactItem[]>(() => {
  if (!currentImage.value) return []
  const artifacts = currentImage.value.modes[selectedMode.value]?.artifacts ?? {}
  const sourceItems: ArtifactItem[] = [
    {
      key: 'input_url',
      label: '原图',
      url: currentImage.value.input_url,
      group: 'source',
      hint: '保留原始 SEM 视图，适合核对边界与灰度变化。',
    },
  ]
  if (artifacts.analysis_input_url) {
    sourceItems.push({
      key: 'analysis_input_url',
      label: '分析区域',
      url: artifacts.analysis_input_url,
      group: 'source',
      hint: '自动裁掉底栏后的有效分析区域，用于实际分割与统计。',
    })
  }
  if (artifacts.footer_panel_url) {
    sourceItems.push({
      key: 'footer_panel_url',
      label: '底部信息栏',
      url: artifacts.footer_panel_url,
      group: 'source',
      hint: '保留原始采集信息，便于标定与追溯。',
      fit: 'contain',
    })
  }
  return sourceItems
})

const resultArtifacts = computed<ArtifactItem[]>(() => {
  const artifacts = currentImage.value?.modes[selectedMode.value]?.artifacts
  if (!artifacts) return []
  const items: ArtifactItem[] = []
  if (artifacts.overlay_url) {
    items.push({
      key: 'overlay_url',
      label: '叠加图',
      url: artifacts.overlay_url,
      group: 'result',
      hint: '优先核查分割轮廓与原图是否贴合。',
    })
  }
  if (artifacts.mask_url) {
    items.push({
      key: 'mask_url',
      label: '掩码',
      url: artifacts.mask_url,
      group: 'result',
      hint: '查看纯二值分割结果，检查漏分与误分。',
    })
  }
  if (artifacts.edge_url) {
    items.push({
      key: 'edge_url',
      label: '边界图',
      url: artifacts.edge_url,
      group: 'result',
      hint: '突出颗粒轮廓，适合检查边界连续性。',
    })
  }
  const numberedArtifact = resolveOptionalArtifact(artifacts)
  if (numberedArtifact) {
    items.push(numberedArtifact)
  }
  return items
})

const artifactGroups = computed(() => {
  const groups: Array<{ title: string; items: ArtifactItem[] }> = []
  if (sourceArtifacts.value.length) {
    groups.push({ title: '原始与分析图层', items: sourceArtifacts.value })
  }
  if (resultArtifacts.value.length) {
    groups.push({ title: `${modeLabel(selectedMode.value)}结果图层`, items: resultArtifacts.value })
  }
  return groups
})

const allArtifacts = computed(() => [...sourceArtifacts.value, ...resultArtifacts.value])
const defaultArtifactKey = computed(
  () =>
    resultArtifacts.value.find((item) => item.key === 'overlay_url')?.key ??
    resultArtifacts.value[0]?.key ??
    sourceArtifacts.value[0]?.key ??
    '',
)

const currentArtifact = computed(
  () =>
    allArtifacts.value.find((item) => item.key === selectedArtifactKey.value) ??
    allArtifacts.value.find((item) => item.key === defaultArtifactKey.value) ??
    null,
)

const imageOptions = computed(() =>
  props.images.map((image, index) => ({
    value: image.image_id,
    label: `${index + 1}. ${image.image_name}`,
  })),
)

const isFooterArtifact = computed(() => currentArtifact.value?.key === 'footer_panel_url')

const previewImageStyle = computed(() => {
  if (fitToWindow.value) {
    return {
      maxWidth: '100%',
      maxHeight: '78vh',
      width: 'auto',
      height: 'auto',
      cursor: 'zoom-in',
    }
  }
  const scaledWidth = naturalWidth.value > 0 ? `${Math.max(1, Math.round(naturalWidth.value * zoom.value))}px` : 'auto'
  const scaledHeight = naturalHeight.value > 0 ? `${Math.max(1, Math.round(naturalHeight.value * zoom.value))}px` : 'auto'
  return {
    maxWidth: 'none',
    maxHeight: 'none',
    width: scaledWidth,
    height: scaledHeight,
    cursor: dragging.value ? 'grabbing' : 'grab',
  }
})

watch(
  () => [props.images, props.currentImageId] as const,
  ([images, externalCurrentImageId]) => {
    const requested = images.find((image) => image.image_id === externalCurrentImageId)
    const existing = images.find((image) => image.image_id === currentId.value)
    currentId.value = requested?.image_id ?? existing?.image_id ?? images[0]?.image_id ?? null
  },
  { immediate: true },
)

watch(currentId, (value) => {
  emit('update:currentImageId', value)
})

watch(selectedMode, (value) => {
  emit('update:currentMode', value || null)
})

watch(
  availableModes,
  (modes) => {
    if (!modes.length) {
      selectedMode.value = ''
      return
    }
    if (props.currentMode && modes.includes(props.currentMode)) {
      selectedMode.value = props.currentMode
      return
    }
    if (!modes.includes(selectedMode.value)) {
      selectedMode.value = modes.includes(props.mode) ? props.mode : modes[0]
    }
  },
  { immediate: true },
)

watch(
  () => props.currentMode,
  (value) => {
    if (!value) return
    if (availableModes.value.includes(value) && selectedMode.value !== value) {
      selectedMode.value = value
    }
  },
)

watch(
  () => `${currentId.value}|${selectedMode.value}|${allArtifacts.value.map((item) => item.key).join(',')}`,
  () => {
    const existing = allArtifacts.value.find((item) => item.key === selectedArtifactKey.value)
    if (!existing) {
      selectedArtifactKey.value = defaultArtifactKey.value
    }
  },
  { immediate: true },
)

watch(previewVisible, (visible) => {
  if (!visible) dragging.value = false
})

const goToImage = (offset: number) => {
  if (!props.images.length || currentIndex.value < 0) return
  const nextIndex = Math.min(props.images.length - 1, Math.max(0, currentIndex.value + offset))
  currentId.value = props.images[nextIndex]?.image_id ?? currentId.value
}

const selectArtifact = (key: string) => {
  selectedArtifactKey.value = key
}

const openPreview = (title: string, url?: string | null) => {
  if (!url) return
  previewTitle.value = title
  previewImageUrl.value = url
  fitToWindow.value = true
  zoom.value = 1
  naturalWidth.value = 0
  naturalHeight.value = 0
  previewVisible.value = true
}

const showOriginalSize = async () => {
  fitToWindow.value = false
  zoom.value = 1
  await nextTick()
  if (previewContainerRef.value) {
    previewContainerRef.value.scrollLeft = 0
    previewContainerRef.value.scrollTop = 0
  }
}

const resetToFit = async () => {
  fitToWindow.value = true
  zoom.value = 1
  await nextTick()
  if (previewContainerRef.value) {
    previewContainerRef.value.scrollLeft = 0
    previewContainerRef.value.scrollTop = 0
  }
}

const getPreviewBaseZoom = () => (fitToWindow.value ? MIN_PREVIEW_ZOOM : zoom.value)

const onPreviewImageLoad = () => {
  if (!previewImageRef.value) return
  naturalWidth.value = previewImageRef.value.naturalWidth
  naturalHeight.value = previewImageRef.value.naturalHeight
}

const applyZoomAroundPoint = async (nextZoom: number, clientX?: number, clientY?: number) => {
  const container = previewContainerRef.value
  if (!container) return
  const previousZoom = zoom.value
  const containerRect = container.getBoundingClientRect()
  const anchorX = clientX ?? containerRect.left + containerRect.width / 2
  const anchorY = clientY ?? containerRect.top + containerRect.height / 2
  const offsetX = anchorX - containerRect.left
  const offsetY = anchorY - containerRect.top
  const imageX = (container.scrollLeft + offsetX) / previousZoom
  const imageY = (container.scrollTop + offsetY) / previousZoom
  fitToWindow.value = false
  zoom.value = nextZoom
  await nextTick()
  container.scrollLeft = Math.max(0, imageX * nextZoom - offsetX)
  container.scrollTop = Math.max(0, imageY * nextZoom - offsetY)
}

const zoomInPreview = async (clientX?: number, clientY?: number) => {
  const nextZoom = Math.min(MAX_PREVIEW_ZOOM, +(getPreviewBaseZoom() + PREVIEW_ZOOM_STEP).toFixed(2))
  await applyZoomAroundPoint(nextZoom, clientX, clientY)
}

const zoomOutPreview = async (clientX?: number, clientY?: number) => {
  const nextZoom = Math.max(MIN_PREVIEW_ZOOM, +(getPreviewBaseZoom() - PREVIEW_ZOOM_STEP).toFixed(2))
  if (nextZoom <= MIN_PREVIEW_ZOOM) {
    resetToFit()
    return
  }
  await applyZoomAroundPoint(nextZoom, clientX, clientY)
}

const onWheelZoom = async (event: WheelEvent) => {
  event.preventDefault()
  if (event.deltaY < 0) {
    await zoomInPreview(event.clientX, event.clientY)
    return
  }
  await zoomOutPreview(event.clientX, event.clientY)
}

const onPointerDown = (event: MouseEvent) => {
  if (fitToWindow.value || !previewContainerRef.value) return
  event.preventDefault()
  dragging.value = true
  dragStartX.value = event.clientX
  dragStartY.value = event.clientY
  scrollStartLeft.value = previewContainerRef.value.scrollLeft
  scrollStartTop.value = previewContainerRef.value.scrollTop
}

const onPointerMove = (event: MouseEvent) => {
  if (!dragging.value || !previewContainerRef.value) return
  previewContainerRef.value.scrollLeft = scrollStartLeft.value - (event.clientX - dragStartX.value)
  previewContainerRef.value.scrollTop = scrollStartTop.value - (event.clientY - dragStartY.value)
}

const stopDragging = () => {
  dragging.value = false
}
</script>

<template>
  <div class="glass-card preview-shell">
    <div class="preview-header">
      <div class="preview-header-top">
        <h3 class="section-title">图像核查</h3>
      </div>

      <div class="preview-nav">
        <el-select v-model="currentId" placeholder="选择图像" class="preview-select">
          <el-option
            v-for="option in imageOptions"
            :key="option.value"
            :label="option.label"
            :value="option.value"
          />
        </el-select>
        <div class="preview-nav-actions">
          <el-button class="preview-nav-button" plain @click="goToImage(-1)" :disabled="currentIndex <= 0">上一张</el-button>
          <el-button class="preview-nav-button" plain @click="goToImage(1)" :disabled="currentIndex < 0 || currentIndex >= images.length - 1">下一张</el-button>
          <span class="preview-nav-counter">第 {{ currentIndex + 1 }} / {{ images.length }} 张</span>
        </div>
      </div>
    </div>

    <div v-if="currentImage" class="inspection-grid">
      <section class="stage-card">
        <div class="stage-header">
          <div>
            <h4 class="stage-title">{{ currentArtifact?.label ?? '结果预览' }}</h4>
          </div>
        </div>

        <div
          class="image-frame stage-frame clickable-frame"
          :class="{ 'stage-frame--footer': isFooterArtifact }"
          @click="openPreview(`${currentArtifact?.label ?? '预览'} · ${currentImage.image_name}`, currentArtifact?.url)"
        >
          <img
            v-if="currentArtifact?.url"
            :src="currentArtifact.url"
            :alt="currentArtifact.label"
            :class="{ 'fit-contain': currentArtifact.fit === 'contain' }"
          />
          <el-empty v-else description="当前图层暂无结果" />
        </div>

      </section>

      <aside class="artifact-rail">
        <section v-if="activeModeOptions.length > 1" class="artifact-panel">
          <div class="panel-heading panel-heading--compact">
            <div>
              <span class="panel-eyebrow">模式</span>
              <h4>分割来源</h4>
            </div>
          </div>
          <el-segmented v-model="selectedMode" :options="activeModeOptions" class="mode-switch" />
        </section>

        <section v-for="group in artifactGroups" :key="group.title" class="artifact-panel">
          <div class="panel-heading panel-heading--compact">
            <div>
              <span class="panel-eyebrow">图层</span>
              <h4>{{ group.title }}</h4>
            </div>
            <span class="panel-count">{{ group.items.length }} 项</span>
          </div>

          <div class="artifact-list">
            <button
              v-for="item in group.items"
              :key="item.key"
              type="button"
              class="artifact-button"
              :class="{ 'is-active': currentArtifact?.key === item.key }"
              @click="selectArtifact(item.key)"
            >
              <strong>{{ item.label }}</strong>
            </button>
          </div>
        </section>
      </aside>
    </div>

    <el-empty v-else :description="props.emptyDescription" />

    <el-dialog
      v-model="previewVisible"
      :title="previewTitle"
      width="86vw"
      top="4vh"
      :lock-scroll="false"
      destroy-on-close
      append-to-body
      class="preview-dialog"
    >
      <div class="preview-toolbar-actions">
        <el-button size="small" @click="resetToFit">适配窗口</el-button>
        <el-button size="small" @click="showOriginalSize">查看原始尺寸</el-button>
        <el-button size="small" @click="() => zoomOutPreview()">-</el-button>
        <span class="zoom-indicator">{{ Math.round(zoom * 100) }}%</span>
        <el-button size="small" @click="() => zoomInPreview()">+</el-button>
      </div>
      <div
        ref="previewContainerRef"
        class="preview-dialog-body"
        :class="{ 'is-draggable': !fitToWindow }"
        @wheel="onWheelZoom"
        @mousedown="onPointerDown"
        @mousemove="onPointerMove"
        @mouseup="stopDragging"
        @mouseleave="stopDragging"
      >
        <div class="preview-stage" :class="{ 'preview-stage-fit': fitToWindow }">
          <img
            ref="previewImageRef"
            :src="previewImageUrl"
            :alt="previewTitle"
            class="preview-dialog-image"
            :style="previewImageStyle"
            draggable="false"
            @load="onPreviewImageLoad"
            @dragstart.prevent
          />
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.preview-shell {
  padding: 20px;
  display: grid;
  gap: 16px;
  min-width: 0;
}

.preview-header {
  display: grid;
  gap: 14px;
  min-width: 0;
}

.preview-header-top {
  min-width: 0;
}

.preview-header .section-title {
  margin: 0;
  white-space: nowrap;
}

.preview-nav {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  width: 100%;
  min-width: 0;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(31, 40, 48, 0.06);
}

.preview-nav-button {
  min-height: 42px;
  padding-inline: 18px;
  border-radius: 14px;
  font-weight: 600;
}

.preview-select {
  flex: 1 1 320px;
  min-width: 0;
}

.preview-nav-actions {
  margin-left: auto;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
}

.preview-nav-counter {
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.inspection-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.48fr) minmax(280px, 360px);
  gap: 16px;
  min-width: 0;
  align-items: start;
}

.stage-card,
.artifact-panel {
  min-width: 0;
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.56);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.stage-card {
  padding: 18px;
  display: grid;
  gap: 14px;
}

.stage-header {
  display: flex;
  align-items: flex-start;
}

.stage-title {
  margin: 0 0 6px;
  font-size: 24px;
  line-height: 1.18;
}

.stage-frame {
  display: block;
  width: 100%;
  max-width: 100%;
  min-height: 0;
  background: linear-gradient(180deg, rgba(18, 23, 28, 0.96), rgba(31, 37, 42, 0.92));
}

.stage-frame img {
  width: 100%;
  max-width: 100%;
  height: auto;
  object-fit: contain;
  object-position: center;
  background: rgba(20, 24, 28, 0.92);
}

.stage-frame--footer {
  min-height: 0;
}

.stage-frame img.fit-contain,
.stage-frame--footer img {
  object-fit: contain;
  background: rgba(20, 24, 28, 0.92);
}

.artifact-rail {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}

.artifact-panel {
  padding: 16px;
}

.panel-heading--compact {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 12px;
}

.panel-heading--compact h4 {
  margin: 4px 0 0;
  font-size: 17px;
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

.panel-count {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}

.artifact-list {
  display: grid;
  gap: 10px;
}

.artifact-button {
  display: grid;
  gap: 4px;
  width: 100%;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(31, 40, 48, 0.08);
  background: rgba(255, 255, 255, 0.72);
  color: inherit;
  text-align: left;
  cursor: pointer;
  transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
}

.artifact-button:hover {
  transform: translateY(-1px);
  border-color: rgba(23, 96, 135, 0.16);
  box-shadow: 0 14px 22px rgba(23, 96, 135, 0.08);
}

.artifact-button strong {
  font-size: 15px;
  line-height: 1.3;
}

.artifact-button.is-active {
  border-color: rgba(23, 96, 135, 0.18);
  background: linear-gradient(135deg, rgba(23, 96, 135, 0.12), rgba(47, 139, 192, 0.08));
}

.mode-switch {
  width: 100%;
}

.clickable-frame {
  cursor: zoom-in;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.clickable-frame:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 26px rgba(23, 96, 135, 0.12);
}

.preview-dialog-body {
  min-height: 60vh;
  max-height: 78vh;
  overflow: auto;
  background: rgba(20, 24, 28, 0.95);
  border-radius: 14px;
}

.preview-stage {
  min-width: 100%;
  min-height: 60vh;
  display: inline-flex;
  align-items: flex-start;
  justify-content: flex-start;
}

.preview-stage-fit {
  align-items: center;
  justify-content: center;
}

.preview-dialog-image {
  display: block;
  object-fit: contain;
  user-select: none;
  -webkit-user-drag: none;
}

.preview-toolbar-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.zoom-indicator {
  min-width: 56px;
  text-align: center;
  font-size: 13px;
  color: #f6efe2;
}

.is-draggable {
  cursor: default;
}

@media (max-width: 1320px) {
  .inspection-grid {
    grid-template-columns: 1fr;
  }

  .artifact-rail {
    order: -1;
  }

}

@media (max-width: 920px) {
  .stage-header {
    flex-direction: column;
    align-items: stretch;
  }

  .preview-nav {
    align-items: stretch;
  }

  .preview-nav-actions {
    width: 100%;
    margin-left: 0;
    justify-content: stretch;
  }

  .preview-nav-button {
    flex: 1 1 0;
    min-width: 0;
  }

  .preview-nav-counter {
    width: 100%;
    text-align: right;
  }
}
</style>
